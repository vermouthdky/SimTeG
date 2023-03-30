import gc
import logging
import os

import evaluate
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.transforms import SIGN
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments as HugTrainerAugments

from ..model import get_model_class
from ..utils import is_dist
from .trainer import Trainer

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None, x_emb=None, x0=None, labels=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
            "x_emb": x_emb,
            "x0": x0,
            "labels": labels.view(-1, 1),
        }

    def __len__(self):
        return self.data["labels"].size(0)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][idx]
        return batch_data


class OneIterTrainer(HugTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        NOTE: add KL divergence here
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        should_compute_kl = False
        if "x0" in inputs:
            x0 = inputs.pop("x0")
            if x0 is not None:
                should_compute_kl = True

        if return_outputs:
            logits, hidden_features = model(**inputs, return_hidden=True)
        else:
            logits = model(**inputs, return_hidden=False)

        loss_op = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor)
        loss = loss_op(logits, labels.view(-1))
        if should_compute_kl:
            kl_loss = F.kl_div(
                input=F.log_softmax(hidden_features, dim=-1),
                target=F.softmax(x0, dim=-1),
                reduction="batchmean",
            )
            kl_loss *= 2**self.args.kl_loss_temp
            loss += self.args.kl_loss_weight * kl_loss
        return (loss, {"logits": logits, "hidden_features": hidden_features}) if return_outputs else loss


class TrainingArguments(HugTrainerAugments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_loss_weight = kwargs.pop("kl_loss_weight", 1.0)
        self.kl_loss_temp = kwargs.pop("kl_loss_temp", 0.0)


class GBert_Trainer(Trainer):
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        super().__init__(args, data, split_idx, evaluator, **kwargs)

    def ckpt_name(self, iter, module_to_train: str):
        assert module_to_train in ["gnn", "lm"]
        return f"iter_{iter}_{module_to_train}.ckpt"

    def _get_dataset(self, mode, module_to_train: str):
        assert mode in ["train", "valid", "test", "all"]
        assert module_to_train in ["gnn", "lm"]
        if module_to_train == "lm":
            dataset = Dataset(
                self.data.input_ids,
                self.data.attention_mask,
                self.data.x_emb if self.iter > 0 else None,
                self.data.x if self.iter > 0 else None,
                self.data.y,
            )
        else:
            dataset = Dataset(x_emb=self.data.x_emb, labels=self.data.y)
        return dataset if mode == "all" else torch.utils.data.Subset(dataset, self.split_idx[mode])

    def _prepare_dataset(self, module_to_train: str):
        assert module_to_train in ["gnn", "lm"]
        self.train_set = self._get_dataset("train", module_to_train)
        self.valid_set = self._get_dataset("valid", module_to_train)

    def _prepare_model(self, module_to_train: str):
        assert module_to_train in ["gnn", "lm"]
        model_class = get_model_class("GBert", self.args.use_adapter)
        model = model_class(self.args, self.iter, module_to_train)
        if self.args.fix_gnn and module_to_train == "lm":
            model.gnn_model.requires_grad_(False)
        return model

    def compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(-1)
        return metric.compute(predictions=predictions, references=labels)

    def _prepare_trainer(self, module_to_train: str):
        assert module_to_train in ["gnn", "lm"]
        if module_to_train == "lm":
            training_args = self._lm_training_arguments()
        else:
            training_args = self._gnn_training_argument()
        return OneIterTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.valid_set,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    def _gnn_training_argument(self):
        # prepare training args
        total_batch_size = self.world_size * self.args.batch_size * self.args.accum_interval
        eval_steps = self.args.eval_patience // total_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        warmup_steps = self.args.warmup_ratio * train_steps
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.args.gnn_lr,
            weight_decay=self.args.weight_decay,
            load_best_model_at_end=True,
            gradient_accumulation_steps=1,
            label_smoothing_factor=self.args.gnn_label_smoothing,
            save_total_limit=1,
            per_device_train_batch_size=self.args.gnn_batch_size,
            per_device_eval_batch_size=self.args.gnn_eval_batch_size,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            num_train_epochs=self.args.gnn_epochs,
            local_rank=self.rank,
            dataloader_num_workers=1,
            ddp_find_unused_parameters=False,
            kl_loss_weight=self.args.kl_loss_weight,
            kl_loss_temp=self.args.kl_loss_temp,
        )
        return training_args

    def _lm_training_arguments(self):
        # prepare training args
        total_batch_size = self.world_size * self.args.batch_size * self.args.accum_interval
        eval_steps = self.args.eval_patience // total_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        warmup_steps = self.args.warmup_ratio * train_steps
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            # overwrite_output_dir=True,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.args.accum_interval,
            label_smoothing_factor=self.args.label_smoothing,
            save_total_limit=1,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            num_train_epochs=self.args.epochs,
            local_rank=self.rank,
            dataloader_num_workers=1,
            ddp_find_unused_parameters=False,
            kl_loss_weight=self.args.kl_loss_weight,
            kl_loss_temp=self.args.kl_loss_temp,
        )
        return training_args

    def train_lm_once(self, iter):
        dist.barrier()
        self.model = self._prepare_model(module_to_train="lm")
        self.train_set, self.valid_set = self._prepare_dataset()
        self.trainer = self._prepare_trainer()
        if self.trial is not None:
            self.trainer._hp_search_setup(self.trial)
        train_output = self.trainer.train()  # none if not using optuna
        if iter > 0:
            self.save_model(self.model.gnn_model, self.ckpt_name(iter, "gnn"))
        self.save_model(self.model.bert_model, self.ckpt_name(iter, "lm"))
        global_step, train_dict = train_output.global_step, train_output.metrics
        train_dict["global_step"] = global_step
        self.trainer.save_metrics("train", train_dict)
        # print train_dict
        logger.critical("".join("{}:{} ".format(k, v) for k, v in train_dict.items()))
        eval_dict = self.trainer.evaluate()
        self.trainer.save_metrics("eval", eval_dict)
        logger.critical("".join("{}:{} ".format(k, v) for k, v in eval_dict.items()))
        gc.collect()
        torch.cuda.empty_cache()
        return eval_dict["eval_accuracy"]

    def train_gnn_once(self, iter):
        dist.barrier()
        self.model = self._prepare_model(module_to_train="gnn")
        self.train_set, set.valid_set = self._prepare_dataset()
        self.trainer = self._prepare_trainer()

    def train(self):
        best_valid_acc = 0.0
        best_count = 0
        for iter in range(self.args.num_iterations):
            self.iter = iter
            logger.critical(f"\n*************** Start iter {iter} training ***************\n")

            if self.iter == 0:
                ckpt_path = os.path.join(self.args.ckpt_dir, "iter_0")
                if self.args.use_cache and os.path.exists(ckpt_path):
                    logger.warning(f"iter {iter} has been trained, use cached ckpt instead!")
                else:
                    lm_valid_acc = self.train_lm_once(iter)
                    valid_acc = self.train_gnn_once(iter)
                    logger.critical("Iter {} Best valid acc: {:.4f}".format(iter, valid_acc))
            else:
                if self.args.inherit:
                    logger.warning("using inherited weights from the last iteration!")
                    if iter - 1 > 0:
                        self.load_model(self.model.gnn_model, f"iter_{iter - 1}_gnn.pt")
                    self.load_model(self.model.bert_model, f"iter_{iter - 1}_lm.pt")
                else:
                    logger.warning("train the model from scratch instead of inheriting!")
                valid_acc = self.train_lm_once(iter)

            logger.warning(f"\n*************** Start iter {iter} testing and Postprocessing ***************\n")
            # NOTE inference for SLE and propogation
            hidden_features, results = self.inference_and_evaluate()

            valid_acc = results["valid_acc"]
            if valid_acc > best_valid_acc:
                self.save_model(self.model, "best.pt")
                best_valid_acc = valid_acc
                best_count = 0
            else:
                best_count += 1
                if best_count >= 2:
                    return best_valid_acc

            self.iter += 1
            # preserve data.x for KL divergence loss
            self.data.x = hidden_features
            num_layers = self.args.gnn_num_layers
            self.data = SIGN(num_layers)(self.data)
            self.data.x_emb = torch.cat([self.data[f"x{i}"] for i in range(1, num_layers + 1)], dim=-1)
            for i in range(1, num_layers + 1):
                del self.data[f"x{i}"]
            gc.collect()
            torch.cuda.empty_cache()
        return best_valid_acc
