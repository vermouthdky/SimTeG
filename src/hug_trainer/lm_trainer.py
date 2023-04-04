import gc
import logging
import os
import os.path as osp
import shutil

import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.transforms import SIGN
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments as HugTrainingArguments

from ..model import get_model_class
from ..utils import EmbeddingHandler, is_dist
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

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class InnerTrainer(HugTrainer):
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

        if return_outputs or should_compute_kl:
            logits, hidden_features = model(**inputs, return_hidden=True)
        else:
            logits = model(**inputs, return_hidden=False)

        loss_op = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor, reduce="mean")
        loss = loss_op(logits, labels.view(-1))
        if should_compute_kl:
            kl_loss = F.kl_div(
                input=F.log_softmax(hidden_features, dim=-1),
                target=F.softmax(x0, dim=-1),
                reduction="batchmean",
            )
            kl_loss = self.args.kl_loss_weight * kl_loss * (2**self.args.kl_loss_temp)
            loss = loss + kl_loss
        return (loss, {"logits": logits, "hidden_features": hidden_features}) if return_outputs else loss


class TrainingArguments(HugTrainingArguments):
    def __init__(self, *args, **kwargs):
        self.kl_loss_weight = kwargs.pop("kl_loss_weight", 1.0)
        self.kl_loss_temp = kwargs.pop("kl_loss_temp", 0.0)
        super().__init__(*args, **kwargs)


class LMTrainer(Trainer):
    def ckpt_path(self, iter, stage="lm", module="lm"):
        return osp.join(self.args.ckpt_dir, f"iter_{iter}_{stage}_{module}.pt")

    def _get_dataset(self, mode):
        assert mode in ["train", "valid", "test", "all"]
        should_feed_x0 = self.iter > 0 and mode == "train" and self.args.compute_kl_loss
        dataset = Dataset(
            self.data.input_ids,
            self.data.attention_mask,
            self.data.x_emb if self.iter > 0 else None,
            self.data.x if should_feed_x0 else None,
            self.data.y,
        )
        return dataset if mode == "all" else torch.utils.data.Subset(dataset, self.split_idx[mode])

    def _prepare_datset(self):
        return self._get_dataset("train"), self._get_dataset("valid")

    def _prepare_model(self):
        model_class = get_model_class(self.args.model_type, self.args.use_adapter)
        model = model_class(self.args, self.iter, "lm")
        if self.args.fix_gnn and self.iter > 0:
            model.gnn_model.requires_grad_(False)
        return model

    def _compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(-1)
        return metric.compute(predictions=predictions, references=labels)

    def _prepare_trainer(self):
        # prepare training args
        total_batch_size = self.world_size * self.args.batch_size * self.args.accum_interval
        eval_steps = self.args.eval_patience // total_batch_size
        train_steps = len(self.train_set) // total_batch_size + 1
        warmup_steps = self.args.warmup_ratio * train_steps
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            # eval_accumulation_steps=10,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            gradient_accumulation_steps=self.args.accum_interval,
            label_smoothing_factor=self.args.label_smoothing,
            save_total_limit=1,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            num_train_epochs=self.args.epochs,
            local_rank=self.rank,
            dataloader_num_workers=12,
            ddp_find_unused_parameters=False,
            kl_loss_weight=self.args.kl_loss_weight,
            kl_loss_temp=self.args.kl_loss_temp,
        )
        return InnerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.valid_set,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    def inference(self, dataset, embs_path):
        x_embs_name = f"iter_{self.iter}_x_embs.pt"
        logits_name = f"iter_{self.iter}_lm_logits.pt"
        emb_handler = EmbeddingHandler(embs_path)
        if self.args.use_cache and emb_handler.has([x_embs_name, logits_name]):
            x_embs = emb_handler.load(x_embs_name)
            logits_embs = emb_handler.load(logits_name)
            if isinstance(x_embs, np.ndarray):
                x_embs, logits_embs = torch.from_numpy(x_embs), torch.from_numpy(logits_embs)
        else:
            eval_output = self.trainer.predict(dataset)
            logits_embs, x_embs = eval_output.predictions[0], eval_output.predictions[1]
            logits_embs, x_embs = torch.from_numpy(logits_embs), torch.from_numpy(x_embs)
            emb_handler.save(x_embs, x_embs_name)
            emb_handler.save(logits_embs, logits_name)
            logger.info(f"save the logits of {self.args.lm_type} to {osp.join(embs_path, logits_name)}")
            logger.info(f"save the hidden features of {self.args.lm_type} to {osp.join(embs_path, x_embs_name)}")
        return logits_embs, x_embs

    def train_once(self, iter):
        dist.barrier()
        self.model = self._prepare_model()
        if iter > 0:
            self.load_model(self.model.gnn_model, self.ckpt_path(iter - 1, "gnn", "gnn"))
            if self.args.inherit:
                self.load_model(self.model.bert_model, self.ckpt_path(iter - 1, "lm", "lm"))

        self.train_set, self.valid_set = self._prepare_datset()
        self.trainer = self._prepare_trainer()
        if self.trial is not None:
            self.trainer._hp_search_setup(self.trial)

        train_output = self.trainer.train()
        self.save_model(self.model.bert_model, self.ckpt_path(iter, "lm", "lm"))
        if iter > 0:
            self.save_model(self.model.gnn_model, self.ckpt_path(iter, "lm", "gnn"))

        global_step, train_dict = train_output.global_step, train_output.metrics
        train_dict["global_step"] = global_step
        self.trainer.save_metrics("train", train_dict)
        # print train_dict
        logger.critical("".join("{}:{} ".format(k, v) for k, v in train_dict.items()))
        gc.collect()
        torch.cuda.empty_cache()
