import logging
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType
from torch import nn
from transformers import AutoConfig, AutoModel, DebertaV2Config, DebertaV2Model
from transformers import logging as transformers_logging  # AutoConfig,; AutoModel,

from .modules import DebertaClassificationHead, SentenceClsHead

# from .peft_model import PeftModel

logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_error()


class T5_model(nn.Module):
    def __init__(self, args):
        super(T5_model, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = args.pretrained_repo
        logger.warning(f"inherit model weights from {pretrained_repo}")
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config)
        self.head = SentenceClsHead(config)
        if args.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.peft_r,
                lora_alpha=args.peft_lora_alpha,
                lora_dropout=args.peft_lora_dropout,
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)
        if int(os.getenv("RANK", -1)):
            __import__("ipdb").set_trace()
        else:
            torch.distributed.barrier()
        out = self.head(sentence_embeddings)

        if return_hidden:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return out, sentence_embeddings
        else:
            return out


class E5_model(nn.Module):
    def __init__(self, args):
        super(E5_model, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = args.pretrained_repo
        logger.warning(f"inherit model weights from {pretrained_repo}")
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config, add_pooling_layer=False)
        self.head = SentenceClsHead(config)
        if args.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.peft_r,
                lora_alpha=args.peft_lora_alpha,
                lora_dropout=args.peft_lora_dropout,
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()

    def average_pool(self, last_hidden_states, attention_mask):  # for E5_model
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.average_pool(bert_out.last_hidden_state, att_mask)
        out = self.head(sentence_embeddings)

        if return_hidden:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return out, sentence_embeddings
        else:
            return out


class Sentence_Transformer(nn.Module):
    def __init__(self, args):
        super(Sentence_Transformer, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = args.pretrained_repo
        logger.warning(f"inherit model weights from {pretrained_repo}")
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config, add_pooling_layer=False)
        self.head = SentenceClsHead(config)
        if args.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.peft_r,
                lora_alpha=args.peft_lora_alpha,
                lora_dropout=args.peft_lora_dropout,
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)
        out = self.head(sentence_embeddings)

        if return_hidden:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return out, sentence_embeddings
        else:
            return out


class Deberta(nn.Module):
    def __init__(self, args):
        super(Deberta, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = args.pretrained_repo
        assert pretrained_repo in ["microsoft/deberta-v2-xxlarge"]
        config = DebertaV2Config.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        if not args.use_default_config:
            config.hidden_dropout_prob = args.hidden_dropout_prob
            config.attention_probs_dropout_prob = args.attention_dropout_prob
        else:
            logger.warning("Using default config")
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = DebertaV2Model.from_pretrained(pretrained_repo, config=config)
        self.head = DebertaClassificationHead(config)
        if args.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.peft_r,
                lora_alpha=args.peft_lora_alpha,
                lora_dropout=args.peft_lora_dropout,
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_hidden:
            return out, bert_out[0][:, 0, :]
        else:
            return out


# class AdapterDebertaV3(nn.Module):
#     def __init__(self, args):
#         super(AdapterDebertaV3, self).__init__()
#         transformers_logging.set_verbosity_error()
#         pretrained_repo = args.pretrained_repo
#         assert pretrained_repo in ["microsoft/deberta-v3-base", "microsoft/deberta-v3-large"]
#         config = DebertaV2Config.from_pretrained(pretrained_repo)
#         config.num_labels = args.num_labels
#         config.adapter_hidden_size = args.adapter_hidden_size
#         config.header_dropout_prob = args.header_dropout_prob
#         config.hidden_dropout_prob = args.hidden_dropout_prob
#         config.attention_probs_dropout_prob = args.attention_dropout_prob
#         config.save_pretrained(save_directory=args.output_dir)
#         # init modules
#         self.bert_model = DebertaV2Model.from_pretrained(pretrained_repo, config=config)
#         self.head = DebertaClassificationHead(config)
#         is_correct = self._check_if_adapter_is_correct()
#         if not is_correct:
#             raise ValueError("Adapter is not correctly initialized!")
#         else:
#             logger.warning("Adapter is correctly initialized!")

#     def _check_if_adapter_is_correct(self):
#         for name, param in self.bert_model.named_parameters():
#             if param.requires_grad and "adapter" not in name:
#                 return False
#             elif not param.requires_grad and "adapter" in name:
#                 return False
#         return True

#     def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
#         bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
#         out = self.head(bert_out[0])
#         if return_hidden:
#             return out, bert_out[0][:, 0, :]
#         else:
#             return out


# class Deberta(nn.Module):
#     def __init__(self, args):
#         super(Deberta, self).__init__()
#         transformers_logging.set_verbosity_error()
#         pretrained_repo = args.pretrained_repo
#         assert pretrained_repo in ["microsoft/deberta-base", "microsoft/deberta-large"]
#         config = DebertaConfig.from_pretrained(pretrained_repo)
#         config.num_labels = args.num_labels
#         config.header_dropout_prob = args.header_dropout_prob
#         if not args.use_default_config:
#             config.hidden_dropout_prob = args.hidden_dropout_prob
#             config.attention_probs_dropout_prob = args.attention_dropout_prob
#         else:
#             logger.warning("Using default config")
#         config.save_pretrained(save_directory=args.output_dir)
#         # init modules
#         self.bert_model = DebertaModel.from_pretrained(pretrained_repo, config=config)
#         self.head = DebertaClassificationHead(config)

#     def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
#         bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
#         out = self.head(bert_out[0])
#         if return_hidden:
#             return out, bert_out[0][:, 0, :]
#         else:
#             return out


# class AdapterDeberta(nn.Module):
#     def __init__(self, args):
#         super(AdapterDeberta, self).__init__()
#         transformers_logging.set_verbosity_error()
#         pretrained_repo = args.pretrained_repo
#         assert pretrained_repo in ["microsoft/deberta-base", "microsoft/deberta-large"]
#         config = DebertaConfig.from_pretrained(pretrained_repo)
#         config.num_labels = args.num_labels
#         config.adapter_hidden_size = args.adapter_hidden_size
#         config.header_dropout_prob = args.header_dropout_prob
#         config.hidden_dropout_prob = args.hidden_dropout_prob
#         config.attention_probs_dropout_prob = args.attention_dropout_prob
#         config.save_pretrained(save_directory=args.output_dir)
#         # init modules
#         self.bert_model = AdapterDebertaModel.from_pretrained(pretrained_repo, config=config)
#         self.head = DebertaClassificationHead(config)
#         is_correct = self._check_if_adapter_is_correct()
#         if not is_correct:
#             raise ValueError("Adapter is not correctly initialized!")
#         else:
#             logger.warning("Adapter is correctly initialized!")

#     def _check_if_adapter_is_correct(self):
#         for name, param in self.bert_model.named_parameters():
#             if param.requires_grad and "adapter" not in name:
#                 return False
#             elif not param.requires_grad and "adapter" in name:
#                 return False
#         return True

#     def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
#         # here we set labels here to fix the bug that raise in huggingface trainer class
#         bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
#         out = self.head(bert_out[0])
#         if return_hidden:
#             return out, bert_out[0][:, 0, :]
#         else:
#             return out


# class Roberta(nn.Module):
#     def __init__(self, args):
#         super(Roberta, self).__init__()
#         transformers_logging.set_verbosity_error()
#         config = RobertaConfig.from_pretrained(args.pretrained_repo)
#         config.num_labels = args.num_labels
#         config.header_dropout_prob = args.header_dropout_prob
#         config.hidden_dropout_prob = args.hidden_dropout_prob
#         config.attention_probs_dropout_prob = args.attention_dropout_prob
#         config.save_pretrained(save_directory=args.output_dir)
#         # init modules
#         self.bert_model = RobertaModel.from_pretrained(args.pretrained_repo, config=config, add_pooling_layer=False)
#         self.head = RobertaClassificationHead(config)

#     def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
#         bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
#         out = self.head(bert_out[0])
#         if return_hidden:
#             return out, bert_out[0][:, 0, :]
#         else:
#             return out


# class AdapterRoberta(nn.Module):
#     def __init__(self, args):
#         super(AdapterRoberta, self).__init__()
#         transformers_logging.set_verbosity_error()
#         config = RobertaConfig.from_pretrained(args.pretrained_repo)
#         config.num_labels = args.num_labels
#         config.header_dropout_prob = args.header_dropout_prob
#         config.hidden_dropout_prob = args.hidden_dropout_prob
#         config.attention_probs_dropout_prob = args.attention_dropout_prob
#         config.adapter_hidden_size = args.adapter_hidden_size
#         config.save_pretrained(save_directory=args.output_dir)
#         # init modules
#         self.bert_model = AdapterRobertaModel.from_pretrained(
#             args.pretrained_repo, config=config, add_pooling_layer=False
#         )
#         self.head = RobertaClassificationHead(config)

#     def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
#         bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
#         out = self.head(bert_out[0])
#         if return_hidden:
#             return out, bert_out[0][:, 0, :]
#         else:
#             return out
