import logging
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers import logging as transformers_logging  # AutoConfig,; AutoModel,

from .modules import LinkPredHead

# from .peft_model import PeftModel

logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_error()


class Link_E5_model(nn.Module):
    def __init__(self, args):
        super(Link_E5_model, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = args.pretrained_repo
        logger.warning(f"inherit model weights from {pretrained_repo}")
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.header_dropout_prob = args.header_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config, add_pooling_layer=False)
        self.head = LinkPredHead(config)
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
        out = self.average_pool(bert_out.last_hidden_state, att_mask)
        if return_hidden:
            sentence_embeddings = F.normalize(out, p=2, dim=1)
            return out, sentence_embeddings
        else:
            return out

    def link_predict(self, x_i, x_j):
        return self.head(x_i, x_j)


class Link_Sentence_Transformer(nn.Module):
    def __init__(self, args):
        super(Link_Sentence_Transformer, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = args.pretrained_repo
        logger.warning(f"inherit model weights from {pretrained_repo}")
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.header_dropout_prob = args.header_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config, add_pooling_layer=False)
        self.head = LinkPredHead(config)
        if args.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask, labels=None):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.mean_pooling(bert_out, att_mask)
        out = F.normalize(out, p=2, dim=1)
        return out

    def link_predict(self, x_i, x_j):
        return self.head(x_i, x_j)
