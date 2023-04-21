import logging
import os

import torch
from torch import nn
from transformers import DebertaConfig, RobertaConfig, RobertaModel
from transformers import logging as transformers_logging  # AutoConfig,; AutoModel,

from .modules import (
    AdapterDebertaModel,
    AdapterRobertaModel,
    DebertaClassificationHead,
    DebertaModel,
    RobertaClassificationHead,
)

logger = logging.getLogger(__name__)

transformers_logging.set_verbosity_error()


class Roberta(nn.Module):
    def __init__(self, args):
        super(Roberta, self).__init__()
        transformers_logging.set_verbosity_error()
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = RobertaModel.from_pretrained("roberta-base", config=config, add_pooling_layer=False)
        self.head = RobertaClassificationHead(config)

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_hidden:
            return out, bert_out[0][:, 0, :]
        else:
            return out


class Deberta(nn.Module):
    def __init__(self, args):
        super(Deberta, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = "microsoft/deberta-base"
        config = DebertaConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = DebertaModel.from_pretrained(pretrained_repo, config=config)
        self.head = DebertaClassificationHead(config)

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_hidden:
            return out, bert_out[0][:, 0, :]
        else:
            return out


class AdapterDeberta(nn.Module):
    def __init__(self, args):
        super(AdapterDeberta, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = "microsoft/deberta-base"
        config = DebertaConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.adapter_hidden_size = args.adapter_hidden_size
        if not args.use_default_config:
            config.header_dropout_prob = args.header_dropout_prob
            config.hidden_dropout_prob = args.hidden_dropout_prob
            config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AdapterDebertaModel.from_pretrained(pretrained_repo, config=config)
        self.head = DebertaClassificationHead(config)
        is_correct = self._check_if_adapter_is_correct()
        if not is_correct:
            raise ValueError("Adapter is not correctly initialized!")
        else:
            logger.warning("Adapter is correctly initialized!")

    def _check_if_adapter_is_correct(self):
        for name, param in self.bert_model.named_parameters():
            if param.requires_grad and "adapter" not in name:
                return False
            elif not param.requires_grad and "adapter" in name:
                return False
        return True

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        # here we set labels here to fix the bug that raise in huggingface trainer class
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_hidden:
            return out, bert_out[0][:, 0, :]
        else:
            return out


class AdapterRoberta(nn.Module):
    def __init__(self, args):
        super(AdapterRoberta, self).__init__()
        transformers_logging.set_verbosity_error()
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.adapter_hidden_size = args.adapter_hidden_size
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AdapterRobertaModel.from_pretrained("roberta-base", config=config, add_pooling_layer=False)
        self.head = RobertaClassificationHead(config)

    def forward(self, input_ids, att_mask, labels=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_hidden:
            return out, bert_out[0][:, 0, :]
        else:
            return out
