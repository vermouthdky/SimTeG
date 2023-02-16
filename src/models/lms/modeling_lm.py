import os

import torch
from torch import nn
from transformers import (
    DebertaConfig,
    DebertaModel,
    RobertaConfig,
    RobertaModel,
    logging,
)

from .modules import (
    AdapterDebertaModel,
    AdapterRobertaModel,
    DebertaClassificationHead,
    RobertaClassificationHead,
)

logging.set_verbosity_error()


class Roberta(nn.Module):
    def __init__(self, args):
        super(Roberta, self).__init__()
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = args.num_labels
        config.classifier_dropout = args.header_dropout_prob
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = RobertaModel.from_pretrained("roberta-base", config=config, add_pooling_layer=False)
        self.head = RobertaClassificationHead(config)

    def forward(self, input_ids, att_mask, return_bert_out=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_bert_out:
            return out, bert_out[0][:, 0, :]
        else:
            return out


class Deberta(nn.Module):
    def __init__(self, args):
        super(Deberta, self).__init__()
        pretrained_repo = "microsoft/deberta-base"
        config = DebertaConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.classifier_dropout = args.header_dropout_prob
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = DebertaModel.from_pretrained(pretrained_repo, config=config)
        self.head = DebertaClassificationHead(config)

    def forward(self, input_ids, att_mask, return_bert_out=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_bert_out:
            return out, bert_out[0][:, 0, :]
        else:
            return out


class AdapterDeberta(nn.Module):
    def __init__(self, args):
        super(AdapterDeberta, self).__init__()
        pretrained_repo = "microsoft/deberta-base"
        config = DebertaConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.classifier_dropout = args.header_dropout_prob
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AdapterDebertaModel.from_pretrained(pretrained_repo, config=config)
        self.head = DebertaClassificationHead(config)

    def forward(self, input_ids, att_mask, return_bert_out=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_bert_out:
            return out, bert_out[0][:, 0, :]
        else:
            return out


class AdapterRoberta(nn.Module):
    def __init__(self, args):
        super(AdapterRoberta, self).__init__()
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = args.num_labels
        config.classifier_dropout = args.header_dropout_prob
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        # init modules
        self.bert_model = AdapterRobertaModel.from_pretrained("roberta-base", config=config, add_pooling_layer=False)
        self.head = RobertaClassificationHead(config)

    def forward(self, input_ids, att_mask, return_bert_out=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_bert_out:
            return out, bert_out[0][:, 0, :]
        else:
            return out
