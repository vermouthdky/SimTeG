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

from .modules import AdapterDebertaModel, AdapterRobertaModel
from .modules.modeling_adapter_deberta import ContextPooler, StableDropout

logging.set_verbosity_error()


class DebertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(output_dim, config.num_labels)
        dropout = getattr(config, "cls_dropout", config.classifier_dropout)
        self.dropout = StableDropout(dropout)

    def forward(self, hidden_states):
        pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# NOTE: is this specific to Roberta or can it be used for other models?
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Roberta(nn.Module):
    def __init__(self, args):
        super(Roberta, self).__init__()
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = args.num_labels
        config.classifier_dropout = args.gnn_dropout  # NOTE TEMP
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
        config.classifier_dropout = args.gnn_dropout
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
        config.classifier_dropout = args.gnn_dropout
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
        config.classifier_dropout = args.gnn_dropout  # NOTE TEMP
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
