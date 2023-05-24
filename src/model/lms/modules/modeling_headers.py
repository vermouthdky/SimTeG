import torch
import torch.functional as F
from torch import nn

from .modeling_adapter_deberta import ContextPooler, StableDropout


class LinkPredHead(nn.Module):
    def __init__(self, config):
        super(LinkPredHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        dropout = config.header_dropout_prob if config.header_dropout_prob is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return torch.sigmoid(x)


class SentenceClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.header_dropout_prob if config.header_dropout_prob is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DebertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(output_dim, config.num_labels)
        dropout = config.header_dropout_prob
        self.dropout = StableDropout(dropout)

    def forward(self, hidden_states):
        pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.header_dropout_prob if config.header_dropout_prob is not None else config.hidden_dropout_prob
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
