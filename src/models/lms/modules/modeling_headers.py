import torch
from torch import nn

from .modeling_adapter_deberta import ContextPooler, StableDropout


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
