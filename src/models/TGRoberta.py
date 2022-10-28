import logging
import os

import numpy as np
import torch.nn.functional as F
from torch import nn, tanh
from torch_geometric.nn import GATConv
from transformers.activations import gelu
from transformers.file_utils import WEIGHTS_NAME
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from .backbones.roberta.configuration_roberta import RobertaConfig
from .backbones.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaForSequenceClassification,
    RobertaLayer,
    RobertaLMHead,
    RobertaModel,
    RobertaPooler,
    RobertaPreTrainedModel,
)

logger = logging.getLogger(__name__)


class TGRobertaConfig(RobertaConfig):
    model_type = "TGRoberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, num_gnn_attention_heads=8, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.num_gnn_attention_heads = num_gnn_attention_heads


class TGRobertaEncoder(nn.Module):
    def __init__(self, config):
        super(TGRobertaEncoder, self).__init__()
        self.config = config
        # has to use the name to use .from_pretrained()
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gnn_layers = nn.ModuleList(
            [GATConv(config.hidden_size, config.hidden_size, heads=config.num_attention_heads) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        adj_t,
        attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    ):
        # hidden_states: B  L  D
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None

        B, L, D = hidden_states.shape  # batch_size, seq_length, hidden_dimension
        for i, (roberta_layer, gnn_layer) in enumerate(zip(self.layer, self.gnn_layers)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = roberta_layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            cls_emb = hidden_states[:, 0].clone()  # select [CLS] as node representation
            cls_emb = gnn_layer(cls_emb, adj_t)
            hidden_states[:, 0] = cls_emb

            if use_cache:
                next_decoder_cache += (layer_outputs[:-1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions += layer_outputs[2]

        # add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class TGRobertaModel(RobertaModel):
    # copied from transformers.models.roberta.modeling_roberta.RobertaModel.__init__
    def __init__(self, config, add_pooling_layer=True):
        super(TGRobertaModel, self).__init__(config=config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = TGRobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class TGRobertaForMaksedLM(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning("If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for " "bi-directional self-attention.")

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()


class TGRobertaForNodeClassification(RobertaForSequenceClassification):
    # copied from RobertaForSequenceClassifiction
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = TGRobertaModel(config, add_pooling_layer=False)
        self.classifier = TGRobertaNodeClsHead(config)

        # Initialize weights and apply final processing
        self.post_init()


class TGRobertaNodeClsHead(nn.Module):
    """TGRobertaNodeClsHead for node classification"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
