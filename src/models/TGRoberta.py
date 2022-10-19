import logging
import os

import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers.file_utils import WEIGHTS_NAME

from models.backbones.roberta.convert_state_dict import state_dict_convert
from models.backbones.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaLayer,
    RobertaEmbeddings,
    RobertaPreTrainedModel,
)

logger = logging.getLogger(__name__)


class TGRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super(TGRoberta, self).__init__(config=config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config=config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.roberta_layers = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        pass

    # TODO tomorrow


class TGRobertaForMaksedLM(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.is_decoder:
            logger.warning("If you want to use RobertaMaskedLM, make sure `config.is_decoder=False`")

        self.Gbert = TGRoberta(config)
        self.lm_head = RobertaLMHead(config)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_header.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(self, input_ids, attention_mask, adj_t):
        # B, L = input_ids.shape
        # D = self.config.hidden_size
        hidden_states = self.Gbert(input_ids, attention_mask, adj_t)
        node_embeddings = hidden_states[0]  # hidden_states of [CLS]
        return node_embeddings

    def train(self, input_ids, attention_mask, adj_t):
        pass

    def test(self, input_ids, attention_mask, adj_t):
        pass


class TGBobertaForNodeClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.Gbert = TGRoberta(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, adj_t):
        # B, L = input_ids.shape
        # D = self.config.hidden_size
        hidden_states = self.Gbert(input_ids, attention_mask, adj_t)
        node_embeddings = hidden_states[0]  # hidden_states of [CLS]
        return node_embeddings

    def train(self, input_ids, attention_mask, adj_t):
        pass

    def test(self, input_ids, attention_mask, adj_t):
        pass
