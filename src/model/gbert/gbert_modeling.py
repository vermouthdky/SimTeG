import logging
import os

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig, DebertaModel, RobertaModel
from transformers import logging as transformers_logging

from ..gnns.modules import GAMLP, GAMLP_SLE, SAGN, SAGN_SLE
from ..lms.modules import (
    AdapterDebertaModel,
    AdapterRobertaModel,
    DebertaClassificationHead,
    RobertaClassificationHead,
)

logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()


class GBert(nn.Module):
    def __init__(self, args, iter: int):
        super(GBert, self).__init__()
        self.iter = iter
        self.hidden_size = args.hidden_size
        if self.iter == 0:
            self.bert_model, self.head = self._get_bert_model(args, head=True)
        else:
            self.bert_model = self._get_bert_model(args, head=False)
            self.gnn_model = self._get_gnn_model(args)

    def _get_config(self, args):
        pretrained_repo = args.pretrained_repo
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.header_dropout_prob = args.header_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.adapter_hidden_size = args.adapter_hidden_size
        config.save_pretrained(save_directory=args.output_dir)
        logger.info("Save model config to %s", args.output_dir)
        return config

    def _get_gnn_model(self, args):
        model_type = args.gnn_type
        assert model_type in ["GAMLP", "SAGN", "SIGN"]
        if model_type == "GAMLP" and args.use_SLE:
            gnn_model = GAMLP_SLE(args)
        elif model_type == "GAMLP" and not args.use_SLE:
            gnn_model = GAMLP(args)
        elif model_type == "SAGN" and args.use_SLE:
            gnn_model = SAGN(args)
        elif model_type == "SAGN" and not args.use_SLE:
            gnn_model = SAGN_SLE(args)
        else:
            raise NotImplementedError
        return gnn_model

    def _get_bert_model(self, args, head: bool):
        pretrained_repo = args.pretrained_repo
        config = self._get_config(args)
        if pretrained_repo == "microsoft/deberta-base" and args.use_adapter:
            bert = AdapterDebertaModel.from_pretrained(pretrained_repo, config=config)
            header = DebertaClassificationHead(config)
        elif pretrained_repo == "microsoft/deberta-base" and not args.use_adapter:
            bert = DebertaModel.from_pretrained(pretrained_repo, config=config)
            header = DebertaClassificationHead(config)
        elif pretrained_repo == "roberta-base" and args.use_adapter:
            bert = AdapterRobertaModel.from_pretrained(pretrained_repo, config=config)
            header = RobertaClassificationHead(config)
        elif pretrained_repo == "roberta-base" and not args.use_adapter:
            bert = RobertaModel.from_pretrained(pretrained_repo, config=config)
            header = RobertaClassificationHead(config)
        else:
            raise NotImplementedError("Invalid model name")
        if args.use_adapter:
            logger.critical("using adapter!!!")
        if head:
            return bert, header
        else:
            return bert

    def forward(self, input_ids, att_mask, x_emb=None, y_emb=None, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        hidden_features = bert_out[0]
        if self.iter == 0:
            logits = self.head(hidden_features)
        else:
            assert x_emb is not None
            bert_out = hidden_features[:, 0, :]
            xs = [bert_out] + [x for x in torch.split(x_emb, self.hidden_size, -1)]
            logits = self.gnn_model(xs)

            # if self.use_SLE and y_emb is not None:
            #     logits += self.lp_model(y_emb).squeeze(dim=1)

        if return_hidden:
            return logits, hidden_features[:, 0, :]
        else:
            return logits
