import logging
import os

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig, DebertaModel, PreTrainedModel, RobertaModel
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
    def __init__(self, args, iter: int = 0, module_to_train: str = "lm"):
        super().__init__()
        assert module_to_train in ["lm", "gnn"]
        self.iter = iter
        self.hidden_size = args.hidden_size
        self.module_to_train = module_to_train

        if module_to_train == "gnn":
            self.gnn_model = self._get_gnn_model(args)
        else:
            self.bert_model = self._get_bert_model(args)
            if self.iter == 0:
                self.head = self._get_header_model(args)
            else:
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

    def _get_header_model(self, args):
        pretrained_repo = args.pretrained_repo
        config = self._get_config(args)
        if pretrained_repo == "microsoft/deberta-base":
            return DebertaClassificationHead(config)
        elif pretrained_repo == "roberta-base":
            return RobertaClassificationHead(config)
        else:
            raise NotImplementedError("Invalid header name from repo {}".format(pretrained_repo))

    def _get_bert_model(self, args):
        pretrained_repo = args.pretrained_repo
        config = self._get_config(args)
        if pretrained_repo == "microsoft/deberta-base" and args.use_adapter:
            bert = AdapterDebertaModel.from_pretrained(pretrained_repo, config=config)
        elif pretrained_repo == "microsoft/deberta-base" and not args.use_adapter:
            bert = DebertaModel.from_pretrained(pretrained_repo, config=config)
        elif pretrained_repo == "roberta-base" and args.use_adapter:
            bert = AdapterRobertaModel.from_pretrained(pretrained_repo, config=config)
        elif pretrained_repo == "roberta-base" and not args.use_adapter:
            bert = RobertaModel.from_pretrained(pretrained_repo, config=config)
        else:
            raise NotImplementedError("Invalid model name")
        if args.use_adapter:
            logger.warning("using adapter!!!")
        return bert

    def gnn_foward(self, x_emb, y_emb):
        if y_emb is not None:
            return self.gnn_model(x_emb, y_emb)
        else:
            return self.gnn_model(x_emb)

    def forward(self, input_ids=None, att_mask=None, x_emb=None, y_emb=None, x0=None, labels=None, return_hidden=False):
        # labels should not be reomved to be consistent with the trainer class (huggingface)
        if self.module_to_train == "lm":
            assert x0 is None
            bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)[0]
            hidden_features = bert_out[:, 0, :]
            if hasattr(self, "head"):
                logits = self.head(bert_out)
            else:  # use gnn as header
                assert x_emb is not None
                x_emb = torch.cat((hidden_features, x_emb), dim=-1)
                logits = self.gnn_foward(x_emb, y_emb)

            if return_hidden == True:
                return logits, hidden_features
            else:
                return logits
        else:
            assert x_emb is not None
            assert x0 is not None
            x_emb = torch.cat((x0, x_emb), dim=-1)
            logits = self.gnn_foward(x_emb, y_emb)
            return logits
