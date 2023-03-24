import logging
import os

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig, DebertaModel, RobertaModel
from transformers import logging as transformers_logging

from ..gnns.modules import GAMLP, SAGN, SIGN
from ..lms.modules import (
    AdapterDebertaModel,
    AdapterRobertaModel,
    DebertaClassificationHead,
    RobertaClassificationHead,
)

logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()


class GBert_v2(nn.Module):
    def __init__(self, args, iter: int):
        super(GBert_v2, self).__init__()
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
        if model_type == "GAMLP":
            gnn_model = GAMLP(args)
        elif model_type == "SAGN":
            gnn_model = SAGN(args, num_hops=args.gnn_num_layers + 1)
        elif model_type == "SIGN":
            gnn_model = SIGN(args, num_hops=args.gnn_num_layers + 1)
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


# class GBert_v1(nn.Module):  # use X_OGB and Roberta
#     def __init__(self, args):
#         super(GBert_v1, self).__init__()
#         config = RobertaConfig.from_pretrained("roberta-base")
#         config.num_labels = args.num_labels
#         config.hidden_dropout_prob = args.hidden_dropout_prob
#         config.save_pretrained(save_directory=args.output_dir)
#         logger.info("Saving model config to %s", args.output_dir)
#         self.bert_model = AdapterRobertaModel.from_pretrained("roberta-base", add_pooling_layer=False, config=config)
#         self.trans_lin = nn.Linear(config.hidden_size, args.num_feats)

#         num_hops = args.gnn_num_layers + 2  # num_layers + orignal features + bert output
#         assert args.gnn_type in ["SAGN", "SIGN"]
#         gnn_model_class = {"SAGN": SAGN, "SIGN": SIGN}
#         self.gnn_model = gnn_model_class[args.gnn_type](args, num_hops=num_hops)
#         self.layer_norms = nn.ModuleList([nn.LayerNorm(args.num_feats) for _ in range(num_hops)])

#     def forward(self, xs, input_ids, att_mask):
#         bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
#         bert_out = self.trans_lin(bert_out[0][:, 0, :])
#         xs.append(bert_out)
#         # NOTE: normalize input to the same scale
#         for i, x in enumerate(xs):
#             xs[i] = self.layer_norms[i](x)
#         out = self.gnn_model(xs)
#         return out
