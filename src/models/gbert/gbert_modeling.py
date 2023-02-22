import logging
import os

import torch
from torch import nn
from transformers import AutoConfig, DebertaModel, RobertaModel
from transformers import logging as transformers_logging

from ..gnns.modules.EnGCN import GroupMLP
from ..lms.modules import (
    AdapterDebertaModel,
    AdapterRobertaModel,
    DebertaClassificationHead,
    RobertaClassificationHead,
)

logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()


class GBert(nn.Module):
    def __init__(self, args):
        super(GBert, self).__init__()
        self.bert_model, self.header = self._get_bert_model(args)
        # TODO: should ablate it in the future
        self.use_SLE = args.use_SLE
        self.lp_model = self._get_label_propogation_model(args) if self.use_SLE else None
        self.alpha = nn.Parameter(torch.tensor(0.8), requires_grad=False)  # moving averge

    def _get_config(self, args):
        pretrained_repo = args.pretrained_model
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.header_dropout_prob = args.header_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        logger.info("Save model config to %s", args.output_dir)
        return config

    def _get_label_propogation_model(self, args):
        return GroupMLP(
            args.num_labels,
            args.mlp_dim_hidden,
            args.num_labels,
            n_heads=1,
            n_layers=3,
            dropout=self.args.header_dropout_prob,
        )

    def _get_bert_model(self, args):
        pretrained_repo = args.pretrained_model
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
        return bert, header

    def forward(self, input_ids, att_mask, x_emb=None, y_emb=None, return_hidden=False):
        # NOTE: propogated_x = adj_T @ x
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        hidden_features = bert_out[0]
        if x_emb is not None:
            hidden_features[:, 0, :] = self.alpha * x_emb + (1 - self.alpha) * hidden_features[:, 0, :]
        logits = self.header(hidden_features)
        if self.use_SLE and y_emb is not None:
            logits += self.lp_model(y_emb)

        if return_hidden:
            return logits, hidden_features
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
