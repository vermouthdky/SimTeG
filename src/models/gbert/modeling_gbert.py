import logging
import os

import torch
from torch import nn
from transformers import AutoConfig, DebertaModel, RobertaModel
from transformers import logging as transformers_logging

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
        self.bert_model, self.header = self._get_model(args)
        self.alpha = nn.Parameter(torch.tensor(0.8), requires_grad=False)  # moving averge

    def _get_config(self, args):
        pretrained_repo = args.pretrained_model
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.classifier_prob = args.header_dropout_prob
        config.attention_probs_dropout_prob = args.attention_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        logger.info("Save model config to %s", args.output_dir)
        return config

    def _get_model(self, args):
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

    def get_adj_t(self, data):
        deg = data.adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return deg_inv_sqrt.view(-1, 1) * data.adj_t * deg_inv_sqrt.view(1, -1)

    def propogate(self, x, adj_t):
        pass

    def forward(self, propogated_x, input_ids, att_mask, return_hidden=True):
        # NOTE: propogated_x = adj_T @ x
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        hidden_features = bert_out[0]
        if propogated_x is not None:
            hidden_features[:, 0, :] = self.alpha * propogated_x + (1 - self.alpha) * hidden_features[:, 0, :]
        logits = self.header(hidden_features)
        return logits, hidden_features if return_hidden else (logits,)


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
