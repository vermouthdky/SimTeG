import logging
import os

import torch
from torch import nn
from transformers import DebertaConfig, DebertaModel, RobertaConfig, RobertaModel
from transformers import logging as transformers_logging

from ..lms.modules import AdapterDebertaModel, AdapterRobertaModel

logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()


class GBert(nn.Module):
    def __init__(self, args):
        super(GBert, self).__init__()
        pretrained_repo = "microsoft/deberta-base"
        config = DebertaConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        logger.info("Save model config to %s", args.output_dir)
        self.bert_model = AdapterDebertaModel.from_pretrained(pretrained_repo, config=config)

    def forward(self, xs, input_ids, att_mask):
        pass


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
