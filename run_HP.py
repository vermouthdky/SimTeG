import json
import os
import sys

hparams = dict(
    # lr=[0.001, 0.0001],
    # gnn_dropout=[0.3, 0.5, 0.7],
    # epochs=[50, 70],
    # gnn_num_layers=[2, 4, 8],
    hidden_dropout_prob=[0.3, 0.5, 0.7]
)

model = "GBert"
dataset = "ogbn-arxiv"

# NOTE: the same with gbert.sh
default_config = dict(
    model_type=model,
    dataset=dataset,
    suffix="main",
    eval_interval=5,
    lr=0.001,
    weight_decay=0.0,
    batch_size=20,
    eval_batch_size=300,
    epochs=50,
    accum_interval=5,
    hidden_dropout_prob=0.1,
    gnn_num_layers=4,
    gnn_type="SAGN",
    gnn_dropout=0.3,
)


def make_cmd_line(exp_name, **kwargs):
    config = default_config.copy()
    config.update(kwargs)
    config["suffix"] = exp_name
    args = " ".join([f"{v}" for _, v in config.items()])
    return "bash scripts/train.sh " + args


# keys = ["gnn_dropout", "gnn_num_layers"]
keys = ["hidden_dropout_prob"]

for k in keys:
    for v in hparams[k]:
        exp_name = f"{k}_{v}"
        cmd_line = make_cmd_line(exp_name, **{k: v})
        print(cmd_line)
        os.system(cmd_line)
