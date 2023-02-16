import json
import os
import sys

hparams = dict(
    lr=[5e-5, 1e-5, 5e-6],
    hidden_dropout_prob=[0.3, 0.5, 0.7],
    header_dropout_prob=[0.5, 0.7],
)

keys = hparams.keys()

model = "Deberta"
dataset = "ogbn-arxiv"

default_config = dict(
    model_type=model,
    dataset=dataset,
    suffix="main",
    eval_interval=1,
    lr=1e-5,
    weight_decay=0.0,
    batch_size=10,
    eval_batch_size=100,
    epochs=5,
    accum_interval=5,
    hidden_dropout_prob=0.1,
    header_dropout_prob=0.3,
)


def make_cmd_line(exp_name, **kwargs):
    config = default_config.copy()
    config.update(kwargs)
    config["suffix"] = exp_name
    args = " ".join([f"{v}" for _, v in config.items()])
    return "bash scripts/train.sh " + args


for k in keys:
    for v in hparams[k]:
        exp_name = f"{k}_{v}"
        cmd_line = make_cmd_line(exp_name, **{k: v})
        print(cmd_line)
        os.system(cmd_line)
