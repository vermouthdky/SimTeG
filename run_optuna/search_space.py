import logging
import os
import warnings

import torch
from HP_search import HP_search
from optuna.exceptions import ExperimentalWarning

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


class LM_HP_search(HP_search):
    def setup_search_space(self, args, trial):
        args.epochs = trial.suggest_int("epochs", 4, 10)
        args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.7)
        args.hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0.1, 0.4)
        args.accum_interval = trial.suggest_categorical("accum_interval", [1, 5, 10])
        if args.use_adapter:
            args.adapter_hidden_size = trial.suggest_categorical("adapter_hidden_size", [16, 32, 64, 128, 512, 768])
        else:
            args.attention_dropout_prob = trial.suggest_float("attention_dropout_prob", 0.1, 0.7)
        args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.7)
        args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.5)
        return args


class GBert_HP_search(HP_search):
    # the model related HPs should be consistent with the ones in LM_HP_search
    def setup_search_space(self, args, trial):
        args.epochs = trial.suggest_int("epochs", 1, 4)
        args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.7)
        args.accum_interval = trial.suggest_categorical("accum_interval", [1, 5, 10])
        args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.7)
        args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.5)
        args.kl_loss_weight = trial.suggest_float("kl_loss_weight", 0.1, 1.0)
        args.kl_loss_temp = trial.suggest_int("kl_loss_temp", 0, 4)
        args.gnn_epochs = trial.suggest_int("gnn_epochs", 5, 50)
        args.gnn_lalel_smoothing = trial.suggest_float("gnn_lalel_smoothing", 0.1, 0.7)
        args.inherit = trial.suggest_categorical("inherit", [True, False])
        args.compute_kl_loss = trial.suggest_categorical("compute_kl_loss", [True, False])
        args.fix_gnn = trial.suggest_categorical("fix_gnn", [True, False])
        return args
