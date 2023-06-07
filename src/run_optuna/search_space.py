import logging
import warnings

from optuna.exceptions import ExperimentalWarning

from .HP_search import Dist_HP_search, Single_HP_search

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


class Decoupling_GNN_HP_search(Dist_HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-5, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_label_smoothing = trial.suggest_float("gnn_label_smoothing", 0.1, 0.7)
        args.gnn_warmup_ratio = trial.suggest_float("gnn_warmup_ratio", 0.1, 0.5)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 4, 8)
        return args


class LM_HP_search(Dist_HP_search):
    def setup_search_space(self, args, trial):
        args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.7)
        args.accum_interval = trial.suggest_categorical("accum_interval", [1, 5, 10])
        args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.5)
        args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.5)
        return args


class PEFT_LM_HP_search(Dist_HP_search):
    def setup_search_space(self, args, trial):
        args.lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.7)
        args.peft_r = trial.suggest_categorical("peft_r", [1, 2, 4, 8])
        args.peft_lora_alpha = trial.suggest_categorical("peft_lora_alpha", [4, 8, 16, 32])
        args.peft_lora_dropout = trial.suggest_float("peft_lora_dropout", 0.1, 0.8)
        args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.8)
        return args


class Sampling_GNN_HP_search(Single_HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-4, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_label_smoothing = trial.suggest_float("gnn_label_smoothing", 0.1, 0.7)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 2, 3)
        return args


class Link_GNN_HP_search(Single_HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-4, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 2, 3)
        return args
