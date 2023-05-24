import gc
import logging
import warnings
from functools import partial

import optuna
import torch
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState

from main import set_seed

from .HP_search import HP_search, save_best_trial

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


class Decoupling_GNN_HP_search(HP_search):
    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-5, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_label_smoothing = trial.suggest_float("gnn_label_smoothing", 0.1, 0.7)
        args.gnn_warmup_ratio = trial.suggest_float("gnn_warmup_ratio", 0.1, 0.5)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 4, 8)
        return args


class LM_HP_search(HP_search):
    def setup_search_space(self, args, trial):
        args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.7)
        args.accum_interval = trial.suggest_categorical("accum_interval", [1, 5, 10])
        args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.5)
        args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.5)
        return args


class PEFT_LM_HP_search(HP_search):
    def setup_search_space(self, args, trial):
        args.peft_r = trial.suggest_categorical("peft_r", [1, 2, 4, 8])
        args.peft_lora_alpha = trial.suggest_categorical("peft_lora_alpha", [4, 8, 16, 32])
        args.peft_lora_dropout = trial.suggest_float("peft_lora_dropout", 0.1, 0.8)
        args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.8)
        return args


class Sampling_GNN_HP_search(HP_search):
    def objective(self, trial):
        args = self.args
        args = self.setup_search_space(args, trial)
        args.optuna = True
        logger.info(args)
        best_acc = self.train(args, trial=trial)
        return best_acc

    def setup_search_space(self, args, trial):
        args.gnn_lr = trial.suggest_float("gnn_lr", 1e-4, 1e-2, log=True)
        args.gnn_weight_decay = trial.suggest_float("gnn_weight_decay", 1e-7, 1e-4, log=True)
        args.gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.8)
        args.gnn_label_smoothing = trial.suggest_float("gnn_label_smoothing", 0.1, 0.7)
        args.gnn_num_layers = trial.suggest_int("gnn_num_layers", 2, 4)
        return args

    def run(self, n_trials):
        # run
        args = self.args
        args.random_seed = args.start_seed
        set_seed(random_seed=args.random_seed)

        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///optuna.db",
            study_name=f"{args.dataset}_{args.model_type}_{args.suffix}",
            load_if_exists=True,
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
        )
        study.optimize(
            self.objective, n_trials=n_trials, callbacks=[partial(save_best_trial, output_dir=args.output_dir)]
        )
        assert study is not None
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logger.info("Study statistics: ")
        logger.info("  Number of finished trials: {}".format(len(study.trials)))
        logger.info("  Number of pruned trials: {}".format(len(pruned_trials)))
        logger.info("  NUmber of complete trials: {}".format(len(complete_trials)))

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info("  Value: {}".format(trial.value))

        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))

        torch.cuda.empty_cache()
        gc.collect()


# class GBert_HP_search(HP_search):
#     # the model related HPs should be consistent with the ones in LM_HP_search
#     def setup_search_space(self, args, trial):
#         args.epochs = trial.suggest_int("epochs", 1, 4)
#         args.gnn_epochs = trial.suggest_int("gnn_epochs", 20, 50)
#         args.lr_scheduler_type = trial.suggest_categorical("lr_scheduler", ["linear", "constant"])
#         args.gnn_inherit = trial.suggest_categorical("gnn_inherit", [True, False])
#         if args.lr_scheduler_type == "linear":
#             args.lr = 8e-4
#             args.gnn_lr = 1e-2
#         elif args.lr_scheduler_type == "constant":
#             args.lr = 5e-4
#             args.gnn_lr = 5e-3
#         args.SLE_threshold = trial.suggest_float("SLE_threshold", 0.5, 0.95)
#         args.SLE_mode = trial.suggest_categorical("SLE_mode", ["gnn", "lm", "both"])
#         return args
