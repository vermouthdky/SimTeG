import gc
import logging
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from functools import partial

import optuna
import torch
import torch.distributed as dist
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState

from main import set_seed

from ..args import parse_args, save_args
from ..run import cleanup, get_trainer_class, load_data, set_single_env

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


def save_best_trial(study, trial, output_dir):  # call back
    best_value = study.best_trial.value if study.best_trial is not None else 0.0
    cur_val = trial.value
    if cur_val is not None and cur_val > best_value:
        best_output_dir = os.path.join(output_dir, "best")
        shutil.copytree(output_dir, best_output_dir)


class HP_search(ABC):
    def __init__(self, args):
        self.args = args

    def objective(self, trial):
        args = self.args
        rank = int(os.environ["RANK"])
        args.random_seed = rank

        args = self.setup_search_space(args, trial)
        args.optuna = True
        logger.info(args)
        dist.broadcast_object_list([args], src=0)
        best_acc = self.train(args, trial=trial)
        return best_acc

    def train(self, args, trial=None):
        data, split_idx, evaluator, processed_dir = load_data(args)
        # trainer
        Trainer = get_trainer_class(args.model_type)
        trainer = Trainer(args, data, split_idx, evaluator, trial=trial)
        best_acc = trainer.train()
        del trainer, data, split_idx, evaluator
        torch.cuda.empty_cache()
        gc.collect()
        return best_acc

    @abstractmethod
    def setup_search_space(self, args, trial):
        # setup search space
        # e.g. args.epochs = trial.suggest_int("epochs", 1, 10)
        pass

    def load_study(self):
        args = parse_args()
        args.random_seed = 0
        study = optuna.load_study(
            storage="sqlite:///optuna.db",
            study_name=f"{args.dataset}_{args.model_type}_{args.suffix}",
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
        return study

    def run(self, n_trials):
        # run
        args = self.args
        args.random_seed = 0
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        # have to set_device in order to use NCCL collective function, e.g. dist.broadcast_object_list
        torch.cuda.set_device(rank)
        set_seed(random_seed=args.random_seed)

        if rank == 0:
            study = optuna.create_study(
                direction="maximize",
                storage="sqlite:///optuna.db",
                study_name=f"{args.dataset}_{args.model_type}_{args.suffix}",
                load_if_exists=True,
                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                callbacks=[partial(save_best_trial, output_dir=args.output_dir)],
            )
            study.optimize(self.objective, n_trials=n_trials)
        else:
            for _ in range(n_trials):
                try:
                    to_broadcast = [args]
                    dist.broadcast_object_list(to_broadcast, src=0)
                    args = to_broadcast[0]
                    self.train(args, trial=None)
                except optuna.TrialPruned:
                    pass

        if rank == 0:
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

        cleanup()
