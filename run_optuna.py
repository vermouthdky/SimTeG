import logging
import os

import optuna
import torch
import torch.distributed as dist
from optuna.trial import TrialState

from main import set_env
from src.options import parse_args, save_args
from src.run import cleanup, get_trainer_class, load_data, load_model, set_single_env
from src.utils import set_logging

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def objective(single_trial):
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # setup optuna and search space
    trial = optuna.integration.TorchDistributedTrial(single_trial)
    args.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.5)
    args.hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0.1, 0.5)
    args.attention_dropout_prob = trial.suggest_float("attention_dropout_prob", 0.1, 0.5)
    args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.7)
    args.optuna = True
    logger.info(args)

    # if rank not in [-1, 0]:
    #     # Make sure only the first process in distributed training will download model & vocab
    #     torch.distributed.barrier()
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    model = load_model(args)
    # if rank == 0:
    #     torch.distributed.barrier()
    # trainer
    Trainer = get_trainer_class(args.model_type)
    trainer = Trainer(args, model, data, split_idx, evaluator, trial=trial)
    best_acc = trainer.train()
    return best_acc


def run(n_trials):
    # run
    set_logging()
    set_env(random_seed=123)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    if rank == 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            try:
                objective(None)
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
            print("    {}: {}".format(key, value))

    cleanup()


if __name__ == "__main__":
    n_trials = 10
    run(n_trials)
