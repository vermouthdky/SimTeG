import gc
import logging
import os
import warnings

import torch
import torch.distributed as dist
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState

import optuna
from main import set_env
from src.args import parse_args, save_args
from src.run import cleanup, get_trainer_class, load_data, set_single_env
from src.utils import set_logging

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


def objective(single_trial):
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # setup optuna and search space
    trial = optuna.integration.TorchDistributedTrial(single_trial)
    args.epochs = trial.suggest_int("epochs", 5, 10)
    args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    args.label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.5)
    args.hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0.1, 0.7)
    if args.use_adapter:
        args.adapter_hidden_size = trial.suggest_categorical("adapter_hidden_size", [32, 128, 512, 768])
    args.attention_dropout_prob = trial.suggest_float("attention_dropout_prob", 0.1, 0.7)
    args.header_dropout_prob = trial.suggest_float("header_dropout_prob", 0.1, 0.7)
    args.schedular_warmup_ratio = trial.suggest_categorical("schedular_warmup_ratio", [0.1, 0.3, 0.5, 0.7])
    args.optuna = True
    logger.info(args)

    if rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    if rank == 0:
        torch.distributed.barrier()
    # trainer
    Trainer = get_trainer_class(args.model_type)
    trainer = Trainer(args, data, split_idx, evaluator, trial=trial)
    best_acc = trainer.train()
    del trainer, data, split_idx, evaluator
    torch.cuda.empty_cache()
    gc.collect()
    return best_acc


def load_study():
    set_logging()
    args = parse_args()
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_gbert.db",
        study_name=f"{args.dataset}_{args.model_type}_{args.suffix}",
        load_if_exists=True,
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


def run(n_trials):
    # run
    set_logging()
    args = parse_args()
    set_env(random_seed=123)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    if rank == 0:
        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///optuna_gbert.db",
            study_name=f"{args.dataset}_{args.model_type}_{args.suffix}",
            load_if_exists=True,
        )
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
            logger.info("    {}: {}".format(key, value))

    cleanup()


if __name__ == "__main__":
    n_trials = 40
    run(n_trials)
    # load_study()
