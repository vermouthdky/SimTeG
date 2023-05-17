import gc
import logging
import os
import sys
import warnings

from optuna.exceptions import ExperimentalWarning

from src.args import parse_args
from src.run_optuna.search_space import GNN_HP_search, LM_HP_search
from src.utils import set_logging

logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
warnings.filterwarnings("ignore", category=FutureWarning)


def get_search_instance(model_type):
    if model_type in [
        "Deberta",
        "all-roberta-large-v1",
        "all-mpnet-base-v2",
        "all-MiniLM-L6-v2",
        "e5-large",
    ]:
        return LM_HP_search
    elif model_type in ["GAMLP", "SAGN", "SGC", "GraphSAGE"]:
        return GNN_HP_search
    else:
        raise NotImplementedError("not implemented HP search class")


def main():
    set_logging()
    args = parse_args()
    hp_search = get_search_instance(args.model_type)(args)
    if args.load_study:
        hp_search.load_study()
    else:
        logger.critical(
            f"Start HP search, optuna the {args.model_type} model on {args.dataset} dataset for {args.n_trials} trials"
        )
        hp_search.run(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
