from ..args import (
    DECOUPLING_GNN_LIST,
    GNN_LIST,
    LINK_PRED_DATASETS,
    LM_LIST,
    NODE_CLS_DATASETS,
    SAMPLING_GNN_LIST,
)
from .gnn_trainer import GNNDecouplingTrainer, GNNSamplingTrainer
from .link_lm_trainer import LinkLMTrainer
from .lm_trainer import LMTrainer


def get_trainer_class(args):
    model_type, dataset = args.model_type, args.dataset
    if model_type in LM_LIST and dataset in LINK_PRED_DATASETS:
        return LinkLMTrainer
    if model_type in LM_LIST and dataset in NODE_CLS_DATASETS:
        return LMTrainer
    if model_type in GNN_LIST and dataset in NODE_CLS_DATASETS:
        return GNNDecouplingTrainer if model_type in DECOUPLING_GNN_LIST else GNNSamplingTrainer
    else:
        raise NotImplementedError("not implemented Trainer class")
