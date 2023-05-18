from .gbert_trainer import GBertTrainer
from .gnn_decoupling_trainer import GNNDecouplingTrainer
from .gnn_sage_trainer import GNNSamplingTrainer
from .lm_trainer import LMTrainer


def get_trainer_class(model_type):
    if model_type in [
        "Roberta",
        "Deberta",
        "all-roberta-large-v1",
        "all-mpnet-base-v2",
        "all-MiniLM-L6-v2",
        "e5-large",
    ]:
        return LMTrainer
    elif model_type in ["GBert"]:
        return GBertTrainer
    elif model_type in ["GAMLP", "SAGN", "SIGN", "SGC"]:
        return GNNDecouplingTrainer
    elif model_type in ["GraphSAGE"]:
        return GNNSamplingTrainer
    else:
        raise NotImplementedError("not implemented Trainer class")
