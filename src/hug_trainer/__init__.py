from .gbert_trainer import GBertTrainer
from .gnn_trainer import GNNTrainer
from .lm_trainer import LMTrainer


def get_trainer_class(model_type):
    if model_type in ["Roberta", "Deberta"]:
        return LMTrainer
    elif model_type in ["GBert"]:
        return GBertTrainer
    elif model_type in ["GAMLP", "SAGN", "SIGN"]:
        return GNNTrainer
    else:
        raise NotImplementedError("not implemented Trainer class")
