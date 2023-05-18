from .gbert_trainer import GBert_Trainer
from .gnn_trainer import GNN_Trainer
from .lm_trainer import LM_Trainer


def get_trainer_class(model_type):
    if model_type in ["Roberta", "Deberta"]:
        return LM_Trainer
    elif model_type in ["SAGN", "SIGN", "GAMLP"]:
        return GNN_Trainer
    elif model_type in ["GBert"]:
        return GBert_Trainer
    else:
        raise NotImplementedError("not implemented Trainer Class")
