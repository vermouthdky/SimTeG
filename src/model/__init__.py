from .gbert.gbert_modeling import GBert
from .gnns.gnn_modeling import GAMLP, SAGN, SGC, SIGN, GraphSAGE
from .lms.lm_modeling import (
    AdapterDeberta,
    AdapterRoberta,
    Deberta,
    E5_model,
    Roberta,
    Sentence_Transformer,
)


def get_model_class(model: str, use_adapter: bool = False):
    model_class = {
        "GraphSAGE": GraphSAGE,
        "SGC": SGC,
        "SAGN": SAGN,
        "GAMLP": GAMLP,
        "SIGN": SIGN,
        "Roberta": Roberta,
        "Deberta": Deberta,
        "all-roberta-large-v1": Sentence_Transformer,
        "all-mpnet-base-v2": Sentence_Transformer,
        "all-MiniLM-L6-v2": Sentence_Transformer,
        "e5-large": E5_model,
    }
    if use_adapter:
        model_class["Roberta"] = AdapterRoberta
        model_class["Deberta"] = AdapterDeberta
    assert model in model_class.keys()
    return model_class[model]
