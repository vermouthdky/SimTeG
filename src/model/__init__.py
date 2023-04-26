from .gbert.gbert_modeling import GBert
from .gnns.gnn_modeling import GAMLP, SAGN, SIGN
from .lms.lm_modeling import (
    AdapterDeberta,
    AdapterRoberta,
    Deberta,
    Roberta,
    Sentence_Transformer,
)


def get_model_class(model: str, use_adapter: bool = False):
    model_class = {
        "GBert": GBert,
        "SAGN": SAGN,
        "GAMLP": GAMLP,
        "SIGN": SIGN,
        "Roberta": Roberta,
        "Deberta": Deberta,
        "all-roberta-large-v1": Sentence_Transformer,
        "all-mpnet-base-v2": Sentence_Transformer,
        "all-MiniLM-L6-v2": Sentence_Transformer,
    }
    if use_adapter:
        model_class["Roberta"] = AdapterRoberta
        model_class["Deberta"] = AdapterDeberta
    assert model in model_class.keys()
    return model_class[model]
