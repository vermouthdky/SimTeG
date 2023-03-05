from .gbert_v2.gbert_v2_modeling import GBert_v2
from .gnns.gnn_modeling import GAMLP, SAGN, SIGN
from .lms.lm_modeling import AdapterDeberta, AdapterRoberta, Deberta, Roberta


def get_model_class(model: str, use_adapter: bool = False):
    model_class = {
        "GBert": GBert_v2,
        "SAGN": SAGN,
        "GAMLP": GAMLP,
        "SIGN": SIGN,
        "Roberta": Roberta,
        "Deberta": Deberta,
    }
    if use_adapter:
        model_class["Roberta"] = AdapterRoberta
        model_class["Deberta"] = AdapterDeberta
    assert model in model_class.keys()
    return model_class[model]
