from .gbert.gbert_modeling import GBert
from .gnns.gnn_modeling import GAMLP, GCN, SAGN, SGC, SIGN, GraphSAGE
from .lms.link_lm_modeling import Link_E5_model, Link_Sentence_Transformer
from .lms.lm_modeling import (
    AdapterDeberta,
    AdapterRoberta,
    Deberta,
    E5_model,
    Roberta,
    Sentence_Transformer,
)

node_cls_model_class = {
    "GraphSAGE": GraphSAGE,
    "GCN": GCN,
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

link_pred_model_class = {
    "GraphSAGE": GraphSAGE,
    "GCN": GCN,
    "SGC": SGC,
    "SAGN": SAGN,
    "GAMLP": GAMLP,
    "SIGN": SIGN,
    "Roberta": Roberta,
    "Deberta": Deberta,
    "all-roberta-large-v1": Link_Sentence_Transformer,
    "all-mpnet-base-v2": Link_Sentence_Transformer,
    "all-MiniLM-L6-v2": Link_Sentence_Transformer,
    "e5-large": Link_E5_model,
}


def get_model_class(model_type, task_type):
    assert model_type in node_cls_model_class.keys()
    if task_type == "node_cls":
        model_class = node_cls_model_class
    elif task_type == "link_pred":
        model_class = link_pred_model_class
    else:
        raise NotImplementedError("not implemented task type")
    return model_class[model_type]
