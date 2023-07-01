from .gnns.gnn_modeling import *
from .gnns.link_gnn_modeling import *
from .lms.link_lm_modeling import *
from .lms.lm_modeling import *

node_cls_model_class = {
    "GraphSAGE": GraphSAGE,
    "GCN": GCN,
    "SGC": SGC,
    "SAGN": SAGN,
    "GAMLP": GAMLP,
    "SIGN": SIGN,
    "MLP": MLP,
    "deberta-v2-xxlarge": Deberta,
    "all-roberta-large-v1": Sentence_Transformer,
    "all-mpnet-base-v2": Sentence_Transformer,
    "all-MiniLM-L6-v2": Sentence_Transformer,
    "instructor-xl": T5_model,
    "sentence-t5-large": T5_model,
    "e5-large": E5_model,
    "e5-large-v2": E5_model,
    "roberta-large": Roberta,
}

link_pred_model_class = {
    "GraphSAGE": LinkGraphSAGE,
    "GCN": LinkGCN,
    "MLP": LinkMLP,
    "deberta-v2-xxlarge": Deberta,
    "all-roberta-large-v1": Link_Sentence_Transformer,
    "all-mpnet-base-v2": Link_Sentence_Transformer,
    "all-MiniLM-L6-v2": Link_Sentence_Transformer,
    "instructor-xl": Sentence_Transformer,
    "sentence-t5-large": T5_model,
    "e5-large": Link_E5_model,
    "e5-large-v2": Link_E5_model,
    "roberta-large": Link_Roberta,
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
