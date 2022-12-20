from .roberta.modeling_roberta import RobertaEmbeddings, RobertaModel, RobertaLMHead
from .roberta.configuration_roberta import RobertaConfig


class LM:
    def __init__(self, model_type: str):
        self.model_type = model_type
        if model_type == "roberta":
            self.Embeddings = RobertaEmbeddings
            self.Model = RobertaModel
            self.LMHead = RobertaLMHead
            self.Config = RobertaConfig
        else:
            raise NotImplementedError
