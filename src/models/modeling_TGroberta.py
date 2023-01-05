import logging
import time
from typing import List, Optional, Tuple, Union

import torch
from torch import nn, tanh
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch_geometric.nn import GATConv
from tqdm import tqdm
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

from .LMs.modeling_gnn import GNNModel
from .LMs.roberta.configuration_roberta import RobertaConfig
from .LMs.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaLMHead,
    RobertaModel,
    RobertaPooler,
    RobertaPreTrainedModel,
)


class TGRobertaConfig(RobertaConfig):
    model_type = "TGRoberta"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        gnn_type="GIN",
        gnn_num_layers=4,
        gnn_dropout=0.1,
        **kwargs,
    ):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # self.gnn_num_attention_heads = gnn_num_attention_heads
        self.gnn_type = gnn_type
        self.gnn_num_layers = gnn_num_layers
        self.gnn_dropout = gnn_dropout


class TGRobertaModel(RobertaModel):
    # copied from transformers.models.roberta.modeling_roberta.RobertaModel.__init__
    def __init__(self, config, add_pooling_layer=True):
        super(TGRobertaModel, self).__init__(config=config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.gnn = GNNModel(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def gnn_forward(
        self,
        batch_size=None,
        input_ids: Optional[torch.Tensor] = None,
        adjs: Optional[list[torch.Tensor]] = None,
    ):
        input_shape = input_ids.size()
        B, L = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :L]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(B, L)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        cls_embedding = embedding_output[:, 0, :].clone()
        gnn_out = self.gnn(x=cls_embedding, adjs=adjs)

        embedding_output[:batch_size, 0, :] = gnn_out
        return embedding_output[:batch_size]

    # copied from transformers.model.roberta.modeling_roberta.RobertaModel.forward
    def forward(
        self,
        batch_size,
        input_ids: Optional[torch.Tensor] = None,
        adjs=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        gnn_output = self.gnn_forward(batch_size, input_ids, adjs)
        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        N, L, D = gnn_output.shape
        out = self.encoder.forward(
            gnn_output,
            attention_mask=extended_attention_mask.cuda(),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = out[0]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
        )


class TGRobertaForNodeClassification(RobertaPreTrainedModel):
    # copied from RobertaForSequenceClassifiction
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = TGRobertaModel(config, add_pooling_layer=False)
        self.classifier = TGRobertaNodeClsHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        batch_size=None,
        input_ids: Optional[torch.LongTensor] = None,
        adjs=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        train_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ):
        sequence_output = self.roberta(
            batch_size, input_ids, adjs, attention_mask, output_attentions, output_hidden_states
        )[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()

                loss = loss_fct(
                    logits.view(-1, self.num_labels)[train_mask],
                    labels.view(-1)[train_mask],
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits[train_mask], labels[train_mask])

        return (logits, loss) if output_hidden_states else loss

    @torch.no_grad()
    def inference(self, data, subgraph_loader, rank):
        logits_all = []
        pbar = tqdm(subgraph_loader, desc="Iteration", disable=False)
        for step, (batch_size, n_id, adjs) in enumerate(subgraph_loader):
            adjs = [adj.to(rank) for adj in adjs]
            logits, loss = self(
                batch_size,
                data.input_ids[n_id].cuda(),
                adjs,
                data.attention_mask[n_id][:batch_size].cuda(),
                data.y[n_id][:batch_size].cuda(),
                data.train_mask[n_id][:batch_size],
                output_hidden_states=True,
            )
            logits_all.append(logits.to("cpu"))
            pbar.update()
        logits_all = torch.cat(logits_all, dim=0)
        return logits_all


class TGRobertaNodeClsHead(nn.Module):
    """TGRobertaNodeClsHead for node classification"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
