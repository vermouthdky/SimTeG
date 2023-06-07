import argparse
import logging

import optuna
import torch
import torch.nn.functional as F
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.utils.data import DataLoader
from torch_geometric.data import NeighborSampler
from torch_geometric.transforms import ToUndirected
from tqdm.auto import tqdm

from ..model import get_model_class

logger = logging.getLogger(__name__)


##########################################
# Base Classes: LinkTrainer, LinkPredictor
###########################################


class LinkTrainer:
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        assert args.dataset in ["ogbl-citation2"]
        self.args = args
        self.data = ToUndirected()(data)  # for ogbl-citation2
        self.split_edge = split_idx
        self.evaluator = evaluator
        self.trial = kwargs.get("trial", None)
        self.model, self.predictor = self._prepare_model()
        self.optimizer = self._prepare_optimizer()

    @property
    def device(self):
        return torch.device(self.args.single_gpu if torch.cuda.is_available() else "cpu")

    def _prepare_model(self):
        raise NotImplementedError

    def _prepare_dataloader(self):
        raise NotImplementedError

    def _prepare_optimizer(self):
        raise NotImplementedError

    def training_step(self, epoch):
        raise NotImplementedError

    def get_embeddings(self):
        raise NotImplementedError

    @torch.no_grad()
    def eval(self):
        h = self.get_embeddings()

        def test_split(split):
            target = self.split_edge[split]["target_node"]
            source = self.split_edge[split]["source_node"]
            target_neg = self.split_edge[split]["target_node_neg"].contiguous()

            pos_preds = []
            for perm in DataLoader(range(source.size(0)), self.args.gnn_batch_size):
                src, dst = source[perm], target[perm]
                pos_preds += [self.predictor(h[src].to(self.device), h[dst].to(self.device)).squeeze().cpu()]
            pos_pred = torch.cat(pos_preds, dim=0)

            neg_preds = []
            source = source.view(-1, 1).repeat(1, 1000).view(-1)
            target_neg = target_neg.reshape(-1)
            for perm in DataLoader(range(source.size(0)), self.args.gnn_batch_size):
                src, dst_neg = source[perm], target_neg[perm]
                neg_preds += [self.predictor(h[src].to(self.device), h[dst_neg].to(self.device)).squeeze().cpu()]
            neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

            eval_results = self.evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})
            return {key[:-5]: value.mean().item() for key, value in eval_results.items()}

        valid_metrics = test_split("valid")
        test_metrics = test_split("test")
        return valid_metrics, test_metrics

    def train(self, return_value="valid"):
        best_val_mrr = 0
        best_val_metrics, final_test_metrics = {}, {}
        accumulate_patience = 0
        for epoch in range(1, self.args.gnn_epochs + 1):
            loss = self.training_step(epoch)
            logger.info(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

            if epoch >= self.args.gnn_eval_warmup and epoch % self.args.gnn_eval_interval == 0:
                valid_metrics, test_metrics = self.eval()
                logger.info(
                    f"Epoch: {epoch}, Valid Metrics:: "
                    + "".join("{}: {:.4f} ".format(k, v) for k, v in valid_metrics.items())
                )
                logger.info(
                    f"Epoch: {epoch}, Test Metrics:: "
                    + "".join("{}: {:.4f} ".format(k, v) for k, v in test_metrics.items())
                )
                val_mrr = valid_metrics["mrr"]
                if val_mrr > best_val_mrr:
                    accumulate_patience = 0
                    best_val_mrr = val_mrr
                    best_val_metrics = valid_metrics
                    final_test_metrics = test_metrics
                else:
                    accumulate_patience += 1
                    if accumulate_patience >= 2:
                        break
            if self.trial is not None:
                self.trial.report(val_mrr, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        logger.info(
            f"Epoch: {epoch}, Best Valid Metrics:: "
            + "".join("{}: {:.4f} ".format(k, v) for k, v in best_val_metrics.items())
        )
        logger.info(
            f"Epoch: {epoch}, Final Test Metrics:: "
            + "".join("{}: {:.4f} ".format(k, v) for k, v in final_test_metrics.items())
        )
        return final_test_metrics, best_val_metrics


class LinkPredictor(torch.nn.Module):
    def __init__(self, args):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(args.num_feats, args.hidden_size))
        for _ in range(args.gnn_num_layers - 2):
            self.lins.append(torch.nn.Linear(args.hidden_size, args.hidden_size))
        self.lins.append(torch.nn.Linear(args.hidden_size, 1))
        self.dropout = args.gnn_dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


##########################################
# GraphSAGE with NeighborSampler
###########################################


class PositiveLinkNeighborSampler(NeighborSampler):
    def __init__(self, edge_index, sizes, num_nodes=None, **kwargs):
        edge_idx = torch.arange(edge_index.size(1))
        super(PositiveLinkNeighborSampler, self).__init__(edge_index, sizes, edge_idx, num_nodes, **kwargs)

    def sample(self, edge_idx):
        if not isinstance(edge_idx, torch.Tensor):
            edge_idx = torch.tensor(edge_idx)
        row, col, _ = self.adj_t.coo()
        batch = torch.cat([row[edge_idx], col[edge_idx]], dim=0)
        return super(PositiveLinkNeighborSampler, self).sample(batch)


class NegativeLinkNeighborSampler(NeighborSampler):
    def __init__(self, edge_index, sizes, num_nodes=None, **kwargs):
        edge_idx = torch.arange(edge_index.size(1))
        super(NegativeLinkNeighborSampler, self).__init__(edge_index, sizes, edge_idx, num_nodes, **kwargs)

    def sample(self, edge_idx):
        num_nodes = self.adj_t.sparse_size(0)
        batch = torch.randint(0, num_nodes, (2 * len(edge_idx),), dtype=torch.long)
        return super(NegativeLinkNeighborSampler, self).sample(batch)


class LinkGNNSamplingTrainer:  # single GPU
    def _prepare_model(self):
        assert self.args.model_type in ["GraphSAGE", "GCN"]
        model_class = get_model_class(self.args.model_type, self.args.task_type)
        return model_class(self.args), LinkPredictor(self.args)

    def _prepare_dataloader(self):
        sizes = [15, 10, 5, 5, 5, 5]
        kwargs = dict(
            sizes=sizes[: self.args.gnn_num_layers],
            num_nodes=self.data.x.size(0),
            batch_size=self.args.gnn_batch_size,
            num_workers=12,
            persistent_workers=True,
        )
        pos_loader = PositiveLinkNeighborSampler(self.data.edge_index, shuffle=True, **kwargs)
        neg_loader = NegativeLinkNeighborSampler(self.data.edge_index, shuffle=False, **kwargs)

        subgraph_loader = NeighborSampler(
            self.data.edge_index,
            node_idx=None,
            sizes=[-1],
            batch_size=self.args.gnn_eval_batch_size,
            shuffle=False,
            num_workers=12,
            persistent_workers=True,
        )
        return pos_loader, neg_loader, subgraph_loader

    def _prepare_optimizer(self):
        return torch.optim.Adam(list(self.model.parameters()) + list(self.predictor.parameters()), lr=self.args.gnn_lr)

    def training_step(self, epoch):
        self.model.train()
        self.predictor.train()
        pbar = tqdm(total=1000)
        pbar.set_description("Training")

        total_loss, total_examples = 0, 0
        for i, (pos_data, neg_data) in enumerate(zip(self.pos_loader, self.neg_loader)):
            self.optimizer.zero_grad()

            batch_size, n_id, adjs = pos_data
            adjs = [adj.to(self.device) for adj in adjs]
            h = self.model(self.data.x[n_id], adjs)
            h_src, h_dst = h.chunk(2, dim=0)
            pos_out = self.predictor(h_src, h_dst)
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            batch_size, n_id, adjs = neg_data
            adjs = [adj.to(self.device) for adj in adjs]
            h = self.model(self.data.x[n_id], adjs)
            h_src, h_dst = h.chunk(2, dim=0)
            neg_out = self.predictor(h_src, h_dst)
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            self.optimizer.step()

            num_examples = h_src.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

            pbar.update(1)

            # we dont run a full epoch here since that just takes too much time
            if (i + 1) % 1000 == 0:
                break

        pbar.close()
        return total_loss / total_examples

    @torch.no_grad()
    def eval(self):
        self.predictor.eval()
        self.model.eval()

        h = self.model.inference(self.data.x, self.subgraph_loader, self.device).to(self.device)

        def test_split(split):
            source = self.split_edge[split]["source_node"].to(self.device)
            target = self.split_edge[split]["target_node"].to(self.device)
            target_neg = self.split_edge[split]["target_node_neg"].to(self.device)

            pos_preds = []
            for perm in DataLoader(range(source.size(0)), self.args.gnn_batch_size):
                src, dst = source[perm], target[perm]
                pos_preds += [self.predictor(h[src], h[dst]).squeeze().cpu()]
            pos_pred = torch.cat(pos_preds, dim=0)

            neg_preds = []
            source = source.view(-1, 1).repeat(1, 1000).view(-1)
            target_neg = target_neg.view(-1)
            for perm in DataLoader(range(source.size(0)), self.args.gnn_batch_size):
                src, dst_neg = source[perm], target_neg[perm]
                neg_preds += [self.predictor(h[src], h[dst_neg]).squeeze().cpu()]
            neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

            return (
                self.evaluator.eval(
                    {
                        "y_pred_pos": pos_pred,
                        "y_pred_neg": neg_pred,
                    }
                )["mrr_list"]
                .mean()
                .item()
            )

        train_mrr = test_split("eval_train")
        valid_mrr = test_split("valid")
        test_mrr = test_split("test")

        results = dict(train_mrr=train_mrr, valid_mrr=valid_mrr, test_mrr=test_mrr)
        return results

    def train(self, return_value="valid"):
        self.model, self.predictor = self._prepare_model()
        self.model.to(self.device)
        self.predictor.to(self.device)
        self.data.x.to(self.device)
        self.model.reset_parameters()
        self.pos_loader, self.neg_loader, self.subgraph_loader = self._prepare_dataloader()
        self.optimizer = self._prepare_optimizer()
        best_val_mrr = final_test_mrr = 0
        accumulate_patience = 0
        for epoch in range(1, self.args.gnn_epochs + 1):
            loss = self.training_step(epoch)
            logger.info(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

            if epoch > 49 and epoch % 5 == 0:
                results = self.eval()
                logger.info("".join("{}:{} ".format(k, v) for k, v in results.items()))
                val_mrr, test_mrr = results["valid_mrr"], results["test_mrr"]
                if val_mrr > best_val_mrr:
                    accumulate_patience = 0
                    best_val_mrr = val_mrr
                    final_test_mrr = test_mrr
                else:
                    accumulate_patience += 1
                    if accumulate_patience >= 2:
                        break
            if self.trial is not None:
                self.trial.report(val_mrr, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        logger.info(f"best_val_mrr: {best_val_mrr:.4f}, final_test_mrr: {final_test_mrr:.4f}")
        return final_test_mrr, best_val_mrr


class LinkMLPTrainer:  # single GPU
    def __init__(self, args, data, split_idx, evaluator, **kwargs):
        self.args = args
        self.data = ToUndirected()(data)
        self.split_edge = split_idx
        self.evaluator = evaluator
        self.trial = kwargs.get("trial", None)
        self.predictor = self._prepare_model().to(self.device)
        self.optimizer = self._prepare_optimizer()

    @property
    def device(self):
        return torch.device(self.args.single_gpu if torch.cuda.is_available() else "cpu")

    def _prepare_model(self):
        return LinkPredictor(self.args)

    def _prepare_optimizer(self):
        return torch.optim.Adam(self.predictor.parameters(), lr=self.args.gnn_lr)

    def training_step(self, epoch):
        self.predictor.train()

        src_edge = self.split_edge["train"]["source_node"]
        dst_edge = self.split_edge["train"]["target_node"]

        total_loss, total_examples = 0, 0
        for perm in DataLoader(range(src_edge.size(0)), self.args.gnn_batch_size, shuffle=True):
            self.optimizer.zero_grad()
            src, dst = src_edge[perm], dst_edge[perm]
            pos_out = self.predictor(self.data.x[src].to(self.device), self.data.x[dst].to(self.device))
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            dst_neg = torch.randint(0, self.data.x.size(0), src.size(), dtype=torch.long)
            neg_out = self.predictor(self.data.x[src].to(self.device), self.data.x[dst_neg].to(self.device))
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def eval(self):
        self.predictor.eval()
        h = self.data.x

        def test_split(split):
            target = self.split_edge[split]["target_node"]
            source = self.split_edge[split]["source_node"]
            target_neg = self.split_edge[split]["target_node_neg"].contiguous()

            pos_preds = []
            for perm in DataLoader(range(source.size(0)), self.args.gnn_batch_size):
                src, dst = source[perm], target[perm]
                pos_preds += [self.predictor(h[src].to(self.device), h[dst].to(self.device)).squeeze().cpu()]
            pos_pred = torch.cat(pos_preds, dim=0)

            neg_preds = []
            source = source.view(-1, 1).repeat(1, 1000).view(-1)
            target_neg = target_neg.reshape(-1)
            for perm in DataLoader(range(source.size(0)), self.args.gnn_batch_size):
                src, dst_neg = source[perm], target_neg[perm]
                neg_preds += [self.predictor(h[src].to(self.device), h[dst_neg].to(self.device)).squeeze().cpu()]
            neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

            eval_results = self.evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})
            return {key[:-5]: value.mean().item() for key, value in eval_results.items()}

        valid_metrics = test_split("valid")
        test_metrics = test_split("test")

        return valid_metrics, test_metrics

    def train(self, return_value="valid"):
        best_val_mrr = 0
        best_val_metrics, final_test_metrics = {}, {}
        accumulate_patience = 0
        for epoch in range(1, self.args.gnn_epochs + 1):
            loss = self.training_step(epoch)
            logger.info(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

            if epoch % 5 == 0:
                valid_metrics, test_metrics = self.eval()
                logger.info(
                    f"Epoch: {epoch}, Valid Metrics:: "
                    + "".join("{}:{} ".format(k, v) for k, v in valid_metrics.items())
                )
                logger.info(
                    f"Epoch: {epoch}, Test Metrics:: " + "".join("{}:{} ".format(k, v) for k, v in test_metrics.items())
                )
                val_mrr = valid_metrics["mrr"]
                if val_mrr > best_val_mrr:
                    accumulate_patience = 0
                    best_val_mrr = val_mrr
                    best_val_metrics = valid_metrics
                    final_test_metrics = test_metrics
                else:
                    accumulate_patience += 1
                    if accumulate_patience >= 2:
                        break
            if self.trial is not None:
                self.trial.report(val_mrr, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        logger.info(
            f"Epoch: {epoch}, Best Valid Metrics:: "
            + "".join("{}:{} ".format(k, v) for k, v in best_val_metrics.items())
        )
        logger.info(
            f"Epoch: {epoch}, Final Test Metrics:: "
            + "".join("{}:{} ".format(k, v) for k, v in final_test_metrics.items())
        )
        return final_test_metrics, best_val_metrics
