import gc
import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, Linear
from torch.utils.data import DataLoader


class EnGCN(torch.nn.Module):
    def __init__(self, args, data, evaluator):
        super(EnGCN, self).__init__()
        # first try multiple weak learners
        self.model = MLP_SLE(args)

        self.evaluator = evaluator
        self.SLE_threshold = args.SLE_threshold
        self.use_label_mlp = args.use_label_mlp
        self.type_model = args.type_model
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.dim_hidden = args.dim_hidden
        self.dropout = args.dropout
        self.epochs = args.epochs
        self.multi_label = args.multi_label
        self.interval = args.eval_steps

        deg = data.adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        self.adj_t = deg_inv_sqrt.view(-1, 1) * data.adj_t * deg_inv_sqrt.view(1, -1)

        del data, deg, deg_inv_sqrt
        gc.collect()

    def forward(self, x):
        pass

    def propagate(self, x):
        return self.adj_t @ x

    def to(self, device):
        self.model.to(device)

    def train_and_test(self, input_dict):
        device, split_masks, x, y, loss_op = (
            input_dict["device"],
            input_dict["split_masks"],
            input_dict["x"],
            input_dict["y"],
            input_dict["loss_op"],
        )
        del input_dict
        gc.collect()
        self.to(device)
        if self.dataset in ["ogbn-papers100M"]:
            y = y.to(torch.long)
            x = x.to(torch.bfloat16)
            results = torch.zeros((y.size(0), self.num_classes), dtype=torch.bfloat16)
            y_emb = torch.zeros((y.size(0), self.num_classes), dtype=torch.bfloat16)
            y_emb[split_masks["train"]] = F.one_hot(y[split_masks["train"]], num_classes=self.num_classes).to(
                torch.bfloat16
            )
            # for self training
            pseudo_labels = torch.zeros_like(y).to(torch.long)
            pseudo_labels[split_masks["train"]] = y[split_masks["train"]]
            pseudo_split_masks = split_masks
        else:
            print(f"dtype y: {y.dtype}")
            results = torch.zeros(y.size(0), self.num_classes)
            y_emb = torch.zeros(y.size(0), self.num_classes)
            y_emb[split_masks["train"]] = F.one_hot(y[split_masks["train"]], num_classes=self.num_classes).to(
                torch.float
            )
            # for self training
            pseudo_labels = torch.zeros_like(y)
            pseudo_labels[split_masks["train"]] = y[split_masks["train"]]
            pseudo_split_masks = split_masks

        print("------ pseudo labels inited, rate: {:.4f} ------".format(pseudo_split_masks["train"].sum() / len(y)))

        for i in range(self.num_layers):
            # NOTE: here the num_layers should be the stages in original SAGN
            print(f"\n------ training weak learner with hop {i} ------")
            self.train_weak_learner(
                i,
                x,
                y_emb,
                pseudo_labels,
                y,  # the ground truth
                pseudo_split_masks,  # ['train'] is pseudo, valide and test are not modified
                device,
                loss_op,
            )
            self.model.load_state_dict(torch.load(f"./.cache/{self.type_model}_{self.dataset}_MLP_SLE.pt"))

            # make prediction
            use_label_mlp = False if i == 0 else self.use_label_mlp
            out = self.model.inference(x, y_emb, device, use_label_mlp)
            # self training: add hard labels
            val, pred = torch.max(F.softmax(out, dim=1), dim=1)
            SLE_mask = val >= self.SLE_threshold
            SLE_pred = pred[SLE_mask]
            # SLE_pred U y
            pseudo_split_masks["train"] = pseudo_split_masks["train"].logical_or(SLE_mask)
            pseudo_labels[SLE_mask] = SLE_pred
            pseudo_labels[split_masks["train"]] = y[split_masks["train"]]
            # update y_emb
            # y_emb[pseudo_split_masks["train"]] = F.one_hot(
            #     pseudo_labels[pseudo_split_masks["train"]], num_classes=self.num_classes
            # ).to(torch.float)
            del val, pred, SLE_mask, SLE_pred
            gc.collect()
            y_emb, x = self.propagate(y_emb), self.propagate(x)
            print(
                "------ pseudo labels updated, rate: {:.4f} ------".format(pseudo_split_masks["train"].sum() / len(y))
            )

            # NOTE: adaboosting (SAMME.R)
            out = F.log_softmax(out, dim=1)
            results += (self.num_classes - 1) * (out - torch.mean(out, dim=1).view(-1, 1))
            del out

        out, acc = self.evaluate(results, y, split_masks)
        print(
            f"Final train acc: {acc['train']*100:.4f}, "
            f"Final valid acc: {acc['valid']*100:.4f}, "
            f"Dianl test acc: {acc['test']*100:.4f}"
        )
        return acc["train"], acc["valid"], acc["test"]

    def evaluate(self, out, y, split_mask):
        acc = {}
        if self.evaluator:
            y_true = y.unsqueeze(-1)
            y_pred = out.argmax(dim=-1, keepdim=True)
            for phase in ["train", "valid", "test"]:
                acc[phase] = self.evaluator.eval(
                    {
                        "y_true": y_true[split_mask[phase]],
                        "y_pred": y_pred[split_mask[phase]],
                    }
                )["acc"]
        else:
            pred = out.argmax(dim=1).to("cpu")
            y_true = y
            correct = pred.eq(y_true)
            for phase in ["train", "valid", "test"]:
                acc[phase] = correct[split_mask[phase]].sum().item() / split_mask[phase].sum().item()
        return out, acc

    def train_weak_learner(self, hop, x, y_emb, pseudo_labels, origin_labels, split_mask, device, loss_op):
        # load self.xs[hop] to train self.mlps[hop]
        x_train = x[split_mask["train"]]
        pesudo_labels_train = pseudo_labels[split_mask["train"]]
        y_emb_train = y_emb[split_mask["train"]]
        train_set = torch.utils.data.TensorDataset(x_train, y_emb_train, pesudo_labels_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )
        best_valid_acc = 0.0
        use_label_mlp = self.use_label_mlp
        if hop == 0:
            use_label_mlp = False  # warm up
        for epoch in range(self.epochs):
            _loss, _train_acc = self.model.train_net(train_loader, loss_op, device, use_label_mlp)
            if (epoch + 1) % self.interval == 0:
                use_label_mlp = False if hop == 0 else self.use_label_mlp
                out = self.model.inference(x, y_emb, device, use_label_mlp)
                out, acc = self.evaluate(out, origin_labels, split_mask)
                print(
                    f"Model: {hop:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Train acc: {acc['train']*100:.4f}, "
                    f"Valid acc: {acc['valid']*100:.4f}, "
                    f"Test acc: {acc['test']*100:.4f}"
                )
                if acc["valid"] > best_valid_acc:
                    best_valid_acc = acc["valid"]
                    if not os.path.exists(".cache/"):
                        os.mkdir(".cache/")
                    torch.save(
                        self.model.state_dict(),
                        f"./.cache/{self.type_model}_{self.dataset}_MLP_SLE.pt",
                    )


class MLP_SLE(torch.nn.Module):
    def __init__(self, args) -> None:
        super(MLP_SLE, self).__init__()
        self.use_label_mlp = args.use_label_mlp
        self.label_mlp = GroupMLP(
            args.num_classes,
            args.dim_hidden,
            args.num_classes,
            args.num_heads,
            args.num_mlp_layers,
            args.dropout,
            normalization="batch" if args.use_batch_norm else "none",
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss_op = torch.nn.NLLLoss()

    def forward(self, x, y, use_label_mlp):
        out = self.base_mlp(x)
        if use_label_mlp:
            out += self.label_mlp(y).mean(1)
        return out

    def load(self):
        self.base_mlp.load_state_dict(torch.load(f"./.cache/{self.type_model}_{self.dataset}_base_mlp.pt"))
        self.label_mlp.load_state_dict(torch.load(f"./.cache/{self.type_model}_{self.dataset}_label_mlp.pt"))

    def train_net(self, train_loader, loss_op, device, use_label_mlp):
        self.train()
        total_correct, total_loss = 0, 0.0
        y_true, y_preds = [], []
        for x, y_emb, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_emb = y_emb.to(device)
            self.optimizer.zero_grad()
            out = self(x, y_emb, use_label_mlp)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            elif isinstance(loss_op, torch.nn.BCEWithLogitsLoss):
                y = y.float()
            loss = loss_op(out, y)
            # loss = loss * sample_weights  # weighted loss
            loss = loss.mean()
            total_loss += float(loss.item())
            loss.backward()
            self.optimizer.step()
            y_preds.append(out.argmax(dim=-1).detach().cpu())
            y_true.append(y.detach().cpu())

        y_true = torch.cat(y_true, 0)
        y_preds = torch.cat(y_preds, 0)
        total_correct = y_preds.eq(y_true).sum().item()
        train_acc = float(total_correct / y_preds.size(0))
        return float(total_loss), train_acc

    @torch.no_grad()
    def inference(self, x, y_emb, device, use_label_mlp):
        self.eval()
        loader = DataLoader(range(x.size(0)), batch_size=100000)
        outs = []
        for perm in loader:
            out = self(x[perm].to(device), y_emb[perm].to(device), use_label_mlp)
            outs.append(out.cpu())
        return torch.cat(outs, dim=0)


class Inner_MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, use_batch_norm):
        super(Inner_MLP, self).__init__()
        self.linear_list = torch.nn.ModuleList()
        self.batch_norm_list = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        self.linear_list.append(Linear(in_dim, hidden_dim))
        self.batch_norm_list.append(BatchNorm1d(hidden_dim))
        for _ in range(self.num_layers - 2):
            self.linear_list.append(Linear(hidden_dim, hidden_dim))
            self.batch_norm_list.append(BatchNorm1d(hidden_dim))
        self.linear_list.append(Linear(hidden_dim, out_dim))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.linear_list[i](x)
            if self.use_batch_norm:
                x = self.batch_norm_list[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        return self.linear_list[-1](x)


# classes from SAGN
class MultiHeadLinear(nn.Module):
    def __init__(self, in_feats, out_feats, n_heads, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(size=(n_heads, in_feats, out_feats)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(n_heads, 1, out_feats)))
        else:
            self.bias = None

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weight, self.bias):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias, -bound, bound)

    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain("relu")
    #     for weight in self.weight:
    #         nn.init.xavier_uniform_(weight, gain=gain)
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    def forward(self, x):
        # input size: [N, d_in] or [H, N, d_in]
        # output size: [H, N, d_out]
        if len(x.shape) == 3:
            x = x.transpose(0, 1)

        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x.transpose(0, 1)


# Modified multi-head BatchNorm1d layer
class MultiHeadBatchNorm(nn.Module):
    def __init__(self, n_heads, in_feats, momentum=0.1, affine=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert in_feats % n_heads == 0
        self._in_feats = in_feats
        self._n_heads = n_heads
        self._momentum = momentum
        self._affine = affine
        if affine:
            self.weight = nn.Parameter(torch.empty(size=(n_heads, in_feats // n_heads)))
            self.bias = nn.Parameter(torch.empty(size=(n_heads, in_feats // n_heads)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(size=(n_heads, in_feats // n_heads)))
        self.register_buffer("running_var", torch.ones(size=(n_heads, in_feats // n_heads)))
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        if self._affine:
            nn.init.zeros_(self.bias)
            for weight in self.weight:
                nn.init.ones_(weight)

    def forward(self, x, eps=1e-5):
        assert x.shape[1] == self._in_feats
        x = x.view(-1, self._n_heads, self._in_feats // self._n_heads)

        self.running_mean = self.running_mean.to(x.device)
        self.running_var = self.running_var.to(x.device)
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        if bn_training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            out = (x - mean) * torch.rsqrt(var + eps)
            self.running_mean = (1 - self._momentum) * self.running_mean + self._momentum * mean.detach()
            self.running_var = (1 - self._momentum) * self.running_var + self._momentum * var.detach()
        else:
            out = (x - self.running_mean) * torch.rsqrt(self.running_var + eps)
        if self._affine:
            out = out * self.weight + self.bias
        return out


class GroupMLP(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden,
        out_feats,
        n_heads,
        n_layers,
        dropout,
        input_drop=0.0,
        residual=False,
        normalization="batch",
    ):
        super(GroupMLP, self).__init__()
        self._residual = residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._n_heads = n_heads
        self._n_layers = n_layers

        self.input_drop = nn.Dropout(input_drop)

        if self._n_layers == 1:
            self.layers.append(MultiHeadLinear(in_feats, out_feats, n_heads))
        else:
            self.layers.append(MultiHeadLinear(in_feats, hidden, n_heads))
            if normalization == "batch":
                self.norms.append(MultiHeadBatchNorm(n_heads, hidden * n_heads))
                # self.norms.append(nn.BatchNorm1d(hidden * n_heads))
            if normalization == "layer":
                self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
            if normalization == "none":
                self.norms.append(nn.Identity())
            for i in range(self._n_layers - 2):
                self.layers.append(MultiHeadLinear(hidden, hidden, n_heads))
                if normalization == "batch":
                    self.norms.append(MultiHeadBatchNorm(n_heads, hidden * n_heads))
                    # self.norms.append(nn.BatchNorm1d(hidden * n_heads))
                if normalization == "layer":
                    self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
                if normalization == "none":
                    self.norms.append(nn.Identity())
            self.layers.append(MultiHeadLinear(hidden, out_feats, n_heads))
        if self._n_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        for head in range(self._n_heads):

            for layer in self.layers:

                nn.init.kaiming_uniform_(layer.weight[head], a=math.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight[head])
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(layer.bias[head], -bound, bound)
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")

        for head in range(self._n_heads):
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight[head], gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias[head])
        for norm in self.norms:
            norm.reset_parameters()
            # for norm in self.norms:
            #     norm.moving_mean[head].zero_()
            #     norm.moving_var[head].fill_(1)
            #     if norm._affine:
            #         nn.init.ones_(norm.scale[head])
            #         nn.init.zeros_(norm.offset[head])
        # print(self.layers[0].weight[0])

    def forward(self, x):
        x = self.input_drop(x)
        if len(x.shape) == 2:
            x = x.view(-1, 1, x.shape[1])
        if self._residual:
            prev_x = x
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)

            if layer_id < self._n_layers - 1:
                shape = x.shape
                x = x.flatten(1, -1)
                x = self.dropout(self.relu(self.norms[layer_id](x)))
                x = x.reshape(shape=shape)

            if self._residual:
                if x.shape[2] == prev_x.shape[2]:
                    x += prev_x
                prev_x = x

        return x
