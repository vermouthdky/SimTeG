import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from ogb.nodeproppred import Evaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import NeighborSampler
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm

from src.datasets import load_dataset
from src.models.modeling_TGroberta import TGRobertaConfig, TGRobertaForNodeClassification

logger = logging.getLogger(__name__)


def set_single_env(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_data(args):
    dataset = load_dataset(
        args.dataset, root=args.data_folder, transform=ToUndirected(), tokenizer=args.pretrained_model
    )
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    evaluator = Evaluator(name=args.dataset)
    for split in ["train", "valid", "test"]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = True
        data[f"{split}_mask"] = mask

    return data, split_idx, evaluator, dataset.processed_dir


def load_model(args):
    model_cached_dir = os.path.join(args.pretrained_dir, args.pretrained_model)
    if args.model_type == "TGRoberta":
        config = TGRobertaConfig.from_pretrained(args.pretrained_model)
        config.num_labels = args.num_labels
        config.gnn_num_layers = args.gnn_num_layers
        config.gnn_type = args.gnn_type
        config.gnn_dropout = args.gnn_dropout
        Model = TGRobertaForNodeClassification(config)
    else:
        raise NotImplementedError

    if not os.path.exists(model_cached_dir):
        model = Model.from_pretrained(
            args.pretrained_model,
            config=config,
            ignore_mismatched_sizes=True,
        )
        model.save_pretrained(model_cached_dir)
    else:
        model = Model.from_pretrained(model_cached_dir, config=config, ignore_mismatched_sizes=True)

    return model


def train(args):
    logger.info("---training-----")
    # setup running envs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    set_single_env(rank, world_size)
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    num_neighbors = [15, 10] + [5 for _ in range(args.gnn_num_layers - 2)]
    dist_sampler = DistributedSampler(dataset=data.edge_index, shuffle=True)
    train_loader = NeighborSampler(
        data.edge_index,
        sizes=num_neighbors,
        node_idx=split_idx["train"],
        batch_size=args.batch_size,
        shuffle=False if args.world_size > 1 else True,
        sampler=dist_sampler if args.world_size > 1 else None,
        num_workers=24,
    )

    # setup DDP model
    model = load_model(args)
    logging.info("load model: {}".format(args.model_type))
    model.cuda()
    if args.world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    if args.cont:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        model.load_state_dict(torch.load(args.ckpt_name, map_location=map_location))
        logging.info("load ckpt:{}".format(args.ckpt_name))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc, early_stop_count = 0.0, 0
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        loss = 0.0
        model.train()
        pbar = tqdm(train_loader, desc="Iteration", disable=args.disable_tqdm)
        for step, (batch_size, n_id, adjs) in enumerate(train_loader):
            adjs = [adj.to(rank) for adj in adjs]
            loss = model(
                batch_size,
                data.input_ids[n_id].to(rank),
                adjs,
                data.attention_mask[n_id][:batch_size].to(rank),
                data.y[n_id][:batch_size].to(rank),
                data.train_mask[n_id][:batch_size],
            )
            loss /= args.accum_interval
            pbar.update(1)
            loss.backward()
            if ((step + 1) % args.accum_interval == 0) or ((step + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            dist.barrier()

        pbar.close()
        if rank == 0 and epoch % args.eval_interval == 0:
            logging.info(
                "[{}] cost_time: {} epoch: {}, lr: {}, train_loss: {:.5f}".format(
                    rank,
                    time.time() - start_time,
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss,
                )
            )
            ckpt_path = args.path.join(
                args.ckpt_dir,
                "{}-epoch-{}.pt".format(args.model_type, epoch + 1),
            )
            torch.save(model.module.state_dict(), ckpt_path)
            train_acc, valid_acc, test_acc = test(model, data, evaluator, rank)
            logging.info(
                "train_acc: {:.4f}, valid_acc: {:.4f}, test_acc: {:.4f}".format(train_acc, valid_acc, test_acc)
            )
            if valid_acc > best_acc:
                ckpt_path = os.path.join(args.ckpt_dir, "{}-best.pt".format(args.model_type))
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Model {args.model_type} saved to {ckpt_path}")
                best_acc = valid_acc
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= 2:
                    start_time = time.time()
                    ckpt_path = os.path.join(args.ckpt_dir, "{}-best.pt".format(args.model_type))
                    model.load_state_dict(torch.load(ckpt_path), map_location="cpu")
                    logging.info("test best model")
                    train_acc, valid_acc, test_acc = test(model, data, evaluator)
                    logging.info(
                        "final train_acc: {:.4f}, final valid_acc: {:.4f}, final test_acc: {:.4f}".format(
                            train_acc, valid_acc, test_acc
                        )
                    )
                    logging.info("test time:{}".format(time.time() - start_time))
                    exit()
        dist.barrier()
    cleanup()


@torch.no_grad()
def test(model, data, evaluator, rank):
    model.eval()
    logits = model.inference(data.x, data.edge_index, rank)
    y_true = data.y.unsqueeze(-1)
    y_pred = logits.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({"y_true": y_true[data.train_mask], "y_pred": y_pred[data.train_mask]})["acc"]
    valid_acc = evaluator.eval({"y_true": y_true[data.valid_mask], "y_pred": y_pred[data.valid_mask]})["acc"]
    test_acc = evaluator.eval({"y_true": y_true[data.test_mask], "y_pred": y_pred[data.test_mask]})["acc"]
    return train_acc, valid_acc, test_acc


# def pretrain(rank, args):
#     set_single_env(rank, args.word_size)
#     model = load_model(args.model_type)
#     logging.info("load model {} for pretraining".format(args.model_type))
#     model = mode.cuda()

#     if args.word_size > 1:
#         model = DDP(model, device_ids=[rank], output_device=rank)

#     if args.cont:
#         map_locatoi
