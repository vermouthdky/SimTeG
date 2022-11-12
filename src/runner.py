import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from ogb.nodeproppred import Evaluator
from pynvml import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.transforms import ToUndirected

from src.datasets import load_dataset
from src.models.TGRoberta import TGRobertaConfig, TGRobertaForMaksedLM, TGRobertaForNodeClassification


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def set_single_env(rank, world_size):
    # initialize the process group
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_data(args):
    dataset = load_dataset(args.dataset, root=args.data_folder, transform=ToUndirected(), tokenizer=args.pretrained_model)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    evaluator = Evaluator(name=args.dataset)
    for split in ["train", "valid", "test"]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = True
        data[f"{split}_mask"] = mask
    return data, evaluator, dataset.processed_dir


def load_model(args):
    if args.model_type == "TGRoberta":
        model_cached_dir = os.path.join(args.pretrained_dir, args.pretrained_model)
        config = TGRobertaConfig.from_pretrained(args.pretrained_model)
        config.num_labels = args.num_labels
        if not os.path.exists(model_cached_dir):
            model = TGRobertaForNodeClassification.from_pretrained(args.pretrained_model, config=config, ignore_mismatched_sizes=True)
            model.save_pretrained(model_cached_dir)
        else:
            model = TGRobertaForNodeClassification.from_pretrained(model_cached_dir, config=config, ignore_mismatched_sizes=True)
    else:
        raise NotImplementedError
    return model


def train(rank, args):
    # setup running envs
    set_single_env(rank, args.world_size)
    # setup dataset: [ogbn-arxiv]
    data, evaluator, processed_dir = load_data(args)
    if args.world_size > 1:
        cluster_data = ClusterData(data, num_parts=args.num_parts, recursive=False, save_dir=processed_dir)
        dist_sampler = DistributedSampler(dataset=cluster_data, shuffle=True)
        train_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=False, sampler=dist_sampler, num_workers=24, pin_memory=True)
    else:
        cluster_data = ClusterData(data, num_parts=args.num_parts, recursive=False, save_dir=processed_dir)
        train_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=True, num_workers=24, pin_memory=True)

    # setup DDP model
    model = load_model(args)
    logging.info("load model: {}".format(args.model_type))
    model.to(rank)
    if args.world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    if args.cont:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        model.load_state_dict(torch.load(args.ckpt_name, map_location=map_location))
        logging.info("load ckpt:{}".format(args.ckpt_name))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc, early_stop_count = 0.0, 0
    for epoch in range(args.epochs):
        start_time = time.time()
        loss = 0.0
        model.train()
        for step, batch in enumerate(train_loader):
            # batch_loss = model()
            batch = batch.to(rank)
            output = model(batch.input_ids, batch.edge_index, batch.attention_mask, labels=batch.y, train_mask=batch.train_mask)
            batch_loss = output.loss
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            dist.barrier()

        if rank == 0 and epoch % args.eval_intercal == 0:
            logging.info(
                "[{}] cost_time: {} epoch: {}, lr: {}, train_loss: {:.5f}".format(
                    rank, time.time() - start_time, epoch, optimizer.param_groups[0]["lr"], loss
                )
            )
            ckpt_path = args.path.join(args.ckpt_dir, "{}-epoch-{}.pt".format(args.model_type, epoch + 1))
            torch.save(model.state_dict(), ckpt_path)
            train_acc, valid_acc, test_acc = test(model, data, evaluator)
            logging.info("train_acc: {:.4f}, valid_acc: {:.4f}, test_acc: {:.4f}".format(train_acc, valid_acc, test_acc))
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
                    logging.info("final train_acc: {:.4f}, final valid_acc: {:.4f}, final test_acc: {:.4f}".format(train_acc, valid_acc, test_acc))
                    logging.info("test time:{}".format(time.time() - start_time))
                    exit()
        dist.barrier()
    cleanup()


@torch.no_grad()
def test(model, data, evaluator):
    model.eval()
    dataloader = ClusterLoader(data, batch_size=5000, shuffle=False, num_workers=12, pin_memory=True)
    logits = []
    for batch in dataloader:
        out = model(batch.input_ids, batch.edge_index, batch.attention_mask, labels=batch.y, train_mask=batch.train_mask)
        batch_logits = out.logits
        logits.append(batch_logits.cpu())
    logits = torch.cat(logits, dim=0)
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
