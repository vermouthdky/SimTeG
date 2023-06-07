import json
import logging
import os
import os.path as osp
import shutil

import gdown
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm
from transformers import AutoTokenizer

from ..utils import dist_barrier_context, set_logging
from .ogb_with_text import OgbWithText

logger = logging.getLogger(__name__)


class OgblCitation2WithText(OgbWithText):
    def __init__(
        self,
        root="data",
        transform=None,
        pre_transform=None,
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        tokenize=True,
    ):
        name = "ogbl-citation2-text"  ## original name, e.g., ogbl-ppa

        meta_info = {
            "download_name": "citation-v2",
            "task_type": "link prediction",
            "eval_metric": "mrr",
            "add_inverse_edge": False,
            "version": 1,
            "has_node_attr": True,
            "has_edge_attr": False,
            "split": "time",
            "additional_node_files": ["node_year"],
            "additional_edge_files": [],
            "is_hetero": False,
            "binary": False,
            "graph_url": "http://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip",
            "text_url": "https://drive.google.com/u/0/uc?id=19_hkbBUDFZTvQrM0oMbftuXhgz5LbIZY&export=download",
            "tokenizer": tokenizer,
        }
        super(OgblCitation2WithText, self).__init__(
            name, meta_info, root, transform, pre_transform, tokenizer, tokenize
        )

    def get_edge_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info["split"]

        path = osp.join(self.root, "split", split_type)

        split_idx = {"train": None, "valid": None, "test": None}
        for key, item in split_idx.items():
            split_idx[key] = replace_numpy_with_torchtensor(torch.load(osp.join(path, f"{key}.pt")))

        # subset with subset_node_idx and relable nodes
        subset_node_idx = self._get_subset_node_idx()

        num_nodes, edge_index = 0, {}
        for key in split_idx.keys():
            edge_index[key] = torch.stack([split_idx[key]["source_node"], split_idx[key]["target_node"]], dim=0)
            if key in ["valid", "test"]:
                edge_index[key] = torch.cat([edge_index[key], split_idx[key]["target_node_neg"].t()], dim=0)
            num_nodes = max(num_nodes, int(edge_index[key].max()) + 1)

        def subset_and_relabeling(edge_index, subset, num_nodes):
            node_mask = index_to_mask(subset, size=num_nodes)
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            edge_index = edge_index[:, edge_mask]
            # relabel
            node_idx = torch.zeros(num_nodes, dtype=torch.long)
            node_idx[node_mask] = torch.arange(node_mask.sum())
            edge_index = node_idx[edge_index]
            # BUG: some out-of-subset edges may appear, relabel them to 0
            return edge_index

        for key in split_idx.keys():
            edge_index[key] = subset_and_relabeling(edge_index[key], subset_node_idx, num_nodes)
            split_idx[key]["source_node"], split_idx[key]["target_node"] = edge_index[key][0], edge_index[key][1]
            if key in ["valid", "test"]:
                split_idx[key]["target_node_neg"] = edge_index[key][2:].t()
        return split_idx

    @property
    def raw_file_names(self):
        if self.meta_info["binary"]:
            if self.meta_info["is_hetero"]:
                return ["edge_index_dict.npz"]
            else:
                return ["data.npz"]
        else:
            if self.meta_info["is_hetero"]:
                return ["num-node-dict.csv.gz", "triplet-type-list.csv.gz"]
            else:
                file_names = ["edge"]
                if self.meta_info["has_node_attr"] == "True":
                    file_names.append("node-feat")
                if self.meta_info["has_edge_attr"] == "True":
                    file_names.append("edge-feat")
                return [file_name + ".csv.gz" for file_name in file_names]

    def download(self):
        graph_url = self.meta_info["graph_url"]
        if decide_download(graph_url):
            path = download_url(graph_url, self.original_root)
            extract_zip(path, self.original_root)
            # download text data from google drive
            output = osp.join(self.original_root, self.meta_info["download_name"], "raw", "idx_title_abs.csv.gz")
            if osp.exists(output) and osp.getsize(output) > 0:
                logger.info(f"Using existing file {output}")
            else:
                gdown.download(url=self.meta_info["text_url"], output=output, quiet=False, fuzzy=False)
            # cleanup
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.meta_info["download_name"]), self.root)
        else:
            logger.warning("Stop downloading.")
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info["add_inverse_edge"] == "True"

        data = read_graph_pyg(
            self.raw_dir,
            add_inverse_edge=add_inverse_edge,
            additional_node_files=self.meta_info["additional_node_files"],
            additional_edge_files=self.meta_info["additional_edge_files"],
            binary=self.meta_info["binary"],
        )[0]

        data = data if self.pre_transform is None else self.pre_transform(data)
        subset_node_idx = self._get_subset_node_idx()
        data = data.subgraph(subset_node_idx)

        print("Saving...")
        torch.save(self.collate([data]), self.processed_paths[0])

    def _mapping_and_tokenizing(self):
        df = pd.read_csv(osp.join(self.raw_dir, "idx_title_abs.csv.gz"))
        df["abstitle"] = "title: " + df["title"] + "; " + "abstract: " + df["abstract"]
        input_ids, attention_mask, truncated_size = [], [], 10000
        text_list = df["abstitle"].values.tolist()
        print("Tokenizing...")
        for i in tqdm(range(0, len(df), truncated_size)):
            j = min(len(text_list), i + truncated_size)
            _encodings = self.tokenizer(text_list[i:j], padding=True, truncation=True, return_tensors="pt")
            input_ids.append(_encodings.input_ids)
            attention_mask.append(_encodings.attention_mask)
        input_ids, attention_mask = torch.cat(input_ids, dim=0), torch.cat(attention_mask, dim=0)
        return input_ids, attention_mask

    def _get_subset_node_idx(self):
        df = pd.read_csv(osp.join(self.raw_dir, "idx_title_abs.csv.gz"))
        df.astype({"node idx": np.int64})
        node_idx = torch.tensor(df["node idx"].values.tolist(), dtype=torch.long)
        return node_idx


if __name__ == "__main__":
    set_logging()
    pyg_dataset = OgblCitation2WithText("../data")
    data = pyg_dataset.data
    split_edge = pyg_dataset.get_edge_split()
    for split in split_edge.keys():
        source_max = split_edge[split]["source_node"].max()
        target_max = split_edge[split]["target_node"].max()
        logger.info(f"max source node id in {split}: {source_max}")
        logger.info(f"max target node id in {split}: {target_max}")
    num_edge_list = []
    for split in split_edge.keys():
        num_edge = split_edge[split]["source_node"].size(0)
        num_edge_list.append(num_edge)
    print(num_edge_list)
    num_edges = sum(num_edge_list)
    print(num_edges)
    print(data.x.size(0) / num_edges)
    print([_ / num_edges for _ in num_edge_list])
    print(data.x.shape)
    exit(-1)
