import json
import logging
import os
import os.path as osp
import shutil

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from torch_geometric.data import InMemoryDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class OgblCitation2WithText(InMemoryDataset):
    def __init__(
        self,
        root="data",
        transform=None,
        pre_transform=None,
        tokenizer="microsoft/deberta-base",
        tokenize=True,
    ):
        self.name = "ogbl-citation2"  ## original name, e.g., ogbl-ppa

        self.dir_name = "_".join(self.name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.meta_info = {
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
            "tokenizer": tokenizer,
        }
        rank = int(os.getenv("RANK", -1))
        self.binary = self.meta_info["binary"]
        self.is_hetero = self.meta_info["is_hetero"]
        self.download_name = self.meta_info["download_name"]
        if rank in [0, -1]:  # check the metainfo
            metainfo = self.load_metainfo()
            if metainfo is not None and tokenize and metainfo["tokenizer"] != tokenizer:
                logger.critical("The tokenizer is changed. Reprocessing the dataset.")
                shutil.rmtree(osp.join(self.root, "processed"), ignore_errors=True)
        if rank not in [0, -1]:
            dist.barrier()
        super(OgblCitation2WithText, self).__init__(self.root, transform, pre_transform)
        if rank == 0:
            dist.barrier()
        if rank in [0, -1] and tokenize:
            self.save_metainfo()
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_edge_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info["split"]

        path = osp.join(self.root, "split", split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

        train = replace_numpy_with_torchtensor(torch.load(osp.join(path, "train.pt")))
        valid = replace_numpy_with_torchtensor(torch.load(osp.join(path, "valid.pt")))
        test = replace_numpy_with_torchtensor(torch.load(osp.join(path, "test.pt")))

        return {"train": train, "valid": valid, "test": test}

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ["edge_index_dict.npz"]
            else:
                return ["data.npz"]
        else:
            if self.is_hetero:
                return ["num-node-dict.csv.gz", "triplet-type-list.csv.gz"]
            else:
                file_names = ["edge"]
                if self.meta_info["has_node_attr"] == "True":
                    file_names.append("node-feat")
                if self.meta_info["has_edge_attr"] == "True":
                    file_names.append("edge-feat")
                return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join("geometric_data_processed.pt")

    def download(self):
        url = self.meta_info["url"]
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print("Stop downloading.")
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info["add_inverse_edge"] == "True"

        if self.meta_info["additional node files"] == "None":
            additional_node_files = []
        else:
            additional_node_files = self.meta_info["additional node files"].split(",")

        if self.meta_info["additional edge files"] == "None":
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info["additional edge files"].split(",")

        if self.is_hetero:
            data = read_heterograph_pyg(
                self.raw_dir,
                add_inverse_edge=add_inverse_edge,
                additional_node_files=additional_node_files,
                additional_edge_files=additional_edge_files,
                binary=self.binary,
            )[0]
        else:
            data = read_graph_pyg(
                self.raw_dir,
                add_inverse_edge=add_inverse_edge,
                additional_node_files=additional_node_files,
                additional_edge_files=additional_edge_files,
                binary=self.binary,
            )[0]

        data = data if self.pre_transform is None else self.pre_transform(data)

        print("Saving...")
        torch.save(self.collate([data]), self.processed_paths[0])

    def save_metainfo(self):
        w_path = osp.join(self.root, "processed/meta_info.json")
        with open(w_path, "w") as outfile:
            json.dump(self.meta_info, outfile)

    def load_metainfo(self):
        r_path = osp.join(self.root, "processed/meta_info.json")
        if not osp.exists(r_path):
            return None
        return json.loads(open(r_path).read())

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


if __name__ == "__main__":
    pyg_dataset = OgblCitation2WithText()
    split_edge = pyg_dataset.get_edge_split()
    print(pyg_dataset[0])
    exit(-1)

    pyg_dataset = PygLinkPropPredDataset(name="ogbl-ddi")
    split_edge = pyg_dataset.get_edge_split()
    print(pyg_dataset[0])
    print(pyg_dataset[0].num_nodes)
    print(split_edge["train"])
    print(split_edge["test"])
    pyg_dataset = PygLinkPropPredDataset(name="ogbl-wikikg")
    split_edge = pyg_dataset.get_edge_split()
    print(pyg_dataset[0])
    print(pyg_dataset[0].num_nodes)
    print(split_edge["train"])
    print(split_edge["test"])
    pyg_dataset = PygLinkPropPredDataset(name="ogbl-citation")
    split_edge = pyg_dataset.get_edge_split()
    print(split_edge["train"])
    print(split_edge["test"])
    pyg_dataset = PygLinkPropPredDataset(name="ogbl-ppa")
    split_edge = pyg_dataset.get_edge_split()
    print(split_edge["train"])
    print(split_edge["test"])
    pyg_dataset = PygLinkPropPredDataset(name="ogbl-collab")
    split_edge = pyg_dataset.get_edge_split()
    print(split_edge["train"])
    print(split_edge["test"])
