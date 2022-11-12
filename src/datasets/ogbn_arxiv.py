import gzip
import os
import os.path as osp
import shutil

import numpy as np
import pandas as pd
import torch
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils.url import download_url, extract_zip
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import ToSparseTensor
from transformers import AutoTokenizer, BatchEncoding


class OgbnArxivWithText(InMemoryDataset):
    def __init__(
        self,
        root="data",
        transform=None,
        pre_transform=None,
        # tokenizer="allenai/longformer-base-4096",
        tokenizer="roberta-base",
    ):
        self.name = "ogbn-arxiv"  ## original name, e.g., ogbn-proteins
        self.dir_name = "_".join(self.name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        # if not osp.exists(self.root):
        #     os.makedirs(self.root)
        self.download_name = "arxiv"
        self.num_tasks = 1
        self.task_type = "multiclass classification"
        self.eval_metric = "acc"
        self.__num_classes__ = 40
        self.is_hetero = False
        self.add_inverse_edge = False
        self.additional_node_files = ["node_year"]
        self.additional_edge_files = []
        self.binary = False
        self.graph_url = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
        self.text_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        super(OgbnArxivWithText, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self):
        split_type = "time"

        path = osp.join(self.root, "split", split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

        train_idx = torch.from_numpy(pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header=None).values.T[0]).to(torch.long)
        valid_idx = torch.from_numpy(pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header=None).values.T[0]).to(torch.long)
        test_idx = torch.from_numpy(pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header=None).values.T[0]).to(torch.long)

        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        file_names = ["edge"]
        file_names.append("node-feat")
        return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join("geometric_data_processed.pt")

    def download(self):
        path = download_url(self.graph_url, self.original_root)
        extract_zip(path, self.original_root)
        text_path = download_url(self.text_url, os.path.join(self.original_root, "arxiv/raw"))
        os.unlink(path)
        shutil.rmtree(self.root)
        shutil.move(osp.join(self.original_root, self.download_name), self.root)

    def process(self):
        data = read_graph_pyg(
            self.raw_dir,
            add_inverse_edge=self.add_inverse_edge,
            additional_node_files=self.additional_node_files,
            additional_edge_files=self.additional_edge_files,
            binary=self.binary,
        )[0]
        ### adding prediction target
        node_label = pd.read_csv(
            osp.join(self.raw_dir, "node-label.csv.gz"),
            compression="gzip",
            header=None,
        ).values

        # detect if there is any nan
        if np.isnan(node_label).any():
            data.y = torch.from_numpy(node_label).to(torch.float32)
        else:
            data.y = torch.from_numpy(node_label).to(torch.long)

        data = data if self.pre_transform is None else self.pre_transform(data)
        text_encoding = self._mapping_and_tokenizing()
        data.input_ids = text_encoding.input_ids
        data.attention_mask = text_encoding.attention_mask

        print("Saving...")
        torch.save(self.collate([data]), self.processed_paths[0])

    def _mapping_and_tokenizing(self):
        """
        1. Mapping the paper ids to node ids
        2. Tokenize title and abstract
        """
        df = pd.read_csv(
            os.path.join(self.raw_dir, "titleabs.tsv.gz"),
            sep="\t",
            names=["paper id", "title", "abstract"],
            header=None
            # dtype={"paper id": np.int64, "title": str, "abstract": str},
        ).dropna()
        # Unzip the file `titleabs.tsv.gz` with gzip, otherwise it encounters the following bug if directly applying read csv
        # BUG: the first column's id is inplaced with 'titleabs.tsv'
        # BUG: Try to fix it manually
        df.iloc[0][0] = 200971
        df_mapping = pd.read_csv(os.path.join(self.root, "mapping/nodeidx2paperid.csv.gz"))
        df["abstitle"] = df["title"] + ". " + df["abstract"]
        df = df.drop(columns=["title", "abstract"])
        df = df.astype({"paper id": np.int64, "abstitle": str})
        df = df_mapping.merge(df, how="inner", on="paper id")
        text_encoding = self.tokenizer(
            df["abstitle"].values.tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return text_encoding

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


if __name__ == "__main__":
    pyg_dataset = OgbnArxivWithText()
    print(pyg_dataset[0])
    split_index = pyg_dataset.get_idx_split()
    print(split_index)
