import gzip
import json
import logging
import os
import os.path as osp
import shutil

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils.url import download_url, extract_zip
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import ToSparseTensor
from transformers import AutoTokenizer

from ..utils import dist_barrier_context, is_dist, set_logging

logger = logging.getLogger(__name__)


class OgbWithText(InMemoryDataset):
    def __init__(
        self,
        name,
        meta_info,
        root="data",
        transform=None,
        pre_transform=None,
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        tokenize=True,
    ):
        self.name = name  ## original name, e.g., ogbn-proteins
        self.meta_info = meta_info
        self.dir_name = "_".join(self.name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.should_tokenize = tokenize
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True) if tokenize else None
        # check if the dataset is already processed with the same tokenizer
        rank = int(os.getenv("RANK", -1))
        with dist_barrier_context():
            super(OgbWithText, self).__init__(self.root, transform, pre_transform)
        if rank in [0, -1] and tokenize:
            self.save_metainfo()
        self.data, self.slices = torch.load(self.processed_paths[0])
        # add input_ids and attention_mask
        if self.should_tokenize:
            with dist_barrier_context():
                self._data.input_ids, self._data.attention_mask = self.mapping_and_tokenizing()

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        return osp.join("geometric_data_processed.pt")

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def _mapping_and_tokenizing(self):
        raise NotImplementedError

    def mapping_and_tokenizing(self):
        tokenizer_name = "_".join(self.tokenizer.name_or_path.split("/"))
        tokenized_path = osp.join(self.root, "processed", f"{tokenizer_name}.pt")
        if osp.exists(tokenized_path):
            logger.info("using cached tokenized data in {}".format(tokenized_path))
            text_encoding = torch.load(tokenized_path)
            return text_encoding["input_ids"], text_encoding["attention_mask"]
        input_ids, attention_mask = self._mapping_and_tokenizing()
        torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, tokenized_path)
        logger.info("saved tokenized data in {}".format(tokenized_path))
        return input_ids, attention_mask

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
