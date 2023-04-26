import gc
import gzip
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
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils.url import download_url, extract_zip
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from ..utils import is_dist, mkdirs_if_not_exists

logger = logging.getLogger(__name__)


class OgbnProductsWithText(InMemoryDataset):
    def __init__(
        self, root="data", transform=None, pre_transform=None, tokenizer="microsoft/deberta-base", tokenize=True
    ):
        self.name = "ogbn-products"  ## original name, e.g., ogbn-proteins
        self.dir_name = "_".join(self.name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        mkdirs_if_not_exists(self.root)
        self.meta = {
            "download_name": "products",
            "num_tasks": 1,
            "task_type": "multiclass classification",
            "eval_metric": "acc",
            "num_classes": 47,
            "is_hetero": False,
            "add_inverse_edge": True,
            "additional_node_files": [],
            "additional_edge_files": [],
            "binary": False,
            "graph_url": "http://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
            "text_url": "https://drive.google.com/u/0/uc?id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN&export=download",
            "tokenizer": tokenizer,
        }
        self.should_tokenize = tokenize
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True) if tokenize else None
        # check if the dataset is already processed with the same tokenizer
        rank = int(os.environ["RANK"]) if is_dist() else -1
        metainfo = self.load_metainfo()
        if metainfo is not None and tokenize and metainfo["tokenizer"] != tokenizer:
            logger.info("The tokenizer is changed. Re-processing the dataset.")
            shutil.rmtree(os.path.join(self.root, "processed"), ignore_errors=True)
        super(OgbnProductsWithText, self).__init__(self.root, transform, pre_transform)
        if rank in [0, -1] and tokenize:
            self.save_metainfo()
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self):
        split_type = "sales_ranking"

        path = osp.join(self.root, "split", split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

        train_idx = torch.from_numpy(
            pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header=None).values.T[0]
        ).to(torch.long)
        valid_idx = torch.from_numpy(
            pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
        ).to(torch.long)
        test_idx = torch.from_numpy(
            pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header=None).values.T[0]
        ).to(torch.long)

        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    @property
    def num_classes(self):
        return self.meta["num_classes"]

    @property
    def raw_file_names(self):
        file_names = ["edge"]
        file_names.append("node-feat")
        return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join("geometric_data_processed.pt")

    def download(self):
        path = download_url(self.meta["graph_url"], self.original_root)
        extract_zip(path, self.original_root)
        # download text data Amazon-3M
        output = osp.join(self.original_root, f"{self.meta['download_name']}/raw/Amazon-3M.raw.zip")
        if osp.exists(output) and osp.getsize(output) > 0:  # pragma: no cover
            logger.info(f"Using exist file {output}")
        else:
            gdown.download(url=self.meta["text_url"], output=output, quiet=False, fuzzy=False)
            extract_zip(output, osp.join(self.original_root, f"{self.meta['download_name']}/raw"))
            os.remove(output)

        os.unlink(path)
        shutil.rmtree(self.root)
        shutil.move(osp.join(self.original_root, self.meta["download_name"]), self.root)

    def process(self):
        data = read_graph_pyg(
            self.raw_dir,
            add_inverse_edge=self.meta["add_inverse_edge"],
            additional_node_files=self.meta["additional_node_files"],
            additional_edge_files=self.meta["additional_edge_files"],
            binary=self.meta["binary"],
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
        if self.should_tokenize:
            data.input_ids, data.attention_mask = self._mapping_and_tokenizing()
        print("Saving...")
        torch.save(self.collate([data]), self.processed_paths[0])
        del data
        gc.collect()

    def _mapping_and_tokenizing(self):
        """
        1. Mapping the paper ids to node ids
        2. Tokenize title and abstract
        3. save the flag of tokenizer
        """
        text_dir = osp.join(self.raw_dir, "Amazon-3M.raw")
        df = self._read_meta_product(text_dir)  # 二维表
        df.replace(np.nan, "", inplace=True)
        df["titlecontent"] = df["title"] + ". " + df["content"]
        df = df.drop(columns=["title", "content"])
        df.rename(columns={"uid": "asin"}, inplace=True)

        df_mapping = pd.read_csv(os.path.join(self.root, "mapping/nodeidx2asin.csv.gz"))
        df = df_mapping.merge(df, how="left", on="asin")
        text_list = df["titlecontent"].values.tolist()
        input_ids, attention_mask, truncated_size = [], [], 10000
        print("Tokenizing...")
        for i in tqdm(range(0, len(df), truncated_size)):
            j = min(len(text_list), i + truncated_size)
            _encodings = self.tokenizer(text_list[i:j], padding=True, truncation=True, return_tensors="pt")
            input_ids.append(_encodings.input_ids)
            attention_mask.append(_encodings.attention_mask)
        input_ids, attention_mask = torch.cat(input_ids, dim=0), torch.cat(attention_mask, dim=0)
        return input_ids, attention_mask

    def _read_product_json(self, raw_text_path):
        # modified from https://github.com/AndyJZhao/GLEM/blob/main/src/utils/data/preprocess_product.py
        if not osp.exists(osp.join(raw_text_path, "trn.json")):
            trn_json = osp.join(raw_text_path, "trn.json.gz")
            trn_json = gzip.GzipFile(trn_json)
            open(osp.join(raw_text_path, "trn.json"), "wb+").write(trn_json.read())
            os.unlink(osp.join(raw_text_path, "trn.json.gz"))
            tst_json = osp.join(raw_text_path, "tst.json.gz")
            tst_json = gzip.GzipFile(tst_json)
            open(osp.join(raw_text_path, "tst.json"), "wb+").write(tst_json.read())
            os.unlink(osp.join(raw_text_path, "tst.json.gz"))
            os.unlink(osp.join(raw_text_path, "Yf.txt"))  # New

        i = 1
        for file in ["trn.json", "tst.json"]:
            file_path = osp.join(raw_text_path, file)
            print(f"transfering {file_path}")
            with open(file_path, "r") as file_in:
                title = []
                for line in file_in.readlines():
                    dic = json.loads(line)

                    dic["title"] = dic["title"].strip('"\n')
                    title.append(dic)
                name_attribute = ["uid", "title", "content"]
                writercsv = pd.DataFrame(columns=name_attribute, data=title)
                writercsv.to_csv(
                    osp.join(raw_text_path, f"product" + str(i) + ".csv"), index=False, encoding="utf_8_sig"
                )  # index=False不输出索引值
                i = i + 1

    def _read_meta_product(self, raw_text_path):
        # copied from https://github.com/AndyJZhao/GLEM/blob/main/src/utils/data/preprocess_product.py
        # 针对read_meta_data
        if not osp.exists(osp.join(raw_text_path, f"products.csv")):
            self._read_product_json(raw_text_path)  # 弄出json文件
            path_product1 = osp.join(raw_text_path, f"product1.csv")
            path_product2 = osp.join(raw_text_path, f"product2.csv")
            pro1 = pd.read_csv(path_product1)
            pro2 = pd.read_csv(path_product2)
            file = pd.concat([pro1, pro2])
            file.drop_duplicates()
            file.to_csv(osp.join(raw_text_path, f"products.csv"), index=False, sep=" ")
        else:
            file = pd.read_csv(osp.join(raw_text_path, "products.csv"), sep=" ")

        return file

    def save_metainfo(self):
        w_path = osp.join(self.root, "processed/meta_info.json")
        with open(w_path, "w") as outfile:
            json.dump(self.meta, outfile)

    def load_metainfo(self):
        r_path = osp.join(self.root, "processed/meta_info.json")
        if not osp.exists(r_path):
            return None
        return json.loads(open(r_path).read())

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


if __name__ == "__main__":
    pyg_dataset = OgbnProductsWithText(root="../data")
    print(pyg_dataset[0])
    split_index = pyg_dataset.get_idx_split()
    print(split_index)
