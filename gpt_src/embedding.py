import os.path as osp
from typing import List, Union

import numpy as np
import openai
import pandas as pd
import torch
from tqdm import tqdm

api_key = "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"

# data parameters
root = "../../data/"
dataset = "ogbn_arxiv"
data_dir = osp.join(root, dataset)

# model parameters
model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000


def get_embedding(text: Union[List[str], str], model="text-embedding-ada-002"):
    text = list(text) if text is str else text
    res = openai.Embedding.create(api_key=api_key, input=text, model=model)
    embedding_list = [d["embedding"] for d in res.data]
    return embedding_list


def process_text():
    df = pd.read_csv(
        osp.join(data_dir, "raw", "titleabs.tsv.gz"),
        sep="\t",
        names=["paper id", "title", "abstract"],
        header=None,
        compression="gzip",
    ).dropna()
    # Unzip the file `titleabs.tsv.gz` with gzip, otherwise it encounters the following bug if directly applying read csv
    # BUG: the first column's id is inplaced with 'titleabs.tsv'. Try to fix it manually
    df.iloc[0][0] = 200971
    df_mapping = pd.read_csv(osp.join(data_dir, "mapping/nodeidx2paperid.csv.gz"))
    df["abstitle"] = "Title: " + df["title"] + "; " + "Abstract: " + df["abstract"]
    df = df.drop(columns=["title", "abstract"])
    df = df.astype({"paper id": np.int64, "abstitle": str})
    df = df_mapping.merge(df, how="inner", on="paper id")
    df.to_csv(osp.join(data_dir, "raw", "aligned_titleabs.csv.gz"))
    return df


def main():
    table_dir = osp.join(data_dir, "raw", "aligned_titleabs.csv.gz")
    if not osp.exists(table_dir):
        print("merging and aligning title abs with node idx ... ")
        process_text()

    embs = []
    for df in tqdm(pd.read_csv(table_dir, chunksize=5)):
        abstitle = df.abstitle.tolist()
        tem_emb = get_embedding(abstitle)

        __import__("ipdb").set_trace()
        torch.save(torch.tensor(embs, dtype=torch.float32), osp.join(data_dir, "raw", "embs.pt"))

    # torch.save(torch.tensor(embs, dtype=torch.float32), osp.join(data_dir, "raw", "embs.pt"))


if __name__ == "__main__":
    main()
    # df = pd.read_csv(osp.join(data_dir, "raw", "embs.csv.gz"))
    # __import__("ipdb").set_trace()
