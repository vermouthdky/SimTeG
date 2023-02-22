from torch_geometric.transforms import Compose, ToSparseTensor, ToUndirected

from .ogbn_arxiv import OgbnArxivWithText


def load_dataset(dataset, root="data", tokenizer="microsoft/deberta-base"):
    if dataset == "ogbn-arxiv":
        return OgbnArxivWithText(root, tokenizer=tokenizer)
    else:
        raise NotImplementedError
