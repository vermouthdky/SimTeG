from torch_geometric.transforms import ToSparseTensor

from .ogbn_arxiv import OgbnArxivWithText


def load_dataset(dataset, root="data", transform=None, pre_transform=None, tokenizer="microsoft/deberta-base"):
    if dataset == "ogbn-arxiv":
        return OgbnArxivWithText(root, transform, pre_transform, tokenizer)
    else:
        raise NotImplementedError
