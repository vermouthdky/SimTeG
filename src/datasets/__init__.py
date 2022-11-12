from .ogbn_arxiv import OgbnArxivWithText
from torch_geometric.transforms import ToSparseTensor


def load_dataset(dataset, root="data", transform=None, pre_transform=None, tokenizer="roberta-base"):
    if dataset == "ogbn-arxiv":
        return OgbnArxivWithText(root, transform, pre_transform, tokenizer)
    else:
        raise NotImplementedError
