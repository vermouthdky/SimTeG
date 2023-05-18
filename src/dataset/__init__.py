from .ogbn_arxiv import OgbnArxivWithText
from .ogbn_products import OgbnProductsWithText


def load_dataset(dataset, root="data", tokenizer="microsoft/deberta-base", tokenize=True):
    datasets = {"ogbn-arxiv": OgbnArxivWithText, "ogbn-products": OgbnProductsWithText}
    assert dataset in datasets.keys()
    return datasets[dataset](root=root, tokenizer=tokenizer, tokenize=tokenize)
