import gc
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from src.args import parse_args
from src.dataset import load_data_bundle


def num_nodes(dataset):
    num_nodes_dict = {"ogbn-arxiv": 169343, "ogbn-products": 2449029}
    return num_nodes_dict[dataset]


def load_x(dataset, model, suffix):
    data, _, _ = load_data_bundle(dataset, root="../data", tokenizer=None, tokenize=False)
    if suffix != "ogb":
        bert_x = torch.load(osp.join("out", dataset, model, suffix, "cached_embs", "x_embs.pt"))
    else:
        bert_x = data.x
    return bert_x, data.y.view(-1)


def get_indices(label, num_samples):
    indices = []
    num_classes = label.max().item() + 1
    for i in range(0, num_classes, 8):
        indice = (label == i).nonzero(as_tuple=False).view(-1)
        if indice.size(0) < num_samples:
            indices.append(indice)
        else:
            idx = torch.randperm(indice.size(0))[:num_samples]
            indices.append(indice[idx])
    return torch.cat(indices, dim=0)


# Load your feature matrix into a PyTorch tensor (N, D)
def visualize(datasets: list, models: list, suffices: list, titles: list, num_samples_per_class=1000):
    assert len(models) == len(suffices)
    n_row, n_col = len(datasets), len(models)
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_row * 8, n_col * 5))
    for i, dataset in enumerate(datasets):
        # indices = torch.randperm(num_nodes(dataset))[:num_samples]
        indices = None
        for j, (model, suffix) in enumerate(zip(models, suffices)):
            x, label = load_x(dataset, model, suffix)
            if indices is None:
                indices = get_indices(label, num_samples_per_class)
            sampled_features = x[indices]
            sampled_labels = label[indices]

            # Run T-SNE on the sampled instances
            # if i == 1 and j == 1:
            #     __import__("ipdb").set_trace()
            tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0)
            tsne_features = tsne.fit_transform(sampled_features)
            if not osp.exists("./tsne_features"):
                os.mkdir("./tsne_features")
            tsne_features = np.concatenate([tsne_features, sampled_labels.view(-1, 1).numpy()], axis=1)
            data = pd.DataFrame(tsne_features)
            data.columns = ["x1", "x2", "label"]
            data.to_csv("./tsne_features/{}_{}_{}.csv".format(dataset, model, titles[j]))

            # Visualize the T-SNE embeddings using seaborn
            ax = axs[i, j]
            sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=sampled_labels, ax=ax)
            ax.set_title(titles[j])
            ax.margins(0)
            gc.collect()
    if not osp.exists("./figs"):
        os.mkdir("./figs")
    plt.savefig(
        "./figs/ogb-deberta-comparison.pdf",
        dpi=300,
        format="pdf",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.5,
    )


def single_visual(dataset="ogbn-arxiv", model="e5-large", suffix="optuna/best", num_nodes=1000):
    x, label = load_x(dataset, model, suffix)
    indices = torch.randperm(label.size(0))[:num_nodes]
    sampled_features = x[indices]
    sampled_labels = label[indices]

    tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0)
    tsne_features = tsne.fit_transform(sampled_features)

    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=sampled_labels)
    if not osp.exists("./figs"):
        os.mkdir("./figs")
    plt.savefig(
        f"./figs/{dataset}-{model}-{suffix.replace('/', '-')}.pdf",
        dpi=300,
        format="pdf",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.5,
    )


if __name__ == "__main__":
    datasets = ["ogbn-arxiv", "ogbn-products"]
    models = ["e5-large", "e5-large", "e5-large"]
    suffices = ["optuna_peft/best", "fix", "ogb"]
    titles = ["PEFL", "Fix", "OGB"]
    visualize(datasets, models, suffices, titles, num_samples_per_class=100)
