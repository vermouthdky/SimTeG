import os
import os.path as osp

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from src.args import parse_args
from src.dataset import load_dataset
from src.utils import dataset2foldername

path = "out/ogbn-arxiv/GBert/"


def load_data(use_bert_x=False):
    args = parse_args()
    dataset = load_dataset("ogbn-arxiv", root=args.data_folder, tokenizer=args.pretrained_repo)
    data = dataset.data
    labels = data.y.view(-1)
    x = data.x
    saved_dir = osp.join(args.data_folder, dataset2foldername(args.dataset), "processed", "bert_x.pt")
    bert_x = torch.load(saved_dir)
    return x, bert_x, labels


# Load your feature matrix into a PyTorch tensor (N, D)
def visualize():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    x, bert_x, labels = load_data()
    indices = torch.randperm(labels.size(0))[:500]
    for i, x in enumerate([x, bert_x]):
        sampled_features = x[indices]
        sampled_labels = labels[indices]

        # Run T-SNE on the sampled instances
        tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0)
        tsne_features = tsne.fit_transform(sampled_features)

        # Visualize the T-SNE embeddings using seaborn
        ax = axs[i]
        sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=sampled_labels, ax=ax)
        subtitle = "OGB" if i == 0 else "DeBERTa"
        ax.set_title(subtitle)
        ax.margins(0)
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


if __name__ == "__main__":
    visualize()
