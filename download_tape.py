import os
import os.path as osp

import gdown
from ogb.utils.url import extract_zip


class TAPE:
    def __init__(self, root="./TAPE"):
        self.root = root
        self.url = "https://drive.google.com/u/0/uc?id=1A6mZSFzDIhJU795497R6mAAM2Y9qutI5&export=download"
        if not osp.exists(root):
            os.mkdir(root)

    def download(self):
        output = osp.join(self.root, "ogbn-arxiv.zip")
        if osp.exists(output) and osp.getsize(output) > 0:
            print(f"Using existing file {output}.")
        else:
            gdown.download(url=self.url, output=output, quiet=False, fuzzy=False)
            extract_zip(output, osp.join(self.root, "ogbn-arxiv"))
            os.remove(output)
