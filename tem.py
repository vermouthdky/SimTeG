import os

import pandas as pd

citation_text = "../data/ogbl_citation2_text/raw/idx_title_abs.csv.gz"
citation_mapping = "../data/ogbl_citation2_text/mapping/nodeidx2paperid.csv.gz"
paper_mapping = "../data/nodeidx2paperid.csv.gz"

citation_text_df = pd.read_csv(citation_text)

citation_idx_df = pd.read_csv(citation_mapping)
paper_idx_df = pd.read_csv(paper_mapping)


__import__("ipdb").set_trace()
