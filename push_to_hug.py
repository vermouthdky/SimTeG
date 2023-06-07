from huggingface_hub import HfApi, hf_hub_download, snapshot_download

repo_id = "vermouthdky/X_lminit"

# api = HfApi()

# api.upload_folder(
#     folder_path="./out/ogbn-products/e5-large/optuna_peft",
#     path_in_repo="ogbn-products/e5-large/optuna_peft",
#     repo_id=repo_id,
#     repo_type="dataset",
#     token="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",
#     # multi_commits=True,
#     # multi_commits_verbose=True,
# )

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    token="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",
    local_dir="../lambda_out",
    local_dir_use_symlinks=False,
    allow_patterns=["ogbn-products/e5-large/optuna_peft/best/cached_embs/*.pt"],
)
