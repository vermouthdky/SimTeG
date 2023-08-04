from huggingface_hub import HfApi, hf_hub_download, snapshot_download, upload_folder

repo_id = "vermouthdky/SimTeG"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="../lambda_out",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "ogbn-products/e5-large/optuna_peft/best/cached_embs/*.pt"
    ],
)
