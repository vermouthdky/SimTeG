from huggingface_hub import HfApi

repo_id = "vermouthdky/X_lminit"

api = HfApi()

api.upload_folder(
    folder_path="./out",
    repo_id=repo_id,
    repo_type="dataset",
    token="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",
    multi_commits=True,
    multi_commits_verbose=True,
)
