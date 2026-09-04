from huggingface_hub import list_repo_files, get_hf_file_metadata, hf_hub_url

repo_id = "XavierJiezou/cloud-adapter-datasets"
files = list_repo_files(repo_id, repo_type="dataset")
print(f"Files in {repo_id}:")
total_size = 0
for f in files:
    url = hf_hub_url(repo_id=repo_id, filename=f, repo_type="dataset")
    try:
        meta = get_hf_file_metadata(url)
        size_mb = meta.size / 1024 / 1024 if meta.size else 0
        total_size += meta.size or 0
        print(f"  {f}: {size_mb:.2f} MB")
    except Exception as e:
        print(f"  {f}: failed to get size ({e})")
print(f"Total size: {total_size / 1024 / 1024:.2f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")
