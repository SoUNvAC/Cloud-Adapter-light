import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "XavierJiezou/cloud-adapter-datasets"
zip_files = [
    "hrc_whu.zip",
    "gf12ms_whu_gf1.zip",
    "gf12ms_whu_gf2.zip",
    "cloudsen12_high_l1c.zip",
    "cloudsen12_high_l2a.zip",
    "l8_biome.zip",
]

# 数据集目录位于 Cloud-Adapter/data（脚本在 Cloud-Adapter/src/scripts 下）
data_dir = Path(__file__).resolve().parent.parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
print(f"Datasets will be extracted to: {data_dir}")

for zip_name in zip_files:
    print(f"\n[1/2] Downloading {zip_name} ...")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_name,
        repo_type="dataset",
        local_dir=str(data_dir),
        local_dir_use_symlinks=False,
    )
    zip_path = Path(zip_path)
    extract_dir = data_dir / zip_name.replace(".zip", "")
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[2/2] Extracting {zip_name} to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))
    print(f"Extracted: {extract_dir}")
    # 解压后删除 zip 以节省空间
    zip_path.unlink()
    print(f"Removed: {zip_path}")

print("\nAll datasets downloaded and extracted successfully.")
