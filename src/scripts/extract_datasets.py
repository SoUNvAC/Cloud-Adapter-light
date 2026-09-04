import os
import zipfile
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent.parent / "data"
print(f"Extracting datasets in: {data_dir}")

zip_files = [
    "hrc_whu.zip",
    "gf12ms_whu_gf1.zip",
    "gf12ms_whu_gf2.zip",
    "cloudsen12_high_l1c.zip",
    "cloudsen12_high_l2a.zip",
    "l8_biome.zip",
]

for zip_name in zip_files:
    zip_path = data_dir / zip_name
    if not zip_path.exists():
        print(f"[SKIP] {zip_name} not found")
        continue
    extract_dir = data_dir / zip_name.replace(".zip", "")
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting {zip_name} -> {extract_dir.name}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))
    print(f"[DONE] {zip_name}")
    # 解压后删除 zip 节省空间
    zip_path.unlink()
    print(f"[REMOVED] {zip_name}")

print("\nAll datasets extracted.")
