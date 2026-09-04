import argparse
from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download


REPO_ID = "XavierJiezou/cloud-adapter-datasets"
ARCHIVE_NAME = "cloudsen12_high_l1c.zip"


def parse_args():
    parser = argparse.ArgumentParser(description="Download CloudSEN12-High L1C")
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"Unsafe path in dataset archive: {member.filename}")
        zip_file.extractall(destination)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    dataset_dir = data_dir / "cloudsen12_high_l1c"
    expected = [
        dataset_dir / "img_dir" / "train",
        dataset_dir / "ann_dir" / "train",
        dataset_dir / "img_dir" / "test",
        dataset_dir / "ann_dir" / "test",
    ]

    if all(path.is_dir() for path in expected):
        print(f"Dataset is already ready at {dataset_dir}")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=ARCHIVE_NAME,
            repo_type="dataset",
            local_dir=data_dir,
        )
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    safe_extract(archive, dataset_dir)

    missing = [str(path) for path in expected if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "Archive was extracted but the expected layout was not found: "
            + ", ".join(missing)
        )
    print(f"Dataset prepared at {dataset_dir}")


if __name__ == "__main__":
    main()
