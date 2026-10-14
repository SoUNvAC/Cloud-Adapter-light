"""Freeze the Phase111 scene without opening imagery, labels, or model weights."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/phase111/phase111_20261014"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {
        "protocol": ROOT / "src/research_plans/PHASE111_OCM_RUNTIME_ACCEPTANCE_20261014.md",
        "phase104_manifest": ROOT / "src/research_results/phase104_20261013/object_manifest.json",
        "phase109_version": ROOT / "src/research_results/phase109_20261013/VERSION_SOURCE_LOCK.json",
        "phase109_weights": ROOT / "src/research_results/phase109_20261013/WEIGHTS_AND_LICENSE.json",
        "phase110_b8a": ROOT / "src/research_results/phase110_20261014/B8A_OBJECTS.json",
    }
    manifest = json.loads(paths["phase104_manifest"].read_text(encoding="utf-8"))
    b8a = json.loads(paths["phase110_b8a"].read_text(encoding="utf-8"))
    version = json.loads(paths["phase109_version"].read_text(encoding="utf-8"))
    weights = json.loads(paths["phase109_weights"].read_text(encoding="utf-8"))["weights"]
    ids = [p["product_id"] for p in manifest["products"]]
    assert len(ids) == len(set(ids)) == 16
    assert set(ids) == {p["product_id"] for p in b8a["products"]}
    ranking = sorted(
        ({"product_id": p, "selection_sha256": hashlib.sha256(
            ("PHASE111_SCENE|" + p).encode("utf-8")).hexdigest()} for p in ids),
        key=lambda row: row["selection_sha256"],
    )
    chosen = next(p for p in b8a["products"] if p["product_id"] == ranking[0]["product_id"])
    assert len(weights) == 2 and version["required_ensemble_size"] == 2
    lock = {
        "phase": 111,
        "selection_rule": "ascending SHA256(UTF8('PHASE111_SCENE|' + product_id))",
        "ranking": ranking,
        "selected_product_id": ranking[0]["product_id"],
        "selected_b8a": chosen,
        "source_files_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
                                for p in paths.values()},
        "code_commit": version["code_commit"],
        "hf_model_revision": version["hf_model_revision"],
        "weights": weights,
        "scope": "single scene, no labels, no performance evaluation, no fitting",
        "resource_caps": {"host_min_free_bytes": 4000000000,
                          "environment_peak_bytes_including_caches": 4000000000,
                          "new_intermediate_bytes": 4000000000,
                          "weight_band_download_bytes_including_retries": 4000000000,
                          "ram_bytes": 16 * 1024**3, "vram_bytes": 20 * 1024**3,
                          "gpu_seconds": 3600, "main_and_repeat_calls": 256},
        "inference": {"channels": ["B04", "B03", "B8A"], "dtype": "float32",
                      "patch_size": 1000, "overlap": 300, "batch_size": 1,
                      "mosaic": "CPU", "export_confidence": True,
                      "softmax_output": True, "compile_models": False,
                      "repeat_first_valid_patches": 2, "repeat_extra_calls": 4,
                      "repeat_max_absolute_difference": 1e-5},
        "resource_gate": "must be measured and passed before installation/download/forward",
        "inherited_declarations_are_not_actual_content_validation": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "PRE_EXECUTION_LOCK.json"
    body = (json.dumps(lock, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if destination.exists():
        assert destination.read_bytes() == body, "Existing lock differs; refusing overwrite"
    else:
        destination.write_bytes(body)
    print(json.dumps({"selected_product_id": lock["selected_product_id"],
                      "selection_sha256": ranking[0]["selection_sha256"],
                      "b8a_declared_bytes": chosen["bytes"],
                      "lock_sha256": digest(destination)}))


if __name__ == "__main__":
    main()
