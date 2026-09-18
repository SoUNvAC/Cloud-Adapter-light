import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase27_landsat_protocol import (  # noqa: E402
    binary_metrics,
    decode_cfmask_cloud,
    landsat_dn_to_rgb_scale,
    map_cloud_adapter_binary,
    map_cloud_adapter_three_class,
    map_truth_binary,
    map_truth_three_class,
    multiclass_metrics,
    paired_scene_bootstrap,
    scene_cloud_iou,
    update_binary_confusion,
    update_multiclass_confusion,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot whole-scene Landsat 8 C2 cross-sensor evaluation."
    )
    parser.add_argument("--audit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-scenes", type=int, default=48)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--core-size", type=int, default=384)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    parser.add_argument("--min-cloud-iou", type=float, default=50.0)
    parser.add_argument("--min-cloud-f1", type=float, default=65.0)
    parser.add_argument("--min-thin-recall", type=float, default=40.0)
    parser.add_argument("--min-scene-macro-iou-ci-low", type=float, default=40.0)
    parser.add_argument("--min-mean-difference-vs-cfmask", type=float, default=-10.0)
    return parser.parse_args()


def make_session(path, ort):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"device_id": 0})],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "ONNX Runtime CUDA provider failed: " + ", ".join(session.get_providers())
        )
    session.disable_fallback()
    model_input = session.get_inputs()
    model_output = session.get_outputs()
    if len(model_input) != 1 or model_input[0].name != "rgb_images":
        raise RuntimeError("Expected one ONNX input named rgb_images")
    if len(model_output) != 1 or model_output[0].name != "seg_logits":
        raise RuntimeError("Expected one ONNX output named seg_logits")
    if model_input[0].shape != [1, 3, 512, 512]:
        raise RuntimeError(f"Unexpected ONNX input shape: {model_input[0].shape}")
    if model_output[0].shape != [1, 4, 512, 512]:
        raise RuntimeError(f"Unexpected ONNX output shape: {model_output[0].shape}")
    return session


def resolve_path(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Audited path escapes data root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def toa_rgb(datasets, window, metadata):
    channels = []
    for band in (4, 3, 2):
        digital_number = datasets[f"b{band}"].read(
            1,
            window=window,
            boundless=True,
            fill_value=0,
            out_dtype="float32",
        )
        channels.append(
            landsat_dn_to_rgb_scale(
                digital_number,
                metadata[f"REFLECTANCE_MULT_BAND_{band}"],
                metadata[f"REFLECTANCE_ADD_BAND_{band}"],
                metadata["SUN_ELEVATION"],
            )
        )
    return np.ascontiguousarray(np.stack(channels, axis=0)[None], dtype=np.float32)


def evaluate_scene(scene, root, session, rasterio, tile_size, core_size):
    paths = {key: resolve_path(root, value) for key, value in scene["paths"].items()}
    datasets = {
        key: rasterio.open(paths[key]) for key in ("b2", "b3", "b4", "truth", "qa_pixel")
    }
    try:
        height = datasets["truth"].height
        width = datasets["truth"].width
        halo = (tile_size - core_size) // 2
        model_confusion = np.zeros((2, 2), dtype=np.int64)
        cfmask_confusion = np.zeros((2, 2), dtype=np.int64)
        three_class_confusion = np.zeros((3, 3), dtype=np.int64)
        nonfinite = 0
        shadow_pixels = 0
        valid_pixels = 0
        tiles = 0
        for row in range(0, height, core_size):
            output_height = min(core_size, height - row)
            for column in range(0, width, core_size):
                output_width = min(core_size, width - column)
                input_window = rasterio.windows.Window(
                    column - halo, row - halo, tile_size, tile_size
                )
                rgb = toa_rgb(
                    datasets, input_window, scene["toa_reflectance_metadata"]
                )
                logits = session.run(None, {"rgb_images": rgb})[0]
                nonfinite += int((~np.isfinite(logits)).sum())
                prediction = logits.argmax(axis=1)[0]
                prediction = prediction[
                    halo : halo + output_height,
                    halo : halo + output_width,
                ].astype(np.uint8)
                output_window = rasterio.windows.Window(
                    column, row, output_width, output_height
                )
                truth = datasets["truth"].read(1, window=output_window)
                qa_pixel = datasets["qa_pixel"].read(1, window=output_window)
                binary_truth, valid = map_truth_binary(truth)
                model_binary = map_cloud_adapter_binary(prediction)
                cfmask_binary = decode_cfmask_cloud(qa_pixel)
                update_binary_confusion(
                    model_confusion, model_binary, binary_truth, valid
                )
                update_binary_confusion(
                    cfmask_confusion, cfmask_binary, binary_truth, valid
                )
                update_multiclass_confusion(
                    three_class_confusion,
                    map_cloud_adapter_three_class(prediction),
                    map_truth_three_class(truth),
                    valid,
                )
                shadow_pixels += int(((prediction == 3) & valid).sum())
                valid_pixels += int(valid.sum())
                tiles += 1
        if valid_pixels == 0:
            raise RuntimeError(f"Scene {scene['scene_id']} contains no valid truth pixels")
        return {
            "scene_id": scene["scene_id"],
            "shape": [height, width],
            "tiles": tiles,
            "valid_pixels": valid_pixels,
            "model_confusion": model_confusion.tolist(),
            "cfmask_confusion": cfmask_confusion.tolist(),
            "three_class_confusion": three_class_confusion.tolist(),
            "model": binary_metrics(model_confusion),
            "cfmask": binary_metrics(cfmask_confusion),
            "model_scene_cloud_iou": scene_cloud_iou(model_confusion),
            "cfmask_scene_cloud_iou": scene_cloud_iou(cfmask_confusion),
            "shadow_prediction_rate": 100.0 * shadow_pixels / valid_pixels,
            "nonfinite_logits": nonfinite,
        }
    finally:
        for dataset in datasets.values():
            dataset.close()


def main():
    args = parse_args()
    if args.tile_size <= 0 or args.core_size <= 0 or args.core_size > args.tile_size:
        raise ValueError("Require 0 < core-size <= tile-size")
    if (args.tile_size - args.core_size) % 2:
        raise ValueError("tile-size minus core-size must be even")
    if args.tile_size != 512:
        raise ValueError("The frozen Phase 27 ONNX contract requires tile-size 512")
    try:
        import onnxruntime as ort
        import rasterio
    except ImportError as error:
        raise RuntimeError(
            "Phase 27 requires onnxruntime-gpu==1.18.0 and rasterio==1.3.10"
        ) from error

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("test_evaluated") is not False:
        raise RuntimeError("Phase 27 audit must pass before external evaluation")
    scenes = audit.get("scenes", [])
    if len(scenes) != args.expected_scenes:
        raise RuntimeError(
            f"Expected {args.expected_scenes} audited scenes, found {len(scenes)}"
        )
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    session = make_session(onnx_path, ort)
    root = Path(args.data_root)

    scene_results = []
    aggregate_model = np.zeros((2, 2), dtype=np.int64)
    aggregate_cfmask = np.zeros((2, 2), dtype=np.int64)
    aggregate_three = np.zeros((3, 3), dtype=np.int64)
    for index, scene in enumerate(scenes, start=1):
        result = evaluate_scene(
            scene, root, session, rasterio, args.tile_size, args.core_size
        )
        scene_results.append(result)
        aggregate_model += np.asarray(result["model_confusion"], dtype=np.int64)
        aggregate_cfmask += np.asarray(result["cfmask_confusion"], dtype=np.int64)
        aggregate_three += np.asarray(
            result["three_class_confusion"], dtype=np.int64
        )
        print(
            f"Evaluated {index}/{len(scenes)} {result['scene_id']}: "
            f"model IoU={result['model_scene_cloud_iou']:.3f}, "
            f"CFMask IoU={result['cfmask_scene_cloud_iou']:.3f}"
        )

    model_metrics = binary_metrics(aggregate_model)
    cfmask_metrics = binary_metrics(aggregate_cfmask)
    three_metrics = multiclass_metrics(aggregate_three)
    bootstrap = paired_scene_bootstrap(
        [row["model_scene_cloud_iou"] for row in scene_results],
        [row["cfmask_scene_cloud_iou"] for row in scene_results],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    total_nonfinite = sum(row["nonfinite_logits"] for row in scene_results)
    gates = {
        "all_48_scenes_evaluated": len(scene_results) == args.expected_scenes,
        "no_nonfinite_logits": total_nonfinite == 0,
        "pooled_cloud_iou": model_metrics["cloud_iou"] >= args.min_cloud_iou,
        "pooled_cloud_f1": model_metrics["cloud_f1"] >= args.min_cloud_f1,
        "thin_cloud_recall": three_metrics["class_recall"]["thin_cloud"]
        >= args.min_thin_recall,
        "scene_macro_iou_ci_low": bootstrap["model_scene_macro_95ci"][0]
        >= args.min_scene_macro_iou_ci_low,
        "mean_difference_vs_cfmask": bootstrap["paired_difference_mean"]
        >= args.min_mean_difference_vs_cfmask,
    }
    result = {
        "phase": 27,
        "dataset": audit.get("dataset"),
        "source": audit.get("source"),
        "sciencebase_item_id": audit.get("sciencebase_item_id"),
        "model": str(onnx_path),
        "input_preprocessing": "Landsat C2 Level-1 TOA RGB reflectance, clip [0,1], scale 0..255",
        "tiling": {
            "tile_size": args.tile_size,
            "core_size": args.core_size,
            "halo": (args.tile_size - args.core_size) // 2,
            "outside_scene_fill": 0,
        },
        "class_harmonization": {
            "binary": "Cloud-Adapter thick+thin=cloud; clear+shadow=non-cloud",
            "three_class": "clear+shadow=clear; thin=thin; thick=opaque",
            "cfmask": "QA_PIXEL bits 1(dilated),2(cirrus),3(cloud)",
        },
        "cloudsen12_test_evaluated": False,
        "external_test_evaluated": True,
        "thresholds": {
            "min_cloud_iou": args.min_cloud_iou,
            "min_cloud_f1": args.min_cloud_f1,
            "min_thin_recall": args.min_thin_recall,
            "min_scene_macro_iou_ci_low": args.min_scene_macro_iou_ci_low,
            "min_mean_difference_vs_cfmask": args.min_mean_difference_vs_cfmask,
        },
        "model_binary_metrics": model_metrics,
        "cfmask_binary_metrics": cfmask_metrics,
        "model_three_class_metrics": three_metrics,
        "model_confusion": aggregate_model.tolist(),
        "cfmask_confusion": aggregate_cfmask.tolist(),
        "model_three_class_confusion": aggregate_three.tolist(),
        "scene_bootstrap": bootstrap,
        "nonfinite_logits": total_nonfinite,
        "scenes": scene_results,
        "gates": gates,
        "passed": all(gates.values()),
        "stop_loss": {
            "on_failure": "close zero-shot transfer and start Phase 28",
            "next_direction": "freeze Phase 25 network; train only a sensor-input adapter on disjoint 38-Cloud training scenes; keep USGS C2 protocol and thresholds frozen",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("model_binary_metrics", "cfmask_binary_metrics", "model_three_class_metrics", "scene_bootstrap", "gates", "passed")}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    print("Phase 27 cross-sensor generalization gate: PASSED")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
