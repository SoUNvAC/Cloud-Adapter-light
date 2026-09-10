import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append ArgMax and uint8 Cast to a static segmentation ONNX"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("input-size must be positive")
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path} (use --force)")

    model = onnx.load(str(input_path), load_external_data=False)
    if len(model.graph.output) != 1:
        raise RuntimeError(f"Expected one graph output, found {len(model.graph.output)}")
    logits_name = model.graph.output[0].name
    existing_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    for name in ("seg_class_ids", "seg_mask"):
        if name in existing_names:
            raise RuntimeError(f"ONNX value name already exists: {name}")

    model.graph.node.extend(
        [
            helper.make_node(
                "ArgMax",
                inputs=[logits_name],
                outputs=["seg_class_ids"],
                name="SemanticArgMax",
                axis=1,
                keepdims=0,
                select_last_index=0,
            ),
            helper.make_node(
                "Cast",
                inputs=["seg_class_ids"],
                outputs=["seg_mask"],
                name="SemanticMaskUint8",
                to=TensorProto.UINT8,
            ),
        ]
    )
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                "seg_mask",
                TensorProto.UINT8,
                [1, args.input_size, args.input_size],
            )
        ]
    )

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    reloaded = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(reloaded)
    graph_output = reloaded.graph.output[0]
    if graph_output.name != "seg_mask":
        raise RuntimeError(f"Unexpected saved output: {graph_output.name}")

    print(f"Source logits ONNX: {input_path}")
    print("Added: ArgMax(axis=1, keepdims=0) -> Cast(uint8)")
    print(f"Output: seg_mask = uint8[1,{args.input_size},{args.input_size}]")
    print(f"ONNX checker: passed ({len(reloaded.graph.node)} nodes)")
    print(f"Artifact: {output_path.stat().st_size / 2**20:.2f} MiB")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
