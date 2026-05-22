import argparse
import shutil
from pathlib import Path

import nncf
import openvino as ov


TOKENIZER_FILES = [
    "config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX BGE-M3 to static INT8 OpenVINO IR.")
    parser.add_argument("--source", type=Path, default=Path("models/onnx"), help="Directory with model.onnx and tokenizer files.")
    parser.add_argument("--output", type=Path, default=Path("models/bge-m3-int8-ov"), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mode", choices=["int8_asym", "int8_sym"], default="int8_asym", help="NNCF weight compression mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = args.source / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    args.output.mkdir(parents=True, exist_ok=True)

    input_shape = [args.batch_size, args.max_length]
    print(f"Converting ONNX to OpenVINO IR with shape {input_shape}...")
    model = ov.convert_model(
        onnx_path,
        input=[
            ("input_ids", input_shape),
            ("attention_mask", input_shape),
        ],
    )

    mode = {
        "int8_asym": nncf.CompressWeightsMode.INT8_ASYM,
        "int8_sym": nncf.CompressWeightsMode.INT8_SYM,
    }[args.mode]
    print(f"Applying NNCF weight compression: {args.mode}...")
    compressed_model = nncf.compress_weights(model, mode=mode)

    model_xml = args.output / "model.xml"
    ov.save_model(compressed_model, model_xml, compress_to_fp16=False)
    print(f"Saved {model_xml}")

    for file_name in TOKENIZER_FILES:
        source_file = args.source / file_name
        if source_file.exists():
            shutil.copy2(source_file, args.output / file_name)

    print(f"static_shape=batch:{args.batch_size},max_length:{args.max_length}")
    print(f"weight_compression={args.mode}")
    print("Done.")


if __name__ == "__main__":
    main()
