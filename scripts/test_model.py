import argparse
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BGE-M3 OpenVINO IR model.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge-m3-int8-ov"))
    parser.add_argument("--device", default="CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_xml = args.model_dir / "model.xml"
    if not model_xml.exists():
        print(f"ERROR: model.xml not found in {args.model_dir}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)

    texts = [
        "What is BGE M3?",
        "BGE M3 is an embedding model.",
    ]

    errors = []
    core = ov.Core()
    model = core.read_model(model_xml)

    expected_input_shapes = {
        "input_ids": [1, 512],
        "attention_mask": [1, 512],
    }
    actual_input_shapes = {}
    for model_input in model.inputs:
        partial_shape = model_input.get_partial_shape()
        actual_input_shapes[model_input.get_any_name()] = (
            list(model_input.get_shape()) if partial_shape.is_static else str(partial_shape)
        )
    if actual_input_shapes != expected_input_shapes:
        errors.append(
            f"Input shapes mismatch: got {actual_input_shapes}, expected {expected_input_shapes}"
        )

    output_shapes = {
        output.get_any_name(): list(output.get_shape())
        for output in model.outputs
        if output.get_partial_shape().is_static
    }
    if output_shapes.get("sentence_embedding") != [1, 1024]:
        errors.append(
            "sentence_embedding shape mismatch: "
            f"got {output_shapes.get('sentence_embedding')}, expected [1, 1024]"
        )

    if errors:
        print("VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    compiled = core.compile_model(model, args.device)

    embeddings = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        available_inputs = {inp.get_any_name() for inp in compiled.inputs}
        inputs = {k: v for k, v in inputs.items() if k in available_inputs}

        result = compiled(inputs)

        embeddings.append(result["sentence_embedding"][0])

    embedding = np.stack(embeddings, axis=0)

    expected_shape = (2, 1024)
    if embedding.shape != expected_shape:
        errors.append(f"Shape mismatch: got {embedding.shape}, expected {expected_shape}")

    if embedding.dtype != np.float32:
        errors.append(f"Dtype mismatch: got {embedding.dtype}, expected float32")

    norms = np.linalg.norm(embedding, axis=-1)
    for i, norm_val in enumerate(norms):
        if not (0.1 < norm_val < 100.0):
            errors.append(f"Embedding {i} L2 norm out of range: {norm_val:.4f}")

    sim = np.dot(embedding[0], embedding[1]) / (norms[0] * norms[1])
    if sim < 0.3:
        errors.append(f"Cosine similarity too low: {sim:.4f} (expected > 0.3)")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"device={args.device}")
    print("output_name=sentence_embedding")
    print(f"output_shape={list(embedding.shape)}")
    print(f"output_dtype={embedding.dtype}")
    print(f"l2_norms={[float(n) for n in norms]}")
    print(f"cosine_similarity={float(sim):.4f}")
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
