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

    model_xml = args.model_dir / "openvino_model.xml"
    if not model_xml.exists():
        model_xml = args.model_dir / "model.xml"
    if not model_xml.exists():
        print(f"ERROR: model.xml not found in {args.model_dir}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)

    texts = [
        "What is BGE M3?",
        "BGE M3 is an embedding model.",
    ]

    core = ov.Core()
    compiled = core.compile_model(model_xml, args.device)

    embeddings = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="np",
            padding=True,
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

        outputs_by_name = {out.get_any_name(): result[out] for out in compiled.outputs}
        emb_raw = outputs_by_name.get("sentence_embedding") or outputs_by_name.get("last_hidden_state")
        if emb_raw is None:
            _, emb_raw = next(iter(outputs_by_name.items()))
        emb = emb_raw[0, 0] if emb_raw.ndim == 3 else emb_raw[0]
        embeddings.append(emb)

    embedding = np.stack(embeddings, axis=0)
    output_name_str = "sentence_embedding"

    errors = []

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
    print(f"output_name={output_name_str}")
    print(f"output_shape={list(embedding.shape)}")
    print(f"output_dtype={embedding.dtype}")
    print(f"l2_norms={[float(n) for n in norms]}")
    print(f"cosine_similarity={float(sim):.4f}")
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
