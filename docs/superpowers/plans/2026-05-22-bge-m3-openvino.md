# BGE-M3 OpenVINO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full pipeline for converting BAAI/bge-m3 from HF PyTorch to OpenVINO IR INT8 + CI/CD for automated HF publishing.

**Architecture:** Three Python scripts (export ONNX, convert to OpenVINO, test) + GitHub Actions workflow triggered on push to main that downloads, converts, tests, and publishes to Hugging Face.

**Tech Stack:** Python 3.10+, PyTorch, ONNX, OpenVINO, NNCF, Transformers, huggingface_hub

---

### Task 1: Project scaffold

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `scripts/requirements.txt`
- Create: `README.md`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
eggs/
*.egg
.eggs/

# Virtual environment
.venv/
venv/
.env/

# Model artifacts (large files, generated)
models/
*.onnx
*.xml
*.bin

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
```

Write to: `.gitignore`

- [ ] **Step 2: Create LICENSE**

```text
MIT License

Copyright (c) 2024 Beijing Academy of Artificial Intelligence (BAAI)
Copyright (c) 2026 The bge-m3-openvino contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Write to: `LICENSE`

- [ ] **Step 3: Create scripts/requirements.txt**

```text
huggingface_hub>=0.23
nncf>=2.14
numpy>=1.26
onnx>=1.16
openvino>=2025.0
torch>=2.2
transformers>=4.40
sentencepiece
tokenizers>=0.19
```

Write to: `scripts/requirements.txt`

- [ ] **Step 4: Create README.md**

```markdown
# BGE-M3 OpenVINO

[BGE-M3](https://huggingface.co/BAAI/bge-m3) embedding model converted to OpenVINO IR with INT8 quantization. Optimized for Intel NPU/CPU/GPU inference.

## Quick Start

### Install

```bash
pip install -r scripts/requirements.txt
```

### Download and convert

```bash
python scripts/export_onnx.py --output models/onnx
python scripts/convert_openvino.py --source models/onnx --output models/bge-m3-openvino-int8
```

### Test

```bash
python scripts/test_model.py --model-dir models/bge-m3-openvino-int8
```

### Use

```python
import openvino as ov
import numpy as np
from transformers import AutoTokenizer

core = ov.Core()
model = core.compile_model("models/bge-m3-openvino-int8/model.xml", "CPU")
tokenizer = AutoTokenizer.from_pretrained("models/bge-m3-openvino-int8")

text = "What is BGE M3?"
encoded = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=512)
result = model({"input_ids": encoded["input_ids"].astype(np.int64), "attention_mask": encoded["attention_mask"].astype(np.int64)})
embedding = result["sentence_embedding"]  # shape: [1, 1024]
print(embedding.shape)
```

## Model Details

| Property | Value |
|----------|-------|
| Base model | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| License | MIT |
| Input shape | [1, 512] (static) |
| Input names | `input_ids` (int64), `attention_mask` (int64) |
| Output | `sentence_embedding` [1, 1024] float32 |
| Quantization | INT8 asymmetric (NNCF) |
| Embedding dim | 1024 |
| Max length | 512 tokens |

## CI/CD

On push to `main`, GitHub Actions:
1. Downloads BAAI/bge-m3 from Hugging Face
2. Exports to ONNX
3. Converts to OpenVINO IR with INT8 quantization
4. Runs validation tests
5. Uploads the model bundle to a Hugging Face repository

## Attribution

This is a converted version of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) by Beijing Academy of Artificial Intelligence, licensed under MIT.
```

Write to: `README.md`

- [ ] **Step 5: Commit**

```bash
git add .gitignore LICENSE scripts/requirements.txt README.md
git commit -m "feat: project scaffold — gitignore, license, deps, readme"
```

---

### Task 2: ONNX export script

**Files:**
- Create: `scripts/export_onnx.py`

- [ ] **Step 1: Create scripts/export_onnx.py**

```python
import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


TOKENIZER_FILES = [
    "config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


class BgeM3OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        cls_embedding = last_hidden[:, 0, :]
        return last_hidden, cls_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BGE-M3 to ONNX with static shapes.")
    parser.add_argument("--model-id", default="BAAI/bge-m3", help="Hugging Face model ID.")
    parser.add_argument("--output", type=Path, default=Path("models/onnx"), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--opset", type=int, default=15, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id}...")
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()

    wrapper = BgeM3OnnxWrapper(model)
    batch = args.batch_size
    length = args.max_length

    dummy_input_ids = torch.zeros((batch, length), dtype=torch.long)
    dummy_attention_mask = torch.zeros((batch, length), dtype=torch.long)

    onnx_path = args.output / "model.onnx"
    print(f"Exporting to {onnx_path} (shape: [{batch}, {length}])...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["token_embeddings", "sentence_embedding"],
        dynamic_axes={},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Saved {onnx_path}")

    for file_name in TOKENIZER_FILES:
        source_file = tokenizer.vocab_files.get(file_name) or tokenizer.init_kwargs.get(file_name)
        if source_file and Path(source_file).exists():
            shutil.copy2(source_file, args.output / file_name)
        else:
            src = Path(tokenizer.name_or_path) / file_name
            if src.exists():
                shutil.copy2(src, args.output / file_name)

    tokenizer.save_pretrained(args.output)
    print(f"Tokenizer files saved to {args.output}")


if __name__ == "__main__":
    main()
```

Write to: `scripts/export_onnx.py`

- [ ] **Step 2: Commit**

```bash
git add scripts/export_onnx.py
git commit -m "feat: add ONNX export script for BGE-M3"
```

---

### Task 3: Convert to OpenVINO IR with INT8

**Files:**
- Create: `scripts/convert_openvino.py`

- [ ] **Step 1: Create scripts/convert_openvino.py**

```python
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
    parser.add_argument("--output", type=Path, default=Path("models/bge-m3-openvino-int8"), help="Output directory.")
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
```

Write to: `scripts/convert_openvino.py`

- [ ] **Step 2: Commit**

```bash
git add scripts/convert_openvino.py
git commit -m "feat: add OpenVINO IR conversion script with NNCF INT8"
```

---

### Task 4: Test script

**Files:**
- Create: `scripts/test_model.py`

- [ ] **Step 1: Create scripts/test_model.py**

```python
import argparse
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BGE-M3 OpenVINO IR model.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge-m3-openvino-int8"))
    parser.add_argument("--device", default="CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_xml = args.model_dir / "model.xml"
    if not model_xml.exists():
        model_xml = args.model_dir / "openvino_model.xml"
    if not model_xml.exists():
        print(f"ERROR: model.xml not found in {args.model_dir}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)

    texts = [
        "What is BGE M3?",
        "BGE M3 is an embedding model.",
    ]

    encoded = tokenizer(
        texts,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    core = ov.Core()
    compiled = core.compile_model(model_xml, args.device)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    available_inputs = {inp.get_any_name() for inp in compiled.inputs}
    inputs = {k: v for k, v in inputs.items() if k in available_inputs}

    result = compiled(inputs)

    outputs_by_name = {out.get_any_name(): result[out] for out in compiled.outputs}
    embedding = outputs_by_name.get("sentence_embedding")
    if embedding is None:
        output_name, embedding = next(iter(outputs_by_name.items()))
        print(f"WARNING: 'sentence_embedding' not found, using '{output_name}'")
        output_name_str = output_name
    else:
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
```

Write to: `scripts/test_model.py`

- [ ] **Step 2: Commit**

```bash
git add scripts/test_model.py
git commit -m "feat: add model validation test script"
```

---

### Task 5: Documentation

**Files:**
- Create: `docs/model_preparation.md`

- [ ] **Step 1: Create docs/model_preparation.md**

Write to `docs/model_preparation.md` with the exact content from the spec, adapted for users:

```markdown
# BGE-M3 Model Preparation for OpenVINO

This document describes how to convert the [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) embedding model to OpenVINO IR with INT8 quantization.

## Prerequisites

- Python 3.10+
- pip

```bash
pip install -r scripts/requirements.txt
```

## Pipeline

### Step 1: Export to ONNX

Downloads the model from Hugging Face and exports to ONNX with static shapes [1, 512].

```bash
python scripts/export_onnx.py --output models/onnx
```

Arguments:
| Arg | Default | Description |
|-----|---------|-------------|
| `--model-id` | `BAAI/bge-m3` | Hugging Face model ID |
| `--output` | `models/onnx` | Output directory |
| `--batch-size` | `1` | Static batch dimension |
| `--max-length` | `512` | Static sequence length |

Outputs: `model.onnx`, tokenizer files (`tokenizer.json`, `config.json`, etc.)

### Step 2: Convert to OpenVINO IR with INT8

Converts ONNX model to OpenVINO IR and applies NNCF INT8 asymmetric weight compression.

```bash
python scripts/convert_openvino.py --source models/onnx --output models/bge-m3-openvino-int8
```

Arguments:
| Arg | Default | Description |
|-----|---------|-------------|
| `--source` | `models/onnx` | Directory with `model.onnx` |
| `--output` | `models/bge-m3-openvino-int8` | Output directory |
| `--batch-size` | `1` | Static batch dimension |
| `--max-length` | `512` | Static sequence length |
| `--mode` | `int8_asym` | NNCF compression mode (`int8_asym` or `int8_sym`) |

Outputs: `model.xml`, `model.bin`, tokenizer files

### Step 3: Test

Validates the converted model by running inference on CPU and checking output shape, dtype, and embedding quality.

```bash
python scripts/test_model.py --model-dir models/bge-m3-openvino-int8
```

Arguments:
| Arg | Default | Description |
|-----|---------|-------------|
| `--model-dir` | `models/bge-m3-openvino-int8` | Model bundle directory |
| `--device` | `CPU` | OpenVINO device (CPU, GPU, NPU) |

### One-liner

```bash
python scripts/export_onnx.py --output models/onnx && \
python scripts/convert_openvino.py --source models/onnx --output models/bge-m3-openvino-int8 && \
python scripts/test_model.py --model-dir models/bge-m3-openvino-int8
```

## Model Signature

| Direction | Name | Shape | Type |
|-----------|------|-------|------|
| Input | `input_ids` | [1, 512] | int64 |
| Input | `attention_mask` | [1, 512] | int64 |
| Output | `token_embeddings` | [1, 512, 1024] | float32 |
| Output | `sentence_embedding` | [1, 1024] | float32 |

## Bundle Files

| File | Description |
|------|-------------|
| `model.xml` | OpenVINO IR graph |
| `model.bin` | OpenVINO IR weights (INT8 compressed) |
| `tokenizer.json` | Hugging Face tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `sentencepiece.bpe.model` | SentencePiece model |
| `special_tokens_map.json` | Special token mappings |
| `config.json` | Model configuration |

## License

MIT. Original model by [BAAI](https://huggingface.co/BAAI/bge-m3).

## Reference

Based on the conversion pipeline from [ai2npu](https://github.com/...) project.
```

Write to: `docs/model_preparation.md`

- [ ] **Step 2: Commit**

```bash
git add docs/model_preparation.md
git commit -m "docs: add model preparation documentation"
```

---

### Task 6: CI/CD — GitHub Actions workflow

**Files:**
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Create .github/workflows/publish.yml**

```yaml
name: Convert and Publish to Hugging Face

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-${{ runner.os }}-${{ hashFiles('scripts/requirements.txt') }}
          restore-keys: pip-${{ runner.os }}-

      - name: Cache Hugging Face models
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-${{ runner.os }}-bge-m3-${{ github.run_id }}
          restore-keys: hf-${{ runner.os }}-bge-m3-

      - name: Install dependencies
        run: pip install -r scripts/requirements.txt

      - name: Export to ONNX
        run: python scripts/export_onnx.py --output models/onnx

      - name: Convert to OpenVINO IR
        run: python scripts/convert_openvino.py --source models/onnx --output models/bge-m3-openvino-int8

      - name: Test model
        run: python scripts/test_model.py --model-dir models/bge-m3-openvino-int8

      - name: Upload to Hugging Face
        if: success()
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python -c "
          from huggingface_hub import HfApi, create_repo
          import os

          api = HfApi(token=os.environ['HF_TOKEN'])
          repo_id = os.environ.get('HF_REPO', 'bge-m3-openvino')

          create_repo(repo_id=repo_id, exist_ok=True, token=os.environ['HF_TOKEN'])

          api.upload_folder(
              folder_path='models/bge-m3-openvino-int8',
              repo_id=repo_id,
              repo_type='model',
          )
          print(f'Published to https://huggingface.co/{repo_id}')
          "
```

Write to: `.github/workflows/publish.yml`

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/publish.yml
git commit -m "ci: add GitHub Actions workflow for HF publishing"
```

---

### Task 7: Final review and README update

- [ ] **Step 1: Add CI/CD and HF info to README**

Append to `README.md`:

```markdown
## CI/CD Pipeline

This repository uses GitHub Actions to automatically convert and publish the model on every push to `main`.

### Required Secrets

Set these in your GitHub repository **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Hugging Face API token with write access |
| `HF_REPO` | Target Hugging Face repository (e.g., `your-username/bge-m3-openvino`) |

### Manual Trigger

Go to **Actions → Convert and Publish to Hugging Face → Run workflow**.
```

Edit `README.md` — add this section before `## Attribution`.

- [ ] **Step 2: Verify project structure**

```bash
ls -la
ls -la scripts/
ls -la .github/workflows/
ls -la docs/
```

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: add CI/CD setup instructions to README"
```
