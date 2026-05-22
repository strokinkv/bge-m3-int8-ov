# BGE-M3 OpenVINO

[Русская версия](README.ru.md)

[BGE-M3](https://huggingface.co/BAAI/bge-m3) embedding model converted to OpenVINO IR with INT8 quantization. Optimized for Intel NPU/CPU/GPU inference.

## Quick Start

### Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts/requirements.txt
```

### Download and convert

```bash
python scripts/export_onnx.py --output models/onnx
python scripts/convert_openvino.py --source models/onnx --output models/bge-m3-int8-ov
```

### Test

```bash
python scripts/test_model.py --model-dir models/bge-m3-int8-ov
```

### One-liner

```bash
python scripts/export_onnx.py --output models/onnx && \
python scripts/convert_openvino.py --source models/onnx --output models/bge-m3-int8-ov && \
python scripts/test_model.py --model-dir models/bge-m3-int8-ov
```

### Use

```python
import openvino as ov
import numpy as np
from transformers import AutoTokenizer

core = ov.Core()
model = core.compile_model("models/bge-m3-int8-ov/model.xml", "CPU")
tokenizer = AutoTokenizer.from_pretrained("models/bge-m3-int8-ov")

text = "What is BGE M3?"
encoded = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=512)
result = model({"input_ids": encoded["input_ids"].astype(np.int64), "attention_mask": encoded["attention_mask"].astype(np.int64)})
embedding = result["sentence_embedding"]  # shape: [1, 1024]
print(embedding.shape)
```

## Pipeline

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. ONNX export | `export_onnx.py` | `BAAI/bge-m3` from Hugging Face | `model.onnx` [1,512] |
| 2. OpenVINO + INT8 | `convert_openvino.py` | `model.onnx` | `model.xml`/`model.bin` INT8 |
| 3. Validation | `test_model.py` | OpenVINO IR | Shape, dtype, L2-norm check |

### Step 1: export_onnx.py

| Arg | Default | Description |
|-----|---------|-------------|
| `--model-id` | `BAAI/bge-m3` | Hugging Face model ID |
| `--output` | `models/onnx` | Output directory |
| `--batch-size` | `1` | Static batch dimension |
| `--max-length` | `512` | Static sequence length |

### Step 2: convert_openvino.py

| Arg | Default | Description |
|-----|---------|-------------|
| `--source` | `models/onnx` | Directory with `model.onnx` |
| `--output` | `models/bge-m3-int8-ov` | Output directory |
| `--batch-size` | `1` | Static batch dimension |
| `--max-length` | `512` | Static sequence length |
| `--mode` | `int8_asym` | NNCF compression mode (`int8_asym` or `int8_sym`) |

### Step 3: test_model.py

| Arg | Default | Description |
|-----|---------|-------------|
| `--model-dir` | `models/bge-m3-int8-ov` | Model bundle directory |
| `--device` | `CPU` | OpenVINO device (CPU, GPU, NPU) |

## Project Structure

```
bge-m3-openvino/
├── scripts/
│   ├── export_onnx.py            # PyTorch → ONNX
│   ├── convert_openvino.py       # ONNX → OpenVINO IR + INT8 NNCF
│   ├── test_model.py             # Model validation
│   └── requirements.txt
├── .github/workflows/
│   └── publish.yml               # CI/CD
├── docs/
│   └── model_preparation.md      # Detailed documentation
├── .gitignore
├── LICENSE                       # MIT
├── README.md                     # English
└── README.ru.md                  # Russian
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
| `model.bin` | OpenVINO IR weights (INT8, ~543 MB) |
| `tokenizer.json` | Hugging Face tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `sentencepiece.bpe.model` | SentencePiece model |
| `special_tokens_map.json` | Special token mappings |
| `config.json` | XLM-RoBERTa model config |

## CI/CD

On push to `main`, GitHub Actions automatically:

1. Downloads `BAAI/bge-m3` from Hugging Face
2. Exports to ONNX
3. Converts to OpenVINO IR with INT8 quantization
4. Runs validation tests
5. Publishes the model bundle to Hugging Face

### Required Secrets

Set in **Settings > Secrets and variables > Actions**:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Hugging Face token with write access |
| `HF_REPO` | Target repo (e.g. `your-username/bge-m3-int8-ov`) |

### Manual Trigger

**Actions > Convert and Publish to Hugging Face > Run workflow**.

## References

- [Original model](https://huggingface.co/BAAI/bge-m3)
- [BGE-M3 paper](https://arxiv.org/abs/2402.03216)

## Attribution

Converted version of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) by Beijing Academy of Artificial Intelligence, MIT license.
