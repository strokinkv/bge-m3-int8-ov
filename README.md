# BGE-M3 OpenVINO

[Русская версия](README.ru.md)

[BGE-M3](https://huggingface.co/BAAI/bge-m3) embedding model converted to OpenVINO IR with INT8 quantization.

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

## CI/CD Pipeline

This repository uses GitHub Actions to automatically convert and publish the model on every push to `main`.

### Required Secrets

Set these in your GitHub repository **Settings > Secrets and variables > Actions**:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Hugging Face API token with write access |
| `HF_REPO` | Target Hugging Face repository (e.g., `your-username/bge-m3-openvino`) |

### Manual Trigger

Go to **Actions > Convert and Publish to Hugging Face > Run workflow**.

## Attribution

This is a converted version of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) by Beijing Academy of Artificial Intelligence, licensed under MIT.
