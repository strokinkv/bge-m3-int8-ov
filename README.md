---
language: en
license: mit
library_name: openvino
tags:
- openvino
- int8
- embedding
- bge-m3
- sentence-similarity
- feature-extraction
- nncf
pipeline_tag: feature-extraction
base_model: BAAI/bge-m3
---

# BGE-M3 OpenVINO

[Русская версия](README.ru.md)

[BGE-M3](https://huggingface.co/BAAI/bge-m3) embedding model converted to OpenVINO IR with INT8 quantization. Optimized for Intel NPU/CPU/GPU inference.

## Quick Start

### Install

```bash
# For conversion:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts/requirements.txt

# For inference only (pre-converted model):
pip install openvino transformers
```

### Download and convert

```bash
optimum-cli export openvino --model BAAI/bge-m3 --task feature-extraction --weight-format int8 models/bge-m3-int8-ov
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
model = core.compile_model("models/bge-m3-int8-ov/openvino_model.xml", "CPU")
tokenizer = AutoTokenizer.from_pretrained("models/bge-m3-int8-ov")

text = "What is BGE M3?"
encoded = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
result = model({"input_ids": encoded["input_ids"].astype(np.int64), "attention_mask": encoded["attention_mask"].astype(np.int64)})
last_hidden = result["last_hidden_state"]
embedding = last_hidden[0, 0, :]  # CLS token = sentence embedding, shape: [1024]
```

## Project Structure

```
bge-m3-openvino/
├── scripts/
│   ├── test_model.py             # Model validation
│   └── requirements.txt
├── .github/workflows/
│   └── publish.yml               # CI/CD
├── .gitignore
├── LICENSE                       # MIT
├── README.md                     # English
└── README.ru.md                  # Russian
```

## Model Details

| Property | Value |
|----------|-------|
| Target model | [bge-m3-int8-ov](https://huggingface.co/strokinkv/bge-m3-int8-ov) |
| Base model | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| License | MIT |
| Input shape | dynamic |
| Input names | `input_ids` (int64), `attention_mask` (int64) |
| Output | `last_hidden_state` [batch, seq, 1024] float32 |
| Sentence embedding | CLS token: `last_hidden_state[:, 0, :]` → [1024] |
| Quantization | INT8 asymmetric (NNCF) |
| Embedding dim | 1024 |
| Max length | 8192 tokens (original model limit) |

## Model Signature

| Direction | Name | Shape | Type |
|-----------|------|-------|------|
| Input | `input_ids` | [batch, seq] | int64 |
| Input | `attention_mask` | [batch, seq] | int64 |
| Output | `last_hidden_state` | [batch, seq, 1024] | float32 |

## Bundle Files

| File | Description |
|------|-------------|
| `openvino_model.xml` | OpenVINO IR graph |
| `openvino_model.bin` | OpenVINO IR weights (INT8, ~543 MB) |
| `tokenizer.json` | Hugging Face tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `sentencepiece.bpe.model` | SentencePiece model |
| `special_tokens_map.json` | Special token mappings |
| `config.json` | XLM-RoBERTa model config |

## References

- [strokinkv/bge-m3-int8-ov](https://huggingface.co/strokinkv/bge-m3-int8-ov)
- [Original model](https://huggingface.co/BAAI/bge-m3)
- [BGE-M3 paper](https://arxiv.org/abs/2402.03216)
- [ai2npu — NPU inference](https://github.com/strokinkv/ai2npu)

## License

MIT. Original model by [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), Beijing Academy of Artificial Intelligence.
