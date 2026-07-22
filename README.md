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
```

## NPU compatibility

OpenVINO IR supports dynamic input shapes, but Intel NPU compilation currently requires static input shapes. The direct `optimum-cli export openvino` pipeline produced unbounded `[-1, -1]` inputs: the model worked on CPU but NPU compilation failed with `ov_core_compile_model failed with status -1`.

This project therefore exports BGE-M3 through a static ONNX graph and converts it to an INT8 OpenVINO IR with fixed `[1, 512]` inputs. The graph also exposes the CLS token directly as `sentence_embedding [1, 1024]`, so ai2npu does not need to reshape the model or post-process `last_hidden_state`.

## Project Structure

```
bge-m3-openvino/
├── scripts/
│   ├── export_onnx.py            # PyTorch → static ONNX
│   ├── convert_openvino.py       # ONNX → OpenVINO IR + INT8 NNCF
│   ├── test_model.py             # Model validation
│   └── requirements.txt
├── .github/workflows/
│   └── publish.yml               # CI/CD
├── .gitignore
├── CHANGELOG.md
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
| `model.xml` | Static OpenVINO IR graph |
| `model.bin` | OpenVINO IR weights (INT8, ~543 MB) |
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
