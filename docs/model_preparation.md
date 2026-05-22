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
