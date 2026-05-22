# BGE-M3 OpenVINO Model Preparation — Design Spec

**Date**: 2026-05-22
**Status**: approved

## Overview

Full pipeline for converting BAAI/bge-m3 (MIT license) from Hugging Face PyTorch to OpenVINO IR with INT8 quantization, optimized for NPU inference. Includes CI/CD for automated publishing to Hugging Face on every push to main.

## Reference

Based on the existing pipeline from `/mnt/c/Users/strokin/projects/ai2npu_c++/`, specifically:
- `scripts/convert_bge_m3_openvino.py` — OpenVINO IR conversion + NNCF quantization
- `scripts/check_bge_m3_npu.py` — model smoke test
- `docs/model_artifacts.md` — documentation

Key gap in ai2npu: ONNX export step is missing. This project fills that gap.

## Target Model Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 1 | NPU compatibility |
| Max sequence length | 512 | Matches ai2npu runtime; faster inference |
| Input shapes | `input_ids` [1,512] int64, `attention_mask` [1,512] int64 | Static shapes required for NPU |
| Output | `sentence_embedding` [1,1024] float32 | Dense embedding vector |
| Quantization | INT8 asymmetric via NNCF | Matches ai2npu; good accuracy/perf tradeoff |
| Weight storage | FP32 IR with INT8 compression | `compress_to_fp16=False` |

## Architecture

### Directory Structure

```
bge-m3-openvino/
├── scripts/
│   ├── export_onnx.py            # Step 1: HF PyTorch → ONNX
│   ├── convert_openvino.py       # Step 2: ONNX → OpenVINO IR + INT8
│   ├── test_model.py             # Step 3: validation
│   └── requirements.txt
├── .github/workflows/
│   └── publish.yml               # CI/CD pipeline
├── docs/
│   └── model_preparation.md      # User-facing docs
├── .gitignore
├── LICENSE                       # MIT
└── README.md
```

### Script 1: `export_onnx.py`

**Input**: `BAAI/bge-m3` from Hugging Face (auto-downloaded by transformers)

**Output**: model in ONNX format with static shapes [1, 512]

Steps:
1. Load `AutoModel` from `BAAI/bge-m3` with `trust_remote_code=True`
2. Export to ONNX via `torch.onnx.export` with dummy inputs shaped [1, 512]
3. Input names: `input_ids`, `attention_mask`
4. Output names: `token_embeddings`, `sentence_embedding`
5. Save `model.onnx` to `--output` directory

Key considerations:
- BGE-M3 uses XLM-RoBERTa backbone, outputs pooled last_hidden_state
- Need to trace the model's forward pass correctly for both outputs
- Use opset version compatible with OpenVINO (15+ recommended)

### Script 2: `convert_openvino.py`

**Input**: `model.onnx` from step 1

**Output**: OpenVINO IR bundle:
- `model.xml` + `model.bin` — INT8 compressed embedding model
- `openvino_tokenizer.xml` + `openvino_tokenizer.bin` — tokenizer IR
- HF tokenizer files: `tokenizer.json`, `tokenizer_config.json`, `sentencepiece.bpe.model`, `special_tokens_map.json`, `config.json`

Steps:
1. `ov.convert_model(onnx_path, input=[("input_ids", [1,512]), ("attention_mask", [1,512])])`
2. `nncf.compress_weights(model, mode=INT8_ASYM)`
3. `ov.save_model(compressed_model, "model.xml", compress_to_fp16=False)`
4. Convert tokenizer via `openvino_tokenizers` to IR
5. Copy HF tokenizer files from source

**Arguments**:
| Arg | Default | Description |
|-----|---------|-------------|
| `--source` | `models/huggingface/BAAI/bge-m3` | Source with ONNX + HF tokenizer |
| `--output` | `models/bge-m3-openvino-int8` | Output directory |
| `--batch-size` | 1 | Static batch dimension |
| `--max-length` | 512 | Static sequence length |
| `--mode` | `int8_asym` | NNCF compression mode |

### Script 3: `test_model.py`

**Input**: OpenVINO IR bundle directory

**Validation checks**:
1. Load model with `ov.Core().read_model()` — no errors
2. Compile for CPU (CI environment, no NPU) — no errors
3. Tokenize a test sentence, run inference
4. Verify output shape: `sentence_embedding` is [1, 1024]
5. Verify output dtype: float32
6. Verify L2 norm of embedding ≈ 1.0 (model outputs normalized embeddings)
7. Verify cosine similarity between two related sentences > 0.5

**Arguments**:
| Arg | Default | Description |
|-----|---------|-------------|
| `--model-dir` | `models/bge-m3-openvino-int8` | Model bundle directory |
| `--device` | `CPU` | OpenVINO device |

### CI/CD Pipeline: `publish.yml`

**Trigger**: `push` to `main` branch

**Jobs** (sequential):

| # | Step | Details |
|---|------|---------|
| 1 | Checkout | Clone repo |
| 2 | Setup Python | Python 3.10, cache pip |
| 3 | Install deps | `pip install -r scripts/requirements.txt` |
| 4 | Download model | `transformers` auto-fetches `BAAI/bge-m3`, cache for subsequent runs |
| 5 | Export ONNX | `python scripts/export_onnx.py --output models/onnx` |
| 6 | Convert to OpenVINO | `python scripts/convert_openvino.py --source models/onnx --output models/bge-m3-openvino-int8` |
| 7 | Test model | `python scripts/test_model.py --model-dir models/bge-m3-openvino-int8` |
| 8 | Upload to Hugging Face | `huggingface_hub.upload_folder` to `${{ secrets.HF_REPO }}` with token `${{ secrets.HF_TOKEN }}` |

**Hugging Face repo**: configured via GitHub Secrets:
- `HF_REPO` — target repo (e.g., `username/bge-m3-openvino`)
- `HF_TOKEN` — write-access token

## Dependencies

```
openvino>=2025.0
nncf>=2.14
transformers>=4.40
tokenizers>=0.19
torch>=2.2
onnx>=1.16
optimum-intel>=1.20
huggingface_hub>=0.23
```

## Licensing

- Original BAAI/bge-m3: MIT (Beijing Academy of Artificial Intelligence)
- This project scripts: MIT
- Converted model bundle: MIT (derivative work, same license as original)
- Include original BAAI copyright notice in LICENSE file

## Self-Review Checklist

- [x] No placeholders or TBDs — all sections filled
- [x] Internal consistency — parameters match across scripts and CI
- [x] Scope — one project, one model, one pipeline; no feature creep
- [x] Ambiguity — input/output names, shapes, dtypes explicitly specified
- [x] Dependencies — all libraries listed with minimum versions
- [x] License — MIT confirmed for both source model and this project
