# Changelog

## Unreleased

### Changed

- Restored the static PyTorch → ONNX → OpenVINO conversion pipeline.
- Fixed model inputs to `[1, 512]` for Intel NPU compatibility.
- Restored the `sentence_embedding [1, 1024]` output expected by ai2npu.
- Strengthened validation to reject dynamic input shapes and missing sentence embeddings.

### Removed

- Removed the direct `optimum-cli export openvino` pipeline because it produced unbounded dynamic inputs that Intel NPU could not compile.
- Configured publication to remove obsolete `openvino_model.xml` and `openvino_model.bin` files from the Hugging Face repository.
