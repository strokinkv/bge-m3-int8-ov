import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


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

    tokenizer.save_pretrained(args.output)
    print(f"Tokenizer files saved to {args.output}")


if __name__ == "__main__":
    main()
