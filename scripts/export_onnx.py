import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer


EXTRA_FILES = [
    "config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
]


class BgeM3OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        sentence_embedding = token_embeddings[:, 0, :]
        return token_embeddings, sentence_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BGE-M3 to ONNX with static shapes.")
    parser.add_argument("--model-id", default="BAAI/bge-m3")
    parser.add_argument("--output", type=Path, default=Path("models/onnx"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()

    wrapper = BgeM3OnnxWrapper(model)
    wrapper.eval()

    dummy_input_ids = torch.zeros((args.batch_size, args.max_length), dtype=torch.long)
    dummy_attention_mask = torch.zeros((args.batch_size, args.max_length), dtype=torch.long)

    onnx_path = args.output / "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["token_embeddings", "sentence_embedding"],
        dynamic_axes=None,
        do_constant_folding=True,
    )

    tokenizer.save_pretrained(args.output)
    model.config.save_pretrained(args.output)

    for file_name in EXTRA_FILES:
        destination = args.output / file_name
        if destination.exists():
            continue
        try:
            source = hf_hub_download(repo_id=args.model_id, filename=file_name)
            shutil.copy2(source, destination)
        except Exception:
            pass

    print(f"onnx_model={onnx_path}")
    print(f"input_shape=[{args.batch_size}, {args.max_length}]")


if __name__ == "__main__":
    main()
