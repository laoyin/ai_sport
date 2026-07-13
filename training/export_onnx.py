from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import ConvTinyTransformer


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: ConvTinyTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        return outputs["stage_logits"], outputs["action_logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="ai_sport/training/checkpoints/best_conv_tiny_transformer.pt",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--output",
        default="ai_sport/rep_counter.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=55)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(Path(args.ckpt), map_location="cpu")

    model = ConvTinyTransformer(
        input_dim=int(ckpt.get("input_dim", args.input_dim)),
        seq_len=int(ckpt.get("seq_len", args.seq_len)),
        d_model=96,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_action_classes=3,
        num_stage_classes=4,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapper = OnnxExportWrapper(model).eval()
    seq_len = int(ckpt.get("seq_len", args.seq_len))
    input_dim = int(ckpt.get("input_dim", args.input_dim))
    dummy = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["stage_logits", "action_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"exported_onnx={output_path}")
    print(f"input_shape=[1,{seq_len},{input_dim}]")


if __name__ == "__main__":
    main()
