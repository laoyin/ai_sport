from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ACTION_TO_ID, STAGE_TO_ID, PoseSequenceDataset
from model import ConvTinyTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to one samples.json or a label root directory")
    parser.add_argument(
        "--ckpt",
        default="ai_sport/training/checkpoints/best_conv_tiny_transformer.pt",
        help="Path to a trained checkpoint",
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def build_model(checkpoint: dict, seq_len: int) -> ConvTinyTransformer:
    model = ConvTinyTransformer(
        input_dim=int(checkpoint.get("input_dim", 55)),
        seq_len=int(checkpoint.get("seq_len", seq_len)),
        d_model=96,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.1,
        num_action_classes=len(ACTION_TO_ID),
        num_stage_classes=len(STAGE_TO_ID),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    dataset = PoseSequenceDataset(
        data_path=args.data,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
    model = build_model(checkpoint, args.seq_len).to(device)

    action_correct = 0
    action_total = 0
    stage_correct = 0
    stage_total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            action = batch["action"].to(device)
            stage = batch["stage"].to(device)

            outputs = model(x)
            pred_action = outputs["action_logits"].argmax(dim=-1)
            pred_stage = outputs["stage_logits"].argmax(dim=-1)

            action_correct += int((pred_action == action).sum().cpu())
            action_total += int(action.numel())
            stage_correct += int((pred_stage == stage).sum().cpu())
            stage_total += int(stage.numel())

    action_acc = action_correct / max(1, action_total)
    stage_acc = stage_correct / max(1, stage_total)

    print(f"checkpoint={args.ckpt}")
    print(f"windows={len(dataset)}")
    print(f"action_acc={action_acc:.4f}")
    print(f"stage_acc={stage_acc:.4f}")


if __name__ == "__main__":
    main()
