from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import PoseSequenceDataset
from model import ConvTinyTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to one samples.json or a label root directory")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="ai_sport/training/checkpoints")
    return parser.parse_args()


def compute_loss(
    outputs: dict[str, torch.Tensor],
    action_target: torch.Tensor,
    stage_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    action_loss = F.cross_entropy(outputs["action_logits"], action_target)
    stage_loss = F.cross_entropy(
        outputs["stage_logits"].reshape(-1, outputs["stage_logits"].size(-1)),
        stage_target.reshape(-1),
    )
    loss = 0.4 * action_loss + 0.6 * stage_loss
    return loss, {
        "action_loss": float(action_loss.detach().cpu()),
        "stage_loss": float(stage_loss.detach().cpu()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    dataset = PoseSequenceDataset(
        data_path=args.data,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = ConvTinyTransformer(
        input_dim=55,
        seq_len=args.seq_len,
        d_model=96,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.1,
        num_action_classes=3,
        num_stage_classes=4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            action = batch["action"].to(device)
            stage = batch["stage"].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss, _ = compute_loss(outputs, action, stage)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        correct_action = 0
        total_action = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                action = batch["action"].to(device)
                stage = batch["stage"].to(device)
                outputs = model(x)
                loss, _ = compute_loss(outputs, action, stage)
                val_loss += float(loss.detach().cpu())

                pred_action = outputs["action_logits"].argmax(dim=-1)
                correct_action += int((pred_action == action).sum().cpu())
                total_action += int(action.numel())

        val_loss /= max(1, len(val_loader))
        action_acc = correct_action / max(1, total_action)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_action_acc={action_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = save_dir / "best_conv_tiny_transformer.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seq_len": args.seq_len,
                    "input_dim": 55,
                },
                ckpt_path,
            )
            print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
