from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from dataset import (
    ACTION_TO_ID,
    STAGE_TO_ID,
    FrameRecord,
    build_frame_feature,
    group_by_video,
    load_records,
)
from model import ConvTinyTransformer


ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}
ID_TO_STAGE = {v: k for k, v in STAGE_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to label root directory")
    parser.add_argument(
        "--ckpt",
        default="ai_sport/training/checkpoints/best_conv_tiny_transformer.pt",
        help="Path to trained checkpoint",
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride during inference")
    parser.add_argument("--counts", default=None, help="Optional counts.json for true count comparison")
    parser.add_argument("--save-json", default=None, help="Optional output json path for per-video predictions")
    return parser.parse_args()


def load_checkpoint(path: str | Path, seq_len: int) -> ConvTinyTransformer:
    checkpoint = torch.load(Path(path), map_location="cpu")
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


def count_from_stage_labels(labels: List[str]) -> int:
    reps = 0
    in_down = False
    for label in labels:
        if label == "down":
            if not in_down:
                reps += 1
                in_down = True
        else:
            in_down = False
    return reps


def infer_video(
    model: ConvTinyTransformer,
    frames: List[FrameRecord],
    seq_len: int,
    stride: int,
) -> Dict[str, object] | None:
    if len(frames) < seq_len:
        return None

    feature_cache = [
        build_frame_feature(frame, frames[i - 1] if i > 0 else None)
        for i, frame in enumerate(frames)
    ]

    stage_votes: List[Counter[str]] = [Counter() for _ in frames]
    action_votes: Counter[str] = Counter()

    with torch.no_grad():
        for start in range(0, len(frames) - seq_len + 1, stride):
            end = start + seq_len
            window_features = feature_cache[start:end]
            x = torch.tensor(window_features, dtype=torch.float32).unsqueeze(0)
            outputs = model(x)

            action_id = int(outputs["action_logits"].argmax(dim=-1).item())
            action_name = ID_TO_ACTION.get(action_id, "background")
            action_votes[action_name] += 1

            stage_ids = outputs["stage_logits"].argmax(dim=-1).squeeze(0).tolist()
            for offset, stage_id in enumerate(stage_ids):
                frame_index = start + offset
                stage_name = ID_TO_STAGE.get(int(stage_id), "background")
                stage_votes[frame_index][stage_name] += 1

    pred_action = action_votes.most_common(1)[0][0] if action_votes else "background"
    pred_stages = [
        votes.most_common(1)[0][0] if votes else "background"
        for votes in stage_votes
    ]
    pred_count = count_from_stage_labels(pred_stages) if pred_action == "push_up" else 0

    true_action_votes = Counter(frame.sport_type for frame in frames if frame.sport_type)
    true_action = true_action_votes.most_common(1)[0][0] if true_action_votes else "unknown"
    true_stages = [frame.stage_label for frame in frames]

    return {
        "video_id": frames[0].video_id,
        "num_frames": len(frames),
        "pred_action": pred_action,
        "pred_count": pred_count,
        "pred_stage_counts": dict(Counter(pred_stages)),
        "true_action": true_action,
        "true_count_from_labels": count_from_stage_labels(true_stages) if true_action == "push_up" else 0,
    }


def load_counts(path: str | Path | None) -> Dict[str, object]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_true_count_entry(entry: object) -> tuple[str | None, int | None]:
    if isinstance(entry, dict):
        sport = entry.get("sport")
        count = entry.get("count")
        return (str(sport) if sport is not None else None, int(count) if count is not None else None)
    if isinstance(entry, int):
        return None, int(entry)
    return None, None


def main() -> None:
    args = parse_args()
    model = load_checkpoint(args.ckpt, args.seq_len)
    records = load_records(args.data)
    grouped = group_by_video(records)
    counts_map = load_counts(args.counts)

    results: List[Dict[str, object]] = []
    for video_id, frames in grouped.items():
        result = infer_video(model, frames, args.seq_len, args.stride)
        if result is None:
            continue
        true_sport, true_count = normalize_true_count_entry(counts_map.get(video_id))
        if true_sport is not None:
            result["true_action_from_counts"] = true_sport
        if true_count is not None:
            result["true_count_from_counts"] = true_count
            result["count_error"] = result["pred_count"] - true_count
        results.append(result)

    total = len(results)
    action_hits = 0
    count_errors: List[int] = []
    for item in results:
        true_action = str(item.get("true_action_from_counts") or item.get("true_action") or "unknown")
        if item["pred_action"] == true_action:
            action_hits += 1
        if "count_error" in item:
            count_errors.append(abs(int(item["count_error"])))

    print(f"checkpoint={args.ckpt}")
    print(f"videos={total}")
    print(f"video_action_acc={action_hits / max(1, total):.4f}")
    if count_errors:
        mae = sum(count_errors) / len(count_errors)
        exact = sum(1 for e in count_errors if e == 0) / len(count_errors)
        print(f"count_mae={mae:.4f}")
        print(f"count_exact_match={exact:.4f}")

    for item in results:
        line = (
            f"{item['video_id']} | pred_action={item['pred_action']} | pred_count={item['pred_count']}"
        )
        if "true_count_from_counts" in item:
            line += (
                f" | true_count={item['true_count_from_counts']} | err={item['count_error']:+d}"
            )
        print(line)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
