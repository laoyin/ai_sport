from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset


SELECTED_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
HANDCRAFTED_FEATURES = [
    "pushup_angle",
    "squat_angle",
    "pushup_depth",
    "torso_linearity",
    "torso_horizontal",
    "torso_vertical",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "hip_to_shoulder_y_diff",
    "hip_to_wrist_y_diff",
]
DELTA_FEATURES = [
    "pushup_angle",
    "pushup_depth",
    "torso_horizontal",
    "hip_to_shoulder_y_diff",
    "left_elbow_angle",
    "right_elbow_angle",
]

ACTION_TO_ID = {
    "background": 0,
    "push_up": 1,
    "sit_up": 2,
}

STAGE_TO_ID = {
    "background": 0,
    "up": 1,
    "down": 2,
    "transition": 3,
}


@dataclass
class FrameRecord:
    video_id: str
    frame_idx: int
    time_ms: int
    sport_type: str
    stage_label: str
    score: float
    keypoints: List[List[float]]
    features: Dict[str, float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def iter_sample_files(path: str | Path) -> List[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Data path not found: {root}")
    sample_files = sorted(root.glob("*/samples.json"))
    if not sample_files:
        raise FileNotFoundError(f"No samples.json found under: {root}")
    return sample_files


def load_records(path: str | Path) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for sample_file in iter_sample_files(path):
        data = json.loads(sample_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("frames", [])

        for item in data:
            sport_type = str(item.get("sport_type", "")).strip()
            stage_label = str(item.get("stage_label", "")).strip()
            if not sport_type or not stage_label:
                continue
            records.append(
                FrameRecord(
                    video_id=str(item.get("video_id", sample_file.parent.name or "unknown")),
                    frame_idx=int(item.get("frame_idx", 0)),
                    time_ms=int(item.get("time_ms", 0)),
                    sport_type=sport_type,
                    stage_label=stage_label,
                    score=_safe_float(item.get("score", 0.0)),
                    keypoints=item.get("keypoints", []),
                    features=item.get("features", {}),
                )
            )
    return records


def group_by_video(records: Sequence[FrameRecord]) -> Dict[str, List[FrameRecord]]:
    grouped: Dict[str, List[FrameRecord]] = {}
    for record in records:
        grouped.setdefault(record.video_id, []).append(record)
    for frames in grouped.values():
        frames.sort(key=lambda r: r.frame_idx)
    return grouped


def _avg_point(points: Sequence[List[float]], idx_a: int, idx_b: int) -> tuple[float, float]:
    pa = points[idx_a] if idx_a < len(points) else [0.0, 0.0, 0.0]
    pb = points[idx_b] if idx_b < len(points) else [0.0, 0.0, 0.0]
    return (float(pa[0]) + float(pb[0])) / 2.0, (float(pa[1]) + float(pb[1])) / 2.0


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def build_frame_feature(
    record: FrameRecord,
    prev_record: FrameRecord | None,
) -> List[float]:
    keypoints = record.keypoints
    shoulder_x, shoulder_y = _avg_point(keypoints, 5, 6)
    hip_x, hip_y = _avg_point(keypoints, 11, 12)
    torso_scale = max(_distance(shoulder_x, shoulder_y, hip_x, hip_y), 1e-4)

    feature: List[float] = []

    for idx in SELECTED_KEYPOINTS:
        if idx < len(keypoints):
            x, y, conf = keypoints[idx]
            feature.extend(
                [
                    (float(x) - hip_x) / torso_scale,
                    (float(y) - hip_y) / torso_scale,
                    float(conf),
                ]
            )
        else:
            feature.extend([0.0, 0.0, 0.0])

    feats = record.features or {}
    prev_feats = prev_record.features if prev_record is not None else {}

    for name in HANDCRAFTED_FEATURES:
        feature.append(_safe_float(feats.get(name, 0.0)))

    for name in DELTA_FEATURES:
        curr = _safe_float(feats.get(name, 0.0))
        prev = _safe_float(prev_feats.get(name, curr))
        feature.append(curr - prev)

    feature.append(float(record.score))
    return feature


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        seq_len: int = 32,
        stride: int = 4,
    ) -> None:
        self.seq_len = seq_len
        self.samples: List[Dict[str, Any]] = []

        records = load_records(data_path)
        grouped = group_by_video(records)

        for _, frames in grouped.items():
            if len(frames) < seq_len:
                continue
            feature_cache = [
                build_frame_feature(frame, frames[i - 1] if i > 0 else None)
                for i, frame in enumerate(frames)
            ]

            for start in range(0, len(frames) - seq_len + 1, stride):
                end = start + seq_len
                window_frames = frames[start:end]
                window_features = feature_cache[start:end]

                action_name = window_frames[seq_len // 2].sport_type
                action_id = ACTION_TO_ID.get(action_name, 0)
                stage_ids = [STAGE_TO_ID.get(f.stage_label, 0) for f in window_frames]

                self.samples.append(
                    {
                        "x": torch.tensor(window_features, dtype=torch.float32),
                        "action": torch.tensor(action_id, dtype=torch.long),
                        "stage": torch.tensor(stage_ids, dtype=torch.long),
                    }
                )

        if not self.samples:
            raise ValueError(f"No training windows built from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]
