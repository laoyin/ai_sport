"""Flask web labeler for the rep-counting dataset.

Lets a human reviewer watch sampled frames as a video, enter the true
repetition count, and reuse ``autolabel.process_clip`` to fill in the
per-frame ``sport_type`` + ``stage_label`` fields without leaving the
browser. The user's count can optionally be written back to
``counts.json``.

Run::

    D:/open-project/ali_ai_match/.venv/Scripts/python.exe app.py
    # open http://127.0.0.1:5000/
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

HERE = Path(__file__).resolve().parent
TRAINING_DIR = HERE.parent
sys.path.insert(0, str(TRAINING_DIR))

from autolabel import (  # noqa: E402  (import after sys.path tweak)
    STAGE_BACKGROUND,
    ClipResult,
    count_from_stage_labels,
    discover_samples,
    load_samples,
    parse_counts_entry,
    process_clip,
    read_device_count,
)

LABELS_ROOT = Path("D:/open-project/ali_ai_match/ai_sport/ai_sport_labels")
COUNTS_PATH = LABELS_ROOT / "counts.json"

VALID_SPORTS = {"push_up", "squat", STAGE_BACKGROUND}
VALID_PRIMARY = {"pv", "thr"}
VALID_STAGES = {"up", "down", "transition", STAGE_BACKGROUND}

DEFAULT_PV_WIN = 1
DEFAULT_PV_MIN_AMP = 25.0
DEFAULT_PV_MIN_DIST = 3
DEFAULT_PRIMARY = "pv"

app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))


# --- helpers -----------------------------------------------------------------
def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_counts() -> Dict[str, Any]:
    if not COUNTS_PATH.exists():
        return {}
    try:
        return json.loads(COUNTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _clip_dir(video_id: str) -> Optional[Path]:
    """Return the clip folder if it exists and is under LABELS_ROOT."""
    if not video_id or "/" in video_id or "\\" in video_id or ".." in video_id:
        return None
    candidate = (LABELS_ROOT / video_id).resolve()
    try:
        candidate.relative_to(LABELS_ROOT.resolve())
    except ValueError:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def _samples_path(video_id: str) -> Optional[Path]:
    clip = _clip_dir(video_id)
    if clip is None:
        return None
    samples = clip / "samples.json"
    return samples if samples.exists() else None


def _manifest(video_id: str) -> Dict[str, Any]:
    clip = _clip_dir(video_id)
    if clip is None:
        return {}
    manifest_path = clip / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _project_frame(record: Dict[str, Any], video_id: str) -> Dict[str, Any]:
    features = record.get("features") or {}
    return {
        "frame_idx": int(record.get("frame_idx", 0)),
        "time_ms": int(record.get("time_ms", 0)),
        "frame_url": f"/labels/{video_id}/{record.get('frame_image', '')}",
        "pose_url": f"/labels/{video_id}/{record.get('pose_image', '')}",
        "score": float(record.get("score", 0.0)),
        "quality_hint": record.get("quality_hint", ""),
        "suggested_sport_type": record.get("suggested_sport_type", ""),
        "suggested_stage_label": record.get("suggested_stage_label", ""),
        "pushup_angle": float(features.get("pushup_angle", 180.0)),
        "squat_angle": float(features.get("squat_angle", 180.0)),
        "stage_label": record.get("stage_label", ""),
        "manual_edited": bool(record.get("manual_edited", False)),
    }


# --- routes ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/clips")
def api_clips():
    clips: List[Dict[str, Any]] = []
    counts = _load_counts()
    for video_id, samples_path in discover_samples(LABELS_ROOT):
        manifest = _manifest(video_id)
        device_count = read_device_count(samples_path)
        sport, true_count = parse_counts_entry(counts.get(video_id))
        # Cheap manual_edited check: peek at first record only (save-labels sets the flag on all).
        manual_edited = False
        try:
            records_peek = load_samples(samples_path)
            if records_peek:
                manual_edited = bool(records_peek[0].get("manual_edited"))
        except Exception:
            manual_edited = False
        clips.append({
            "video_id": video_id,
            "frame_count": int(manifest.get("frame_count", 0) or 0),
            "device_count": device_count,
            "inferred_sport_type": manifest.get("inferred_sport_type", ""),
            "confidence": float(manifest.get("confidence", 0.0) or 0.0),
            "exported_at": manifest.get("exported_at", ""),
            "true_count": true_count,
            "sport": sport,
            "labeled": video_id in counts,
            "manual_edited": manual_edited,
        })
    clips.sort(key=lambda c: c["video_id"])
    return jsonify({"clips": clips})


@app.route("/api/clips/<video_id>")
def api_clip_detail(video_id: str):
    samples_path = _samples_path(video_id)
    if samples_path is None:
        return jsonify({"error": "clip not found"}), 404
    records = load_samples(samples_path)
    manifest = _manifest(video_id)
    counts = _load_counts()
    raw_entry = counts.get(video_id)
    sport, true_count = parse_counts_entry(raw_entry)
    frames = [_project_frame(r, video_id) for r in records]
    duration_ms = frames[-1]["time_ms"] if frames else 0
    manual_edited = any(bool(r.get("manual_edited")) for r in records)
    saved_stage_labels = [r.get("stage_label", "") for r in records]
    return jsonify({
        "video_id": video_id,
        "manifest": manifest,
        "device_count": read_device_count(samples_path),
        "counts_entry_raw": raw_entry,
        "true_count": true_count,
        "sport": sport,
        "duration_ms": duration_ms,
        "frame_count": len(frames),
        "frames": frames,
        "manual_edited": manual_edited,
        "saved_stage_labels": saved_stage_labels,
        "saved_count": count_from_stage_labels(saved_stage_labels) if manual_edited else 0,
    })


@app.route("/api/clips/<video_id>/label", methods=["POST"])
def api_clip_label(video_id: str):
    samples_path = _samples_path(video_id)
    if samples_path is None:
        return jsonify({"error": "clip not found"}), 404

    body = request.get_json(silent=True) or {}
    sport = body.get("sport", "push_up")
    if sport not in VALID_SPORTS:
        return jsonify({"error": f"invalid sport: {sport}"}), 400

    true_count_raw = body.get("true_count")
    if sport == STAGE_BACKGROUND:
        true_count: Optional[int] = None
    elif true_count_raw is None:
        true_count = None
    else:
        try:
            true_count = int(true_count_raw)
            if true_count < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "true_count must be a non-negative int"}), 400

    primary = body.get("primary", DEFAULT_PRIMARY)
    if primary not in VALID_PRIMARY:
        return jsonify({"error": f"invalid primary: {primary}"}), 400

    try:
        pv_win = int(body.get("pv_win", DEFAULT_PV_WIN))
        pv_min_amp = float(body.get("pv_min_amp", DEFAULT_PV_MIN_AMP))
        pv_min_dist = int(body.get("pv_min_dist", DEFAULT_PV_MIN_DIST))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid pv_* parameters"}), 400

    force = bool(body.get("force", False))
    preserve_manual = not force

    records = load_samples(samples_path)
    result: ClipResult = process_clip(
        video_id, records, sport, true_count,
        pv_win, pv_min_amp, pv_min_dist, primary,
        preserve_manual=preserve_manual,
    )

    stage_pv = [r.get("stage_label_pv", "") for r in records]
    stage_thr = [r.get("stage_label_thr", "") for r in records]
    stage_primary = [r.get("stage_label", "") for r in records]

    return jsonify({
        "video_id": video_id,
        "sport": result.sport,
        "true_count": result.true_count,
        "count_pv": int(result.count_pv),
        "count_thr": int(result.count_thr),
        "count_manual": int(result.count_manual),
        "mismatch": bool(result.mismatch),
        "up_thr": float(result.up_thr),
        "down_thr": float(result.down_thr),
        "primary_method": primary,
        "manual_edited": bool(result.manual_edited),
        "raw_angle": result.raw_angle.tolist(),
        "smooth_angle": result.smooth_angle.tolist(),
        "extrema": [[int(i), str(k)] for i, k in result.extrema],
        "stage_labels_pv": stage_pv,
        "stage_labels_thr": stage_thr,
        "stage_labels_primary": stage_primary,
    })


@app.route("/api/clips/<video_id>/save", methods=["POST"])
def api_clip_save(video_id: str):
    if _clip_dir(video_id) is None:
        return jsonify({"error": "clip not found"}), 404

    body = request.get_json(silent=True) or {}
    sport = body.get("sport", "push_up")
    if sport not in VALID_SPORTS:
        return jsonify({"error": f"invalid sport: {sport}"}), 400

    if sport == STAGE_BACKGROUND:
        entry: Dict[str, Any] = {"sport": STAGE_BACKGROUND}
    else:
        raw = body.get("true_count")
        try:
            true_count = int(raw)
            if true_count < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "true_count must be a non-negative int"}), 400
        entry = {"count": true_count, "sport": sport}

    counts = _load_counts()
    counts[video_id] = entry
    _atomic_write_json(COUNTS_PATH, counts)
    return jsonify({
        "video_id": video_id,
        "saved": True,
        "entry": entry,
        "total_entries": len(counts),
    })


@app.route("/api/clips/<video_id>/save-labels", methods=["POST"])
def api_clip_save_labels(video_id: str):
    """Persist user-edited per-frame stage labels back to samples.json.

    Writes ``stage_label`` + ``sport_type`` per record and marks every record
    ``manual_edited: true`` so the next ``process_clip`` call (with
    ``preserve_manual=True``, the default) keeps the user's labels.
    """
    samples_path = _samples_path(video_id)
    if samples_path is None:
        return jsonify({"error": "clip not found"}), 404

    body = request.get_json(silent=True) or {}
    stage_labels = body.get("stage_labels")
    if not isinstance(stage_labels, list):
        return jsonify({"error": "stage_labels must be a list"}), 400

    sport = body.get("sport", "push_up")
    if sport not in VALID_SPORTS:
        return jsonify({"error": f"invalid sport: {sport}"}), 400

    records = load_samples(samples_path)
    if len(stage_labels) != len(records):
        return jsonify({
            "error": f"stage_labels length {len(stage_labels)} != frame count {len(records)}",
        }), 400

    for i, label in enumerate(stage_labels):
        if label not in VALID_STAGES:
            return jsonify({"error": f"invalid stage at frame {i}: {label!r}"}), 400

    for rec, label in zip(records, stage_labels):
        rec["stage_label"] = label
        rec["sport_type"] = sport
        rec["manual_edited"] = True

    _atomic_write_json(samples_path, records)
    saved_count = count_from_stage_labels(stage_labels)
    return jsonify({
        "video_id": video_id,
        "saved": True,
        "frame_count": len(records),
        "saved_count": saved_count,
        "sport": sport,
        "manual_edited": True,
    })


@app.route("/labels/<path:rel_path>")
def serve_label_file(rel_path: str):
    return send_from_directory(LABELS_ROOT, rel_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
