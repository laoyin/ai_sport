"""Semi-automatic stage/action labeler for the rep-counting dataset.

Reads the on-device annotation exports (`samples.json`, produced by
`AnnotationExport.kt`) and fills in the two fields that `train.py` needs but
that are exported empty:

  * ``sport_type``   - one value per clip (push_up / squat / background)
  * ``stage_label``  - per-frame up / down / transition / background

The human only supplies **one number per clip**: the true repetition count.
Everything else is derived from the ``pushup_angle`` / ``squat_angle`` signal.

Two independent stage labelings are generated so you can compare them:

  * ``pv``  (peak / valley)   - smooth the angle, find alternating extrema;
                                valley = bottom = ``down``, peak = top = ``up``.
  * ``thr`` (dual-threshold)  - band the angle by the same 155/100 thresholds
                                and hysteresis state machine the app uses on
                                device (mirrors ``ExerciseAnalyzer.kt``), so the
                                labels match production counting behaviour.

Both are written to every frame (``stage_label_pv`` / ``stage_label_thr``); the
one chosen by ``--primary`` also fills the canonical ``stage_label`` that
``dataset.py`` reads. Single-frame pose glitches (e.g. an elbow angle that
collapses to 26 deg for one frame) are removed with a median de-spike before
labeling, so neither method invents a fake rep.

The script never blocks: it batch-labels everything, then prints which clips
need a human look because a detected count disagrees with the supplied count.

Usage
-----
    # 1. discover clips and write a counts template you fill in by hand
    python autolabel.py --input-root D:/datasets/ai_sport/labels --init-counts

    # 2. edit counts.json  ->  {"<video_id>": 9, "rest_clip_01": {"sport": "background"}}

    # 3. label everything into train.json + per-clip review plots
    python autolabel.py --input-root D:/datasets/ai_sport/labels \
        --counts D:/datasets/ai_sport/labels/counts.json \
        --out D:/datasets/ai_sport/train.json \
        --review-dir D:/datasets/ai_sport/review
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Thresholds mirrored from ExerciseAnalyzer.kt / AnnotationExport.kt --------
SPORT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    # sport_type: (up_angle, down_angle)
    "push_up": (155.0, 100.0),
    "squat": (160.0, 105.0),
}
SPORT_SIGNAL_FEATURE: Dict[str, str] = {
    "push_up": "pushup_angle",
    "squat": "squat_angle",
}

STAGE_BACKGROUND = "background"
STAGE_UP = "up"
STAGE_DOWN = "down"
STAGE_TRANSITION = "transition"


# --- Signal cleaning ----------------------------------------------------------
def moving_average_3(x: np.ndarray) -> np.ndarray:
    """3-point moving average, matching ExerciseAnalyzer.smooth()."""
    if len(x) < 3:
        return x.copy()
    padded = np.concatenate([x[:1], x, x[-1:]])
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def median_despike(
    angle: np.ndarray,
    score: np.ndarray,
    win: int = 5,
    spike_deg: float = 30.0,
    min_score: float = 0.2,
) -> np.ndarray:
    """Replace single-frame outliers and low-confidence frames by a local median.

    A frame whose angle deviates from the windowed median by more than
    ``spike_deg`` (a broken pose estimate) or whose pose ``score`` is below
    ``min_score`` is treated as unreliable and overwritten with the median.
    """
    n = len(angle)
    if n == 0:
        return angle.copy()
    half = max(1, win // 2)
    med = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        med[i] = float(np.median(angle[lo:hi]))
    bad = (np.abs(angle - med) > spike_deg) | (score < min_score)
    cleaned = angle.copy().astype(np.float64)
    cleaned[bad] = med[bad]
    return cleaned


def prepare_signal(angle: np.ndarray, score: np.ndarray) -> np.ndarray:
    """De-spike then double 3-point smooth (the app smooths twice too)."""
    cleaned = median_despike(angle, score)
    return moving_average_3(moving_average_3(cleaned))


# --- Method A: peak / valley --------------------------------------------------
def find_extrema(s: np.ndarray, min_amp: float, min_dist: int) -> List[Tuple[int, str]]:
    """Return alternating extrema ``[(idx, 'peak'|'valley'), ...]``.

    Shallow oscillations (peak-to-valley amplitude < ``min_amp``) and extrema
    closer than ``min_dist`` frames are merged away, leaving one clean up/down
    swing per real repetition.
    """
    n = len(s)
    if n < 3:
        return []

    ext: List[List[Any]] = []
    for i in range(1, n - 1):
        if s[i] > s[i - 1] and s[i] >= s[i + 1]:
            ext.append([i, "peak"])
        elif s[i] < s[i - 1] and s[i] <= s[i + 1]:
            ext.append([i, "valley"])

    def collapse_same_type(items: List[List[Any]]) -> List[List[Any]]:
        out: List[List[Any]] = []
        for idx, kind in items:
            if out and out[-1][1] == kind:
                prev = out[-1][0]
                if kind == "peak":
                    out[-1][0] = idx if s[idx] >= s[prev] else prev
                else:
                    out[-1][0] = idx if s[idx] <= s[prev] else prev
            else:
                out.append([idx, kind])
        return out

    ext = collapse_same_type(ext)

    # Iteratively drop the shallowest adjacent pair until all swings clear the bar.
    changed = True
    while changed and len(ext) >= 2:
        changed = False
        amps = [abs(s[ext[k][0]] - s[ext[k + 1][0]]) for k in range(len(ext) - 1)]
        dists = [abs(ext[k + 1][0] - ext[k][0]) for k in range(len(ext) - 1)]
        weakest = int(np.argmin(amps))
        if amps[weakest] < min_amp or dists[weakest] < min_dist:
            del ext[weakest:weakest + 2]
            ext = collapse_same_type(ext)
            changed = True

    return [(int(i), str(k)) for i, k in ext]


def label_peak_valley(n: int, extrema: List[Tuple[int, str]], win: int) -> List[str]:
    labels = [STAGE_TRANSITION] * n
    for idx, kind in extrema:
        stage = STAGE_UP if kind == "peak" else STAGE_DOWN
        for j in range(max(0, idx - win), min(n, idx + win + 1)):
            labels[j] = stage
    return labels


def count_peak_valley(extrema: List[Tuple[int, str]]) -> int:
    return sum(1 for _, kind in extrema if kind == "valley")


def count_from_stage_labels(labels: List[str]) -> int:
    """Count reps from a stage_label array: each contiguous ``down`` run = 1 rep.

    Used to recompute the count after a human edits per-frame labels in the web
    tool (``process_clip`` with ``preserve_manual=True`` keeps the user's
    ``stage_label`` on edited frames).
    """
    reps = 0
    in_down = False
    for label in labels:
        if label == STAGE_DOWN:
            if not in_down:
                reps += 1
                in_down = True
        else:
            in_down = False
    return reps


# --- Method B: dual-threshold hysteresis (mirrors ExerciseAnalyzer.kt) --------
def label_threshold(s: np.ndarray, up_thr: float, down_thr: float) -> List[str]:
    labels: List[str] = []
    for a in s:
        if a <= down_thr:
            labels.append(STAGE_DOWN)
        elif a >= up_thr:
            labels.append(STAGE_UP)
        else:
            labels.append(STAGE_TRANSITION)
    return labels


def count_threshold(
    s: np.ndarray,
    up_thr: float,
    down_thr: float,
    down_hold: int = 1,
    up_hold: int = 1,
    min_gap: int = 2,
) -> int:
    """Hysteresis rep count, matching ExerciseAnalyzer.countByStateMachine()."""
    if len(s) < 5:
        return 0
    if float(s.max() - s.min()) < 18.0:
        return 0

    reps = 0
    stage = "up" if s[0] >= up_thr else "mid"
    down_streak = up_streak = cooldown = 0
    for a in s:
        if cooldown > 0:
            cooldown -= 1
        if a <= down_thr:
            down_streak += 1
            up_streak = 0
        elif a >= up_thr:
            up_streak += 1
            down_streak = 0
        else:
            down_streak = up_streak = 0
        if down_streak >= down_hold and cooldown == 0:
            stage = "down"
        if stage == "down" and up_streak >= up_hold and cooldown == 0:
            reps += 1
            stage = "up"
            cooldown = min_gap
            down_streak = up_streak = 0
    return reps


# --- Per-clip processing ------------------------------------------------------
class ClipResult:
    def __init__(self, video_id: str, records: List[Dict[str, Any]]):
        self.video_id = video_id
        self.records = records
        self.sport = "push_up"
        self.true_count: Optional[int] = None
        self.count_pv = 0
        self.count_thr = 0
        self.count_manual = 0
        self.raw_angle = np.zeros(0)
        self.smooth_angle = np.zeros(0)
        self.extrema: List[Tuple[int, str]] = []
        self.up_thr = 155.0
        self.down_thr = 100.0
        self.manual_edited = False

    @property
    def mismatch(self) -> bool:
        if self.sport == STAGE_BACKGROUND:
            return False
        if self.true_count is None:
            return True  # unverified -> always ask for a look
        return self.true_count not in (self.count_pv, self.count_thr, self.count_manual)


def process_clip(
    video_id: str,
    records: List[Dict[str, Any]],
    sport: str,
    true_count: Optional[int],
    pv_win: int,
    pv_min_amp: float,
    pv_min_dist: int,
    primary: str,
    preserve_manual: bool = False,
) -> ClipResult:
    result = ClipResult(video_id, records)
    result.sport = sport
    result.true_count = true_count
    n = len(records)

    if not preserve_manual:
        for rec in records:
            rec.pop("manual_edited", None)

    if sport == STAGE_BACKGROUND:
        for rec in records:
            rec["sport_type"] = STAGE_BACKGROUND
            if not (preserve_manual and rec.get("manual_edited")):
                rec["stage_label"] = STAGE_BACKGROUND
            rec["stage_label_pv"] = STAGE_BACKGROUND
            rec["stage_label_thr"] = STAGE_BACKGROUND
        result.manual_edited = any(rec.get("manual_edited") for rec in records)
        result.count_manual = count_from_stage_labels([rec.get("stage_label", "") for rec in records])
        return result

    up_thr, down_thr = SPORT_THRESHOLDS.get(sport, SPORT_THRESHOLDS["push_up"])
    feature = SPORT_SIGNAL_FEATURE.get(sport, "pushup_angle")
    result.up_thr, result.down_thr = up_thr, down_thr

    angle = np.array(
        [float((rec.get("features") or {}).get(feature, 180.0)) for rec in records],
        dtype=np.float64,
    )
    score = np.array([float(rec.get("score", 0.0)) for rec in records], dtype=np.float64)
    smooth = prepare_signal(angle, score)
    result.raw_angle = angle
    result.smooth_angle = smooth

    extrema = find_extrema(smooth, pv_min_amp, pv_min_dist)
    result.extrema = extrema
    labels_pv = label_peak_valley(n, extrema, pv_win)
    labels_thr = label_threshold(smooth, up_thr, down_thr)
    result.count_pv = count_peak_valley(extrema)
    result.count_thr = count_threshold(smooth, up_thr, down_thr)

    primary_labels = labels_pv if primary == "pv" else labels_thr
    for i, rec in enumerate(records):
        rec["sport_type"] = sport
        if not (preserve_manual and rec.get("manual_edited")):
            rec["stage_label"] = primary_labels[i]
        rec["stage_label_pv"] = labels_pv[i]
        rec["stage_label_thr"] = labels_thr[i]
    result.manual_edited = any(rec.get("manual_edited") for rec in records)
    result.count_manual = count_from_stage_labels([rec.get("stage_label", "") for rec in records])
    return result


# --- Review plot (optional; degrades gracefully without matplotlib) -----------
def plot_review(result: ClipResult, out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(result.raw_angle))
    ax.plot(x, result.raw_angle, color="#c9d1d9", lw=1, label="raw angle")
    ax.plot(x, result.smooth_angle, color="#1f6feb", lw=2, label="cleaned+smoothed")
    ax.axhline(result.up_thr, color="#2da44e", ls="--", lw=1, label=f"up {result.up_thr:.0f}")
    ax.axhline(result.down_thr, color="#cf222e", ls="--", lw=1, label=f"down {result.down_thr:.0f}")
    for idx, kind in result.extrema:
        if kind == "valley":
            ax.plot(idx, result.smooth_angle[idx], "v", color="#cf222e", ms=9)
        else:
            ax.plot(idx, result.smooth_angle[idx], "^", color="#2da44e", ms=9)
    true_txt = "?" if result.true_count is None else str(result.true_count)
    flag = "  <-- CHECK" if result.mismatch else ""
    ax.set_title(
        f"{result.video_id}   true={true_txt}  pv={result.count_pv}  thr={result.count_thr}{flag}"
    )
    ax.set_xlabel("frame")
    ax.set_ylabel("angle (deg)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


# --- IO / discovery -----------------------------------------------------------
def discover_samples(input_root: Path) -> List[Tuple[str, Path]]:
    if input_root.is_file():
        return [(input_root.parent.name or input_root.stem, input_root)]
    found: List[Tuple[str, Path]] = []
    for samples_path in sorted(input_root.rglob("samples.json")):
        found.append((samples_path.parent.name, samples_path))
    return found


def load_samples(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("frames", data.get("samples", []))
    return list(data)


def read_device_count(samples_path: Path) -> Optional[int]:
    manifest = samples_path.parent / "manifest.json"
    if not manifest.exists():
        return None
    try:
        obj = json.loads(manifest.read_text(encoding="utf-8"))
        return int(obj.get("repetition_count")) if "repetition_count" in obj else None
    except Exception:
        return None


def parse_counts_entry(entry: Any) -> Tuple[str, Optional[int]]:
    """Accept ``9``  or  ``{"count": 9, "sport": "push_up"}``  or  ``{"sport": "background"}``."""
    if isinstance(entry, dict):
        sport = str(entry.get("sport", "push_up"))
        count = entry.get("count")
        return sport, (int(count) if count is not None else None)
    if entry is None:
        return "push_up", None
    return "push_up", int(entry)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", required=True, help="Folder with per-clip samples.json (or a single samples.json)")
    ap.add_argument("--counts", help="counts.json mapping video_id -> true rep count (or {sport,count})")
    ap.add_argument("--out", default="train.json", help="Output training json")
    ap.add_argument("--review-dir", help="Where to write per-clip review plots (default: <out_dir>/review)")
    ap.add_argument("--primary", choices=["pv", "thr"], default="pv", help="Which method fills stage_label")
    ap.add_argument("--sport", default="push_up", help="Default sport_type when a clip is not in counts.json")
    ap.add_argument("--pv-win", type=int, default=1, help="Frames on each side of an extremum labeled up/down")
    ap.add_argument("--pv-min-amp", type=float, default=25.0, help="Min peak-to-valley amplitude (deg) for a rep")
    ap.add_argument("--pv-min-dist", type=int, default=3, help="Min frames between accepted extrema")
    ap.add_argument(
        "--preserve-manual", dest="preserve_manual", action="store_true", default=True,
        help="Don't overwrite stage_label on frames marked manual_edited (default).",
    )
    ap.add_argument(
        "--no-preserve-manual", dest="preserve_manual", action="store_false",
        help="Overwrite stage_label on all frames, ignoring manual_edited.",
    )
    ap.add_argument("--init-counts", action="store_true", help="Only write a counts template (device counts prefilled) and exit")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    clips = discover_samples(input_root)
    if not clips:
        raise SystemExit(f"No samples.json found under {input_root}")

    # --init-counts: emit an editable template and stop.
    if args.init_counts:
        template = {vid: (read_device_count(p) if read_device_count(p) is not None else 0) for vid, p in clips}
        dest = (input_root if input_root.is_dir() else input_root.parent) / "counts.json"
        dest.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote counts template with {len(template)} clips -> {dest}")
        print("Edit the numbers to the true rep counts, then re-run without --init-counts.")
        return

    counts: Dict[str, Any] = {}
    if args.counts:
        counts = json.loads(Path(args.counts).read_text(encoding="utf-8"))

    out_path = Path(args.out)
    review_dir = Path(args.review_dir) if args.review_dir else out_path.parent / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    results: List[ClipResult] = []
    plotted = False

    for video_id, samples_path in clips:
        records = load_samples(samples_path)
        if not records:
            print(f"[skip] {video_id}: empty samples.json")
            continue
        if video_id in counts:
            sport, true_count = parse_counts_entry(counts[video_id])
        else:
            sport, true_count = args.sport, read_device_count(samples_path)

        result = process_clip(
            video_id, records, sport, true_count,
            args.pv_win, args.pv_min_amp, args.pv_min_dist, args.primary,
            args.preserve_manual,
        )
        results.append(result)
        all_records.extend(records)
        plotted = plot_review(result, review_dir / f"{video_id}.png") or plotted

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_records, ensure_ascii=False), encoding="utf-8")

    # Summary table + CSV.
    print(f"\nLabeled {len(all_records)} frames from {len(results)} clips -> {out_path}")
    if plotted:
        print(f"Review plots -> {review_dir}")
    else:
        print("matplotlib not available: skipped plots (pip install matplotlib to enable).")

    csv_lines = ["video_id,sport,true,pv,thr,needs_check"]
    checks: List[ClipResult] = []
    print("\n  clip                                        true  pv  thr")
    print("  " + "-" * 62)
    for r in results:
        t = "?" if r.true_count is None else str(r.true_count)
        mark = "  <-- CHECK" if r.mismatch else ""
        if r.mismatch:
            checks.append(r)
        print(f"  {r.video_id[:42]:42} {t:>4} {r.count_pv:>3} {r.count_thr:>3}{mark}")
        csv_lines.append(f"{r.video_id},{r.sport},{t},{r.count_pv},{r.count_thr},{int(r.mismatch)}")

    (review_dir / "summary.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    if checks:
        print(f"\n{len(checks)} clip(s) need a human look (open their plot in {review_dir}):")
        for r in checks:
            reason = "no true count" if r.true_count is None else f"true={r.true_count} but pv={r.count_pv}/thr={r.count_thr}"
            print(f"  - {r.video_id}: {reason}")
    else:
        print("\nAll clips agree with their true counts. No manual review needed.")


if __name__ == "__main__":
    main()
