# AI Sport Training

This folder contains a CPU-friendly training baseline for repetition counting based on:

- `YOLO26n-pose` keypoints
- handcrafted motion features already used in the Android app
- a lightweight `Conv1D + Tiny Transformer` sequence model

## Goal

Train a model that takes a fixed-length pose feature window and predicts:

- per-frame motion stage labels
- per-window action class labels

The final repetition count should still come from a small state machine over the stage predictions. This keeps the result more stable and easier to debug on device.

## Recommended Sample Format

Export one JSON object per frame, or one JSON array per video, with at least:

```json
{
  "video_id": "pushup_001",
  "frame_idx": 123,
  "time_ms": 8200,
  "sport_type": "push_up",
  "stage_label": "down",
  "score": 0.91,
  "keypoints": [[x, y, c], [x, y, c], "... 17 items total"],
  "features": {
    "pushup_angle": 92.3,
    "squat_angle": 176.0,
    "pushup_depth": 0.84,
    "torso_linearity": 0.88,
    "torso_horizontal": 0.91,
    "torso_vertical": 0.09,
    "left_elbow_angle": 90.2,
    "right_elbow_angle": 94.1,
    "left_knee_angle": 176.8,
    "right_knee_angle": 175.5,
    "hip_to_shoulder_y_diff": -0.62,
    "hip_to_wrist_y_diff": 0.33
  }
}
```

## Feature Design

Each frame is converted into a `55`-dim feature vector:

1. `12` upper/lower body keypoints with normalized `x, y, conf`
   - indices: `5,6,7,8,9,10,11,12,13,14,15,16`
   - dims: `12 * 3 = 36`
2. handcrafted features
   - dims: `12`
3. temporal deltas for key motion indicators
   - dims: `6`
4. global pose score
   - dims: `1`

Total: `36 + 12 + 6 + 1 = 55`

## Labels

Recommended window action labels:

- `background`
- `push_up`
- `sit_up`

Recommended frame stage labels:

- `background`
- `up`
- `down`
- `transition`

For sit-up data, you can either:

- reuse the same generic stage names, or
- map them in a task-specific way such as `lie`, `top`, `transition`

If you want a single shared model, keep the generic 4-class stage space.

## Train On CPU

This baseline is intentionally small enough for CPU training:

- sequence length: `32`
- feature dim: `55`
- model dim: `96`
- transformer layers: `2`
- attention heads: `4`

Suggested first run:

```bash
python ai_sport/training/train.py --data D:/datasets/ai_sport/train.json --epochs 20 --batch-size 16
```

## Suggested Workflow

1. Export frame-level pose records from the Android pipeline or an offline pose script.
2. Label `sport_type` and `stage_label`.
3. Train the baseline model in this folder.
4. Validate with:
   - frame stage accuracy / F1
   - window action accuracy
   - repetition MAE after state-machine counting
5. Export to ONNX.
6. Convert ONNX to MNN after accuracy is stable.




编译模型：
docker run -it --rm `
  -v "D:\open-project\ali_ai_match\MNN\MNN:/workspace/MNN" `
  -v "D:\open-project\ali_ai_match\ai_sport:/workspace/ai_sport" `
  -w /workspace/MNN `
  mnn:v1.0 /bin/bash


/workspace/MNN/build-linux-converter/MNNConvert   -f ONNX   --modelFile /workspace/ai_sport/rep_counter.onnx   --MNNModel /workspace/ai_sport/rep_counter.mnn   --bizCode AI_SPORT