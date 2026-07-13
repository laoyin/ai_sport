# Deploy Rep Counter To MNN

This is the recommended path for the trained `Conv1D + Tiny Transformer` rep counter.

## 1. Export PyTorch checkpoint to ONNX

From the workspace root:

```powershell
python ai_sport/training/export_onnx.py `
  --ckpt D:\open-project\ali_ai_match\ai_sport\training\checkpoints\best_conv_tiny_transformer.pt `
  --output D:\open-project\ali_ai_match\ai_sport\rep_counter.onnx
```

Expected input shape:

```text
[1, 32, 55]
```

Outputs:

- `stage_logits`: `[1, 32, 4]`
- `action_logits`: `[1, 3]`

## 2. Convert ONNX to MNN

Use the bundled MNN converter already present in this repo:

```powershell
powershell -ExecutionPolicy Bypass -File ai_sport/training/convert_to_mnn.ps1
```

Or specify paths explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File ai_sport/training/convert_to_mnn.ps1 `
  -OnnxPath D:\open-project\ali_ai_match\ai_sport\rep_counter.onnx `
  -MnnPath D:\open-project\ali_ai_match\ai_sport\rep_counter.mnn `
  -MnnConvertPath D:\open-project\ali_ai_match\MNN\MNN\build-android-arm64-4b-onnx\MNNConvert
```

## 3. Put the MNN model into the Android app

Recommended asset path:

```text
app/src/main/assets/rep/rep_counter.mnn
```

Or keep the current generated-assets style and sync it during Gradle build, similar to the pose model.

## 4. On-device input pipeline

The Android app already has most of the required pieces:

1. Video is sampled in `VideoFrameSampler.kt`
2. Pose keypoints come from `NativePoseEstimator`
3. Handcrafted features are computed in `ExerciseAnalyzer.buildFrameSample`

To match training exactly, the app must build a rolling window of:

- `32` frames
- `55` features per frame

Feature order must match `ai_sport/training/dataset.py`:

1. normalized keypoints for indices `5,6,7,8,9,10,11,12,13,14,15,16`
2. handcrafted features
3. delta features
4. pose score

## 5. On-device output pipeline

The MNN model outputs:

- `stage_logits`
- `action_logits`

Recommended post-process:

1. `argmax(action_logits)` -> current action class
2. `argmax(stage_logits[t])` -> stage per frame
3. Sliding-window vote smoothing over stage labels
4. Count reps by contiguous `down` segments for `push_up`

This is the same logic used in `predict_videos.py`.

## 6. Validation before Android integration

Before wiring JNI, verify consistency on the desktop:

```powershell
python ai_sport/training/predict_videos.py `
  --data D:\open-project\ali_ai_match\ai_sport\ai_sport_labels `
  --counts D:\open-project\ali_ai_match\ai_sport\ai_sport_labels\counts.json
```

If those counts look right, then move the exact same windowing + counting logic into the Android app.
