param(
    [string]$OnnxPath = "D:\open-project\ali_ai_match\ai_sport\rep_counter.onnx",
    [string]$MnnPath = "D:\open-project\ali_ai_match\ai_sport\rep_counter.mnn",
    [string]$MnnConvertPath = "D:\open-project\ali_ai_match\MNN\MNN\build-android-arm64-4b-onnx\MNNConvert"
)

if (!(Test-Path $OnnxPath)) {
    throw "ONNX file not found: $OnnxPath"
}
if (!(Test-Path $MnnConvertPath)) {
    throw "MNNConvert not found: $MnnConvertPath"
}

& $MnnConvert -f ONNX --modelFile $OnnxPath --MNNModel $MnnPath --bizCode AI_SPORT

if ($LASTEXITCODE -ne 0) {
    throw "MNNConvert failed with exit code $LASTEXITCODE"
}

Write-Output "converted_mnn=$MnnPath"
