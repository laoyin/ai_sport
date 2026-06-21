# AI Sport MVP

## Open Source

- GitHub: [https://github.com/laoyin/ai_sport](https://github.com/laoyin/ai_sport)

当前版本是一个最快可落地的 Android MVP：

- 选择一张运动图片
- 直接相机拍照
- 选择或录制一段运动视频
- 从视频里自动抽取关键帧
- 使用 `Qwen3-VL-MNN` 做结构化运动点评
- 自动生成一张可保存到相册的运动宣传卡

## 当前实现

- Android 独立工程：`ai_sport/`
- MNN + Qwen3-VL JNI 接入
- 自动从工作区 `qwen3-vl-mnn/` 同步模型资源到打包资产
- 自动从 `MNN/MNN` 选择并同步 `libMNN.so / libMNN_Express.so / libllm.so`
- 相册选图
- 相机拍照输入
- 相册选视频 / 相机录视频
- 视频关键帧自动选择
- `YOLO-Pose` 标准接口占位
- 关键帧骨架叠加到宣传卡
- 结构化 JSON 解析
- 海报 Canvas 合成
- 保存到系统相册

## 代码入口

- App UI: [MainActivity.kt](/D:/open-project/ali_ai_match/ai_sport/app/src/main/java/com/aisport/ui/MainActivity.kt)
- 视觉分析: [SportAnalyzer.kt](/D:/open-project/ali_ai_match/ai_sport/app/src/main/java/com/aisport/vision/SportAnalyzer.kt)
- Pose 接口: [PoseEstimator.kt](/D:/open-project/ali_ai_match/ai_sport/app/src/main/java/com/aisport/pose/PoseEstimator.kt)
- 视频抽帧: [VideoFrameSampler.kt](/D:/open-project/ali_ai_match/ai_sport/app/src/main/java/com/aisport/video/VideoFrameSampler.kt)
- 海报生成: [PosterComposer.kt](/D:/open-project/ali_ai_match/ai_sport/app/src/main/java/com/aisport/poster/PosterComposer.kt)
- JNI 推理: [mnn_jni.cpp](/D:/open-project/ali_ai_match/ai_sport/app/src/main/cpp/mnn_jni.cpp)
- Gradle 配置: [app/build.gradle.kts](/D:/open-project/ali_ai_match/ai_sport/app/build.gradle.kts)

## 说明

- 当前版本已经打通“相机/图片/视频输入 -> 关键帧 -> 运动分析 -> 出图”闭环。
- `YOLO-Pose` 目前是稳定接口位 + 占位实现，后续可直接替换成真实 MNN / ONNX 推理输出。
- 下一步建议直接补：
  - 真实 `YOLO-Pose` 推理
  - 深蹲深度 / 膝盖内扣 / 左右对称性规则
  - 连续动作计数
  - 视频级训练战报

## 编译提示

- 这个目录目前没有单独放 `gradlew` wrapper。
- 建议直接用 Android Studio 打开 `ai_sport/`。
- 如果你后面想让我继续，我下一步就直接接 `YOLO-Pose` 的输入层和关键帧选择逻辑。
