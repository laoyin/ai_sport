package com.aisport.ui

import android.Manifest
import android.content.ClipData
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.aisport.camera.LiveWorkoutChart
import com.aisport.camera.LiveWorkoutSession
import com.aisport.camera.LiveWorkoutSnapshot
import com.aisport.camera.RealtimeFrameAnalyzer
import com.aisport.engine.MnnEngine
import com.aisport.exercise.ExerciseFrameSample
import com.aisport.poster.PoseDebugRenderer
import com.aisport.poster.PosterComposer
import com.aisport.poster.WorkoutTimelineRenderer
import com.aisport.pose.NativePoseEstimator
import com.aisport.pose.PoseEstimate
import com.aisport.rep.NativeRepCounter
import com.aisport.training.AnnotationExport
import com.aisport.video.SampledPoseFrame
import com.aisport.video.VideoFrameSampler
import com.aisport.video.VideoMotionSummary
import com.aisport.vision.SportAnalysis
import com.aisport.vision.SportAnalyzer
import com.aisport.workout.WorkoutInsights
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.concurrent.thread

data class SportUiState(
    val modelReady: Boolean = false,
    val loading: Boolean = false,
    val status: String = "正在准备模型…",
    val analysisMode: String = "auto",
    val analysis: SportAnalysis? = null,
    val selectedBitmap: Bitmap? = null,
    val diagnosticBitmap: Bitmap? = null,
    val sampledDiagnosticFrames: List<DiagnosticFramePreview> = emptyList(),
    val sampledPoseFrames: List<SampledPoseFrame> = emptyList(),
    val frameSamples: List<ExerciseFrameSample> = emptyList(),
    val timelineBitmap: Bitmap? = null,
    val poseEstimate: PoseEstimate? = null,
    val motionSummary: VideoMotionSummary? = null,
    val posterBitmap: Bitmap? = null,
    val lastSavedUri: String? = null,
    val lastExportDir: String? = null,
    val sourceLabel: String = "未选择输入",
    val liveMode: Boolean = false,
    val liveRunning: Boolean = false,
    val liveRepCount: Int = 0,
    val liveConfidence: Float = 0f,
    val liveWaveform: List<Float> = emptyList(),
    val liveElapsedMs: Long = 0L,
    val liveSportType: String = "unknown",
    val liveDebug: String = "",
    val liveCalories: Float = 0f,
    val liveAverageRepSeconds: Float = 0f,
    val liveOverlayBitmap: Bitmap? = null
)

data class DiagnosticFramePreview(
    val timeMs: Long,
    val score: Float,
    val bitmap: Bitmap
)

class MainActivity : ComponentActivity() {

    private lateinit var mnnEngine: MnnEngine
    private lateinit var sportAnalyzer: SportAnalyzer
    private lateinit var poseEstimator: NativePoseEstimator
    private lateinit var repCounter: NativeRepCounter

    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var livePreviewView: PreviewView? = null
    private var liveCameraBound = false
    private var liveWorkoutSession: LiveWorkoutSession? = null
    private var pendingVideoUri: Uri? = null

    private var uiState by mutableStateOf(SportUiState())

    private val requestCameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startLiveWorkoutMode()
        } else {
            uiState = uiState.copy(status = "未授予相机权限，无法启动实时运动模式。")
        }
    }

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) loadBitmap(uri)
    }

    private val takePicturePreviewLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) applySelectedBitmap(bitmap, "相机拍照")
    }

    private val pickVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) loadVideo(uri, "相册视频")
    }

    private val captureVideoLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val uri = pendingVideoUri
        pendingVideoUri = null
        if (result.resultCode == RESULT_OK && uri != null) {
            loadVideo(uri, "相机录制视频")
        } else {
            uiState = uiState.copy(status = "已取消录制视频。")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mnnEngine = MnnEngine(this)
        sportAnalyzer = SportAnalyzer(cacheDir, mnnEngine)
        poseEstimator = NativePoseEstimator(this)
        repCounter = NativeRepCounter(this)
        bootstrapModel()
        setContent { AiSportScreen() }
    }

    override fun onDestroy() {
        cameraProvider?.unbindAll()
        liveCameraBound = false
        cameraExecutor.shutdown()
        mnnEngine.release()
        poseEstimator.release()
        repCounter.release()
        super.onDestroy()
    }

    private fun bootstrapModel() {
        thread {
            val llmReady = mnnEngine.loadModelBundle("qwen3-vl-mnn")
            val poseReady = poseEstimator.loadModel()
            val repReady = repCounter.loadModel()
            runOnUiThread {
                uiState = uiState.copy(
                    modelReady = llmReady && poseReady,
                    status = when {
                        llmReady && poseReady && repReady -> "AI Sport 已就绪，支持实时识别、战报和宣传卡。"
                        llmReady && poseReady -> "基础模型已就绪，时序模型未加载，将回退到规则计数。"
                        else -> "模型加载失败，请确认资源文件已打包。"
                    }
                )
            }
        }
    }

    private fun ensureCameraPermissionAndStart() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (granted) startLiveWorkoutMode() else requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startLiveWorkoutMode() {
        liveWorkoutSession = LiveWorkoutSession(
            poseEstimator = { bitmap -> poseEstimator.estimate(bitmap) },
            repCounter = repCounter,
            forcedSportType = uiState.analysisMode
        )
        uiState = uiState.copy(
            liveMode = true,
            liveRunning = true,
            liveRepCount = 0,
            liveConfidence = 0f,
            liveWaveform = emptyList(),
            liveElapsedMs = 0L,
            liveSportType = "unknown",
            liveDebug = "",
            liveCalories = 0f,
            liveAverageRepSeconds = 0f,
            liveOverlayBitmap = null,
            selectedBitmap = null,
            diagnosticBitmap = null,
            sampledDiagnosticFrames = emptyList(),
            sampledPoseFrames = emptyList(),
            frameSamples = emptyList(),
            timelineBitmap = null,
            poseEstimate = null,
            motionSummary = null,
            posterBitmap = null,
            analysis = null,
            lastSavedUri = null,
            lastExportDir = null,
            sourceLabel = "实时运动模式",
            status = "实时运动模式已启动，请开始动作。"
        )
        liveCameraBound = false
        livePreviewView?.let { bindLiveCamera(it) }
    }

    private fun stopLiveWorkoutMode() {
        cameraProvider?.unbindAll()
        liveCameraBound = false
        val snapshot = liveWorkoutSession?.finish()
        liveWorkoutSession = null
        if (snapshot == null) {
            uiState = uiState.copy(
                liveMode = false,
                liveRunning = false,
                status = "实时运动已结束，但有效动作帧不足。"
            )
            return
        }
        finalizeLiveWorkout(snapshot)
    }

    private fun bindLiveCamera(previewView: PreviewView) {
        if (!uiState.liveMode || !uiState.liveRunning) return
        if (liveCameraBound && livePreviewView === previewView) return
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(cameraExecutor, RealtimeFrameAnalyzer { bitmap, timestampMs ->
                        val metrics = liveWorkoutSession?.processFrame(bitmap, timestampMs) ?: return@RealtimeFrameAnalyzer
                        runOnUiThread {
                            uiState = uiState.copy(
                                liveRepCount = metrics.repetitionCount,
                                liveConfidence = metrics.confidence,
                                liveWaveform = metrics.waveform,
                                liveElapsedMs = metrics.elapsedMs,
                                liveSportType = metrics.inferredSportType,
                                liveDebug = metrics.debug,
                                liveCalories = metrics.calories,
                                liveAverageRepSeconds = metrics.averageRepSeconds,
                                liveOverlayBitmap = metrics.overlayBitmap,
                                poseEstimate = metrics.poseEstimate,
                                status = "实时识别中：${sportTypeLabel(metrics.inferredSportType)} ${metrics.repetitionCount} 次"
                            )
                        }
                    })
                }

            provider.unbindAll()
            try {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
            } catch (_: Throwable) {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
            liveCameraBound = true
        }, ContextCompat.getMainExecutor(this))
    }

    private fun finalizeLiveWorkout(snapshot: LiveWorkoutSnapshot) {
        val diagnosticBitmap = PoseDebugRenderer.render(
            snapshot.bestFrame,
            snapshot.bestPoseEstimate,
            snapshot.sourceLabel
        )
        uiState = uiState.copy(
            liveMode = false,
            liveRunning = false,
            selectedBitmap = snapshot.bestFrame,
            diagnosticBitmap = diagnosticBitmap,
            frameSamples = snapshot.frameSamples,
            timelineBitmap = WorkoutTimelineRenderer.render(snapshot.motionSummary),
            poseEstimate = snapshot.bestPoseEstimate,
            motionSummary = snapshot.motionSummary,
            sourceLabel = snapshot.sourceLabel,
            status = "运动已结束，正在生成 AI 运动战报…"
        )
        analyzeAndGeneratePoster()
    }

    private fun loadBitmap(uri: Uri) {
        contentResolver.openInputStream(uri)?.use { stream ->
            val bitmap = BitmapFactory.decodeStream(stream)
            if (bitmap != null) applySelectedBitmap(bitmap, "相册图片")
        }
    }

    private fun applySelectedBitmap(bitmap: Bitmap, sourceLabel: String) {
        val poseEstimate = poseEstimator.estimate(bitmap)
        val diagnosticBitmap = PoseDebugRenderer.render(bitmap, poseEstimate, sourceLabel)
        uiState = uiState.copy(
            selectedBitmap = bitmap,
            diagnosticBitmap = diagnosticBitmap,
            sampledDiagnosticFrames = emptyList(),
            sampledPoseFrames = emptyList(),
            frameSamples = emptyList(),
            timelineBitmap = null,
            poseEstimate = poseEstimate,
            motionSummary = null,
            posterBitmap = null,
            analysis = null,
            lastSavedUri = null,
            lastExportDir = null,
            sourceLabel = sourceLabel,
            status = "YOLO Pose 已完成关键点检测，当前为单图分析模式。"
        )
    }

    private fun loadVideo(uri: Uri, sourceLabel: String) {
        uiState = uiState.copy(
            loading = true,
            selectedBitmap = null,
            diagnosticBitmap = null,
            sampledDiagnosticFrames = emptyList(),
            sampledPoseFrames = emptyList(),
            frameSamples = emptyList(),
            timelineBitmap = null,
            poseEstimate = null,
            motionSummary = null,
            posterBitmap = null,
            analysis = null,
            lastSavedUri = null,
            lastExportDir = null,
            sourceLabel = sourceLabel,
            status = "正在对视频抽帧并进行动作计数… 当前模式：${analysisModeLabel(uiState.analysisMode)}"
        )
        thread {
            val result = VideoFrameSampler.extractBestFrame(
                context = this,
                uri = uri,
                poseEstimator = poseEstimator,
                repCounter = repCounter,
                forcedSportType = uiState.analysisMode
            )
            runOnUiThread {
                if (result != null) {
                    val keyframeLabel = "$sourceLabel · 代表帧 ${result.bestFrameTimeMs}ms / ${result.durationMs}ms · 检测 ${result.sampledFrames} 帧"
                    val diagnosticBitmap = PoseDebugRenderer.render(
                        result.bestFrame,
                        result.poseEstimate,
                        keyframeLabel
                    )
                    uiState = uiState.copy(
                        loading = false,
                        selectedBitmap = result.bestFrame,
                        diagnosticBitmap = diagnosticBitmap,
                        sampledDiagnosticFrames = buildSampledDiagnosticFrames(result.sampledPoseFrames),
                        sampledPoseFrames = result.sampledPoseFrames,
                        frameSamples = result.frameSamples,
                        timelineBitmap = WorkoutTimelineRenderer.render(result.motionSummary),
                        poseEstimate = result.poseEstimate,
                        motionSummary = result.motionSummary,
                        sourceLabel = keyframeLabel,
                        status = "视频分析完成：${sportTypeLabel(result.motionSummary.inferredSportType)} ${result.motionSummary.repetitionCount} 次"
                    )
                } else {
                    uiState = uiState.copy(
                        loading = false,
                        status = "视频关键帧提取失败，请换一段更清晰的运动视频。"
                    )
                }
            }
        }
    }

    private fun captureVideo() {
        val uri = createCaptureVideoUri()
        pendingVideoUri = uri
        val intent = Intent(MediaStore.ACTION_VIDEO_CAPTURE).apply {
            putExtra(MediaStore.EXTRA_OUTPUT, uri)
            clipData = ClipData.newUri(contentResolver, "ai_sport_capture_video", uri)
            addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try {
            uiState = uiState.copy(status = "正在打开系统相机录制视频…")
            captureVideoLauncher.launch(intent)
        } catch (_: Throwable) {
            pendingVideoUri = null
            uiState = uiState.copy(status = "打开录制视频失败，请确认设备上有可用相机应用。")
            Toast.makeText(this, "无法打开录制视频", Toast.LENGTH_SHORT).show()
        }
    }

    private fun createCaptureVideoUri(): Uri {
        val dir = File(getExternalFilesDir(null) ?: cacheDir, "captures").apply { mkdirs() }
        val file = File(dir, "sport_capture_${System.currentTimeMillis()}.mp4")
        return FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
    }

    private fun analyzeAndGeneratePoster() {
        val originalInput = uiState.selectedBitmap ?: return
        val analysisInput = uiState.diagnosticBitmap ?: originalInput
        val poseEstimate = uiState.poseEstimate
        val motionSummary = uiState.motionSummary
        val sourceLabel = uiState.sourceLabel
        uiState = uiState.copy(loading = true, status = "正在生成 AI 运动战报和宣传卡…")
        thread {
            val analysis = sportAnalyzer.analyze(analysisInput, poseEstimate, motionSummary)
            val poster = analysis?.let {
                PosterComposer.createPoster(
                    source = analysisInput,
                    analysis = it,
                    poseEstimate = null,
                    sourceLabel = sourceLabel
                )
            }
            runOnUiThread {
                uiState = uiState.copy(
                    loading = false,
                    analysis = analysis,
                    posterBitmap = poster,
                    status = if (analysis != null && poster != null) {
                        "AI 运动战报已生成，可以保存或分享。"
                    } else {
                        "总结生成失败，请再试一次。"
                    }
                )
            }
        }
    }

    private fun savePoster() {
        val poster = uiState.posterBitmap ?: return
        val uri = PosterComposer.savePoster(this, poster)
        uiState = uiState.copy(
            lastSavedUri = uri,
            status = if (uri != null) "海报已保存到系统相册。" else "保存失败，请重试。"
        )
        Toast.makeText(this, uiState.status, Toast.LENGTH_SHORT).show()
    }

    private fun exportAnnotationPackage() {
        if (uiState.sampledPoseFrames.isEmpty() || uiState.frameSamples.isEmpty()) {
            uiState = uiState.copy(status = "当前没有可导出的标注数据，请先分析一段视频。")
            return
        }
        uiState = uiState.copy(
            loading = true,
            status = "正在导出标注数据包，包含时序 JSON 和 Pose 图片…"
        )
        thread {
            try {
                val result = AnnotationExport.exportVideoAnnotationPackage(
                    context = this,
                    videoLabel = uiState.sourceLabel,
                    analysisMode = uiState.analysisMode,
                    sampledPoseFrames = uiState.sampledPoseFrames,
                    frameSamples = uiState.frameSamples,
                    motionSummary = uiState.motionSummary
                )
                runOnUiThread {
                    uiState = uiState.copy(
                        loading = false,
                        lastExportDir = result.directory.absolutePath,
                        status = "标注数据已导出：${result.frameCount} 帧 · ${result.directory.absolutePath}"
                    )
                    Toast.makeText(this, "标注数据已导出", Toast.LENGTH_SHORT).show()
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    uiState = uiState.copy(
                        loading = false,
                        status = "导出标注数据失败：${t.message ?: "unknown error"}"
                    )
                    Toast.makeText(this, "导出失败", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun sharePosterImage() {
        val poster = uiState.posterBitmap ?: return
        val uri = PosterComposer.createShareUri(this, poster)
        if (uri == null) {
            uiState = uiState.copy(status = "海报导出失败，请重新生成后再试。")
            Toast.makeText(this, "海报导出失败", Toast.LENGTH_SHORT).show()
            return
        }

        val weChatIntent = Intent(Intent.ACTION_SEND).apply {
            type = "image/*"
            putExtra(Intent.EXTRA_STREAM, uri)
            clipData = ClipData.newUri(contentResolver, "ai_sport_poster", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            setPackage("com.tencent.mm")
        }
        grantShareUriPermission(uri, weChatIntent)
        try {
            startActivity(weChatIntent)
            uiState = uiState.copy(status = "已拉起微信，可以继续分享到好友或朋友圈。")
            return
        } catch (_: Throwable) {
        }

        val genericShare = Intent(Intent.ACTION_SEND).apply {
            type = "image/*"
            putExtra(Intent.EXTRA_STREAM, uri)
            clipData = ClipData.newUri(contentResolver, "ai_sport_poster", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        grantShareUriPermission(uri, genericShare)
        try {
            startActivity(Intent.createChooser(genericShare, "分享运动战报"))
            uiState = uiState.copy(status = "已拉起系统分享面板。")
        } catch (_: Throwable) {
            uiState = uiState.copy(status = "当前设备没有可用的分享应用。")
            Toast.makeText(this, "无法拉起分享", Toast.LENGTH_SHORT).show()
        }
    }

    private fun grantShareUriPermission(uri: Uri, intent: Intent) {
        packageManager.queryIntentActivities(intent, 0).forEach { resolveInfo ->
            grantUriPermission(
                resolveInfo.activityInfo.packageName,
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun AiSportScreen() {
        MaterialTheme {
            Scaffold(
                topBar = {
                    TopAppBar(
                        title = { Text("AI Sport", fontWeight = FontWeight.Bold) },
                        colors = TopAppBarDefaults.topAppBarColors(
                            containerColor = Color(0xFFF6F1E7),
                            titleContentColor = Color(0xFF111827)
                        )
                    )
                },
                containerColor = Color(0xFFF7F3EA)
            ) { paddingValues ->
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.verticalGradient(
                                listOf(
                                    Color(0xFFF7F0E2),
                                    Color(0xFFF9FBFF),
                                    Color(0xFFF1F7EE)
                                )
                            )
                        )
                        .padding(paddingValues)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(18.dp)
                ) {
                    HeroHeader(uiState)
                    Column(
                        modifier = Modifier.padding(horizontal = 16.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        StatusCard(uiState)
                        ActionArea()
                        if (uiState.liveMode) {
                            LiveWorkoutCard(uiState)
                        }
                        if (!uiState.liveMode && (uiState.motionSummary != null || uiState.analysis != null)) {
                            WorkoutReportCard(
                                analysis = uiState.analysis,
                                motionSummary = uiState.motionSummary,
                                samples = uiState.frameSamples,
                                fallbackQuality = uiState.poseEstimate?.qualityHint
                            )
                        }
                        PreviewCard("最佳动作截图", uiState.selectedBitmap)
                        PreviewCard("姿态骨架诊断", uiState.diagnosticBitmap)
                        PreviewCard("运动波动曲线", uiState.timelineBitmap)
                        PreviewCard("生成的宣传海报", uiState.posterBitmap)
                        SampledFrameGallery(uiState.sampledDiagnosticFrames)
                        AnalysisCard(uiState.analysis, uiState.lastSavedUri)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }

    @Composable
    private fun HeroHeader(state: SportUiState) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            shape = RoundedCornerShape(32.dp),
            colors = CardDefaults.cardColors(containerColor = Color.Transparent)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Brush.linearGradient(
                            listOf(
                                Color(0xFF0F172A),
                                Color(0xFF21415C),
                                Color(0xFF2F6A60)
                            )
                        ),
                        shape = RoundedCornerShape(32.dp)
                    )
                    .padding(24.dp)
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text(
                        "AI SPORT COACH",
                        color = Color.White.copy(alpha = 0.74f),
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 12.sp
                    )
                    Text(
                        if (state.liveRunning) "把每一次发力\n变成可见反馈" else "把训练过程\n变成可分享的战报",
                        color = Color.White,
                        fontWeight = FontWeight.ExtraBold,
                        fontSize = 30.sp,
                        lineHeight = 34.sp
                    )
                    Text(
                        if (state.liveRunning) {
                            "${sportTypeLabel(state.liveSportType)} · ${state.liveRepCount} 次 · ${WorkoutInsights.formatCalories(state.liveCalories)}"
                        } else {
                            "端侧推理 · 实时计数 · 姿态分析 · AI 战报"
                        },
                        color = Color(0xFFDCE6E0),
                        fontSize = 15.sp
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        HeroBadge(if (state.modelReady) "模型已就绪" else "模型加载中")
                        HeroBadge(if (state.liveRunning) "实时模式" else "海报模式")
                    }
                }
            }
        }
    }

    @Composable
    private fun HeroBadge(label: String) {
        Card(
            shape = RoundedCornerShape(99.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White.copy(alpha = 0.14f))
        ) {
            Text(
                text = label,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }

    @Composable
    private fun StatusCard(state: SportUiState) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(26.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(
                modifier = Modifier.padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                Text(
                    "当前状态",
                    fontWeight = FontWeight.ExtraBold,
                    color = Color(0xFF111827),
                    fontSize = 22.sp
                )
                Text(
                    state.status,
                    color = Color(0xFF475569),
                    lineHeight = 22.sp
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    SmallMetricCard("模型", if (state.modelReady) "已就绪" else "未就绪", Modifier.weight(1f))
                    SmallMetricCard("模式", analysisModeLabel(state.analysisMode), Modifier.weight(1f))
                    SmallMetricCard("海报", if (state.posterBitmap != null) "已生成" else "待生成", Modifier.weight(1f))
                }
                Text(
                    "输入来源：${state.sourceLabel}",
                    color = Color(0xFF334155)
                )
                state.poseEstimate?.let {
                    Text(
                        "Pose · ${it.engineName} · score=${"%.2f".format(it.score)} · ${it.qualityHint}",
                        color = Color(0xFF6D4CC3)
                    )
                }
                if (state.loading) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(10.dp))
                        Text("正在处理模型结果…", color = Color(0xFF475569))
                    }
                }
            }
        }
    }

    @Composable
    private fun ActionArea() {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(26.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(
                modifier = Modifier.padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text("训练控制台", fontWeight = FontWeight.ExtraBold, color = Color(0xFF111827), fontSize = 22.sp)
                Text(
                    "先选动作模式，再进入实时训练或导入素材生成战报。",
                    color = Color(0xFF64748B)
                )

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("动作模式", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        ModeChip("自动", uiState.analysisMode == "auto", Modifier.weight(1f)) {
                            uiState = uiState.copy(analysisMode = "auto", status = "已切换到自动识别模式。")
                        }
                        ModeChip("深蹲", uiState.analysisMode == "squat", Modifier.weight(1f)) {
                            uiState = uiState.copy(analysisMode = "squat", status = "已切换到深蹲计数模式。")
                        }
                        ModeChip("俯卧撑", uiState.analysisMode == "push_up", Modifier.weight(1f)) {
                            uiState = uiState.copy(analysisMode = "push_up", status = "已切换到俯卧撑计数模式。")
                        }
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("实时训练", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        PrimaryActionButton(
                            label = "开始实时运动",
                            enabled = uiState.modelReady && !uiState.liveRunning,
                            modifier = Modifier.weight(1f),
                            onClick = { ensureCameraPermissionAndStart() }
                        )
                        SecondaryActionButton(
                            label = "结束运动",
                            enabled = uiState.liveRunning,
                            modifier = Modifier.weight(1f),
                            onClick = { stopLiveWorkoutMode() }
                        )
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("导入素材", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        PrimaryActionButton("选择运动图", true, Modifier.weight(1f)) {
                            pickImageLauncher.launch("image/*")
                        }
                        PrimaryActionButton("相机拍照", true, Modifier.weight(1f)) {
                            takePicturePreviewLauncher.launch(null)
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        SecondaryActionButton("选择视频", true, Modifier.weight(1f)) {
                            pickVideoLauncher.launch("video/*")
                        }
                        SecondaryActionButton("录制视频", true, Modifier.weight(1f)) {
                            captureVideo()
                        }
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("输出结果", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        PrimaryActionButton(
                            label = "生成战报海报",
                            enabled = uiState.modelReady && uiState.selectedBitmap != null && !uiState.loading,
                            modifier = Modifier.weight(1f),
                            onClick = { analyzeAndGeneratePoster() }
                        )
                        SecondaryActionButton(
                            label = "保存到相册",
                            enabled = uiState.posterBitmap != null && !uiState.loading,
                            modifier = Modifier.weight(1f),
                            onClick = { savePoster() }
                        )
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        SecondaryActionButton(
                            label = "导出标注数据",
                            enabled = uiState.sampledPoseFrames.isNotEmpty() && !uiState.loading,
                            modifier = Modifier.weight(1f),
                            onClick = { exportAnnotationPackage() }
                        )
                        SecondaryActionButton(
                            label = "分享微信 / 朋友圈",
                            enabled = uiState.posterBitmap != null && !uiState.loading,
                            modifier = Modifier.weight(1f),
                            onClick = { sharePosterImage() }
                        )
                    }
                }
            }
        }
    }

    @Composable
    private fun ModeChip(label: String, selected: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
        Button(
            onClick = onClick,
            modifier = modifier,
            shape = RoundedCornerShape(18.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (selected) Color(0xFF111827) else Color(0xFFF3F4F6),
                contentColor = if (selected) Color.White else Color(0xFF475569)
            )
        ) {
            Text(label, fontWeight = FontWeight.SemiBold)
        }
    }

    @Composable
    private fun PrimaryActionButton(
        label: String,
        enabled: Boolean,
        modifier: Modifier = Modifier,
        onClick: () -> Unit
    ) {
        Button(
            onClick = onClick,
            enabled = enabled,
            modifier = modifier,
            shape = RoundedCornerShape(20.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF6D4CC3),
                contentColor = Color.White,
                disabledContainerColor = Color(0xFFD8D2EA),
                disabledContentColor = Color.White.copy(alpha = 0.72f)
            )
        ) {
            Text(label, fontWeight = FontWeight.SemiBold)
        }
    }

    @Composable
    private fun SecondaryActionButton(
        label: String,
        enabled: Boolean,
        modifier: Modifier = Modifier,
        onClick: () -> Unit
    ) {
        OutlinedButton(
            onClick = onClick,
            enabled = enabled,
            modifier = modifier,
            shape = RoundedCornerShape(20.dp)
        ) {
            Text(label, fontWeight = FontWeight.SemiBold)
        }
    }

    @Composable
    private fun LiveWorkoutCard(state: SportUiState) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(28.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                Text("实时运动教练", fontWeight = FontWeight.ExtraBold, color = Color(0xFF111827), fontSize = 24.sp)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(420.dp)
                        .background(
                            brush = Brush.linearGradient(
                                listOf(Color(0xFFF2E8D7), Color(0xFFF8FBFF), Color(0xFFEAF7F0))
                            ),
                            shape = RoundedCornerShape(24.dp)
                        )
                ) {
                    AndroidView(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(420.dp),
                        factory = { context ->
                            PreviewView(context).also { previewView ->
                                previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
                                livePreviewView = previewView
                                bindLiveCamera(previewView)
                            }
                        },
                        update = { previewView ->
                            livePreviewView = previewView
                            bindLiveCamera(previewView)
                        }
                    )
                    Card(
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(16.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xCC0F172A))
                    ) {
                        Column(modifier = Modifier.padding(horizontal = 18.dp, vertical = 14.dp)) {
                            Text("实时次数", color = Color.White.copy(alpha = 0.78f), fontSize = 14.sp)
                            Text(
                                state.liveRepCount.toString(),
                                color = Color.White,
                                fontWeight = FontWeight.ExtraBold,
                                fontSize = 48.sp
                            )
                            Text(
                                sportTypeLabel(state.liveSportType),
                                color = Color(0xFFF6D777),
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                    Card(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(16.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xF7FFFFFF))
                    ) {
                        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text("持续时长 ${WorkoutInsights.formatDuration(state.liveElapsedMs)}", color = Color(0xFF0F172A))
                            Text("热量 ${WorkoutInsights.formatCalories(state.liveCalories)}", color = Color(0xFFB45309))
                            Text("节奏 ${WorkoutInsights.formatPace(state.liveAverageRepSeconds)}", color = Color(0xFF0F766E))
                            Text("置信度 ${"%.2f".format(state.liveConfidence)}", color = Color(0xFF475569))
                        }
                    }
                }

                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    Card(
                        modifier = Modifier.weight(1.05f),
                        shape = RoundedCornerShape(22.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC))
                    ) {
                        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("实时姿态骨架", fontWeight = FontWeight.SemiBold, color = Color(0xFF111827))
                            val overlay = state.liveOverlayBitmap
                            if (overlay != null) {
                                Image(
                                    bitmap = overlay.asImageBitmap(),
                                    contentDescription = "live_overlay",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(180.dp)
                                        .background(Color.White, RoundedCornerShape(16.dp)),
                                    contentScale = ContentScale.Crop
                                )
                            } else {
                                PlaceholderBox("等待骨架识别…", 180)
                            }
                        }
                    }
                    Card(
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(22.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC))
                    ) {
                        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("动作波动曲线", fontWeight = FontWeight.SemiBold, color = Color(0xFF111827))
                            LiveWorkoutChart(values = state.liveWaveform)
                            Text(
                                state.liveDebug.ifBlank { "模型正在持续统计动作节奏…" },
                                color = Color(0xFF64748B),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun WorkoutReportCard(
        analysis: SportAnalysis?,
        motionSummary: VideoMotionSummary?,
        samples: List<ExerciseFrameSample>,
        fallbackQuality: String?
    ) {
        val metrics = WorkoutInsights.buildSummary(motionSummary, samples, fallbackQuality, analysis?.sportType)
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(28.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                Text("AI 运动战报", fontWeight = FontWeight.ExtraBold, fontSize = 24.sp, color = Color(0xFF111827))
                Text(
                    analysis?.summaryTitle?.ifBlank { "本次训练表现稳定，继续保持。" } ?: "本次训练表现稳定，继续保持。",
                    color = Color(0xFF475569)
                )
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    SmallMetricCard("总次数", (motionSummary?.repetitionCount ?: 0).toString(), Modifier.weight(1f))
                    SmallMetricCard("持续时长", WorkoutInsights.formatDuration(metrics.durationMs), Modifier.weight(1f))
                    SmallMetricCard("平均节奏", WorkoutInsights.formatPace(metrics.averageRepSeconds), Modifier.weight(1f))
                    SmallMetricCard("热量消耗", WorkoutInsights.formatCalories(metrics.calories), Modifier.weight(1f))
                }
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(22.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC))
                ) {
                    Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("姿态建议", fontWeight = FontWeight.SemiBold, color = Color(0xFF111827))
                        Text(
                            analysis?.postureAdvice?.ifBlank { metrics.advice } ?: metrics.advice,
                            color = Color(0xFF475569)
                        )
                        Text(
                            "最佳截图：${analysis?.bestShotLabel?.ifBlank { metrics.bestShotLabel } ?: metrics.bestShotLabel}",
                            color = Color(0xFF6D4CC3)
                        )
                    }
                }
            }
        }
    }

    @Composable
    private fun SmallMetricCard(label: String, value: String, modifier: Modifier = Modifier) {
        Card(
            modifier = modifier,
            shape = RoundedCornerShape(18.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC))
        ) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(label, color = Color(0xFF64748B), style = MaterialTheme.typography.bodySmall)
                Text(value, color = Color(0xFF111827), fontWeight = FontWeight.Bold)
            }
        }
    }

    @Composable
    private fun PreviewCard(title: String, bitmap: Bitmap?) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(title, fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                if (bitmap == null) {
                    PlaceholderBox("还没有内容", 120)
                } else {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = title,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(320.dp)
                            .background(Color(0xFFF8FAFC), RoundedCornerShape(18.dp)),
                        contentScale = ContentScale.Fit
                    )
                }
            }
        }
    }

    @Composable
    private fun PlaceholderBox(text: String, heightDp: Int) {
        Text(
            text,
            modifier = Modifier
                .fillMaxWidth()
                .height(heightDp.dp)
                .background(Color(0xFFF8FAFC), RoundedCornerShape(18.dp))
                .padding(24.dp),
            color = Color(0xFF64748B)
        )
    }

    @Composable
    private fun SampledFrameGallery(frames: List<DiagnosticFramePreview>) {
        if (frames.isEmpty()) return
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("采样帧诊断图", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    frames.forEach { frame ->
                        Column(
                            modifier = Modifier.width(280.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Image(
                                bitmap = frame.bitmap.asImageBitmap(),
                                contentDescription = "sample_${frame.timeMs}",
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(220.dp)
                                    .background(Color(0xFFF8FAFC), RoundedCornerShape(14.dp)),
                                contentScale = ContentScale.Crop
                            )
                            Text(
                                "${frame.timeMs}ms · score=${"%.2f".format(frame.score)}",
                                color = Color(0xFF475569),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun AnalysisCard(analysis: SportAnalysis?, savedUri: String?) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("结构化结果", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                if (analysis == null) {
                    Text("生成后会在这里展示运动总结、建议和原始 JSON。", color = Color(0xFF64748B))
                } else {
                    Text("标题：${analysis.summaryTitle.ifBlank { "运动战报" }}")
                    Text("动作：${sportTypeLabel(analysis.sportType)}")
                    Text("次数：${analysis.repetitionCount}")
                    Text("阶段：${analysis.stage.ifBlank { "-" }}")
                    Text("姿态：${analysis.poseQuality}")
                    Text("热量：${WorkoutInsights.formatCalories(analysis.calories)}")
                    Text("时长：${WorkoutInsights.formatDuration(analysis.durationMs)}")
                    Text("建议：${analysis.postureAdvice.ifBlank { analysis.riskTip.ifBlank { "-" } }}")
                    if (savedUri != null) {
                        Text("相册 URI：$savedUri", color = Color(0xFF0F766E))
                    }
                    Text("原始 JSON：${analysis.rawJson}", color = Color(0xFF64748B))
                }
            }
        }
    }

    private fun analysisModeLabel(mode: String): String = when (mode) {
        "squat" -> "深蹲"
        "push_up" -> "俯卧撑"
        else -> "自动"
    }

    private fun sportTypeLabel(type: String): String = when (type) {
        "squat" -> "深蹲"
        "push_up" -> "俯卧撑"
        "sit_up" -> "仰卧起坐"
        else -> "未知动作"
    }
    private fun buildSampledDiagnosticFrames(frames: List<SampledPoseFrame>): List<DiagnosticFramePreview> {
        return frames.map { frame ->
            DiagnosticFramePreview(
                timeMs = frame.timeMs,
                score = frame.poseEstimate.score,
                bitmap = PoseDebugRenderer.renderThumbnail(
                    source = frame.bitmap,
                    poseEstimate = frame.poseEstimate,
                    sourceLabel = "${frame.timeMs}ms",
                    withHeader = false
                )
            )
        }
    }
}
