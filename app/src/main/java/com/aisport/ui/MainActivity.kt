package com.aisport.ui

import android.content.ClipData
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.aisport.engine.MnnEngine
import com.aisport.exercise.ExerciseFrameSample
import com.aisport.poster.PosterComposer
import com.aisport.poster.PoseDebugRenderer
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
import androidx.core.content.FileProvider
import java.io.File
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
    val sourceLabel: String = "未选择输入"
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
    private var pendingVideoUri: Uri? = null

    private var uiState by mutableStateOf(SportUiState())

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            loadBitmap(uri)
        }
    }

    private val takePicturePreviewLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) {
            applySelectedBitmap(bitmap, "相机拍照输入")
        }
    }

    private val pickVideoLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            loadVideo(uri, "相册视频")
        }
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
        mnnEngine.release()
        poseEstimator.release()
        repCounter.release()
        super.onDestroy()
    }

    private fun bootstrapModel() {
        thread {
            val ready = mnnEngine.loadModelBundle("qwen3-vl-mnn")
            val poseReady = poseEstimator.loadModel()
            val repReady = repCounter.loadModel()
            runOnUiThread {
                uiState = uiState.copy(
                    modelReady = ready && poseReady,
                    status = if (ready && poseReady && repReady) {
                        "大模型、Pose 模型和时序计数模型都已就绪，可以开始分析。"
                    } else if (ready && poseReady) {
                        "大模型和 Pose 已就绪，但时序计数模型加载失败，当前仍会回退到规则计数。"
                    } else {
                        "模型加载失败，请先确认 qwen3-vl-mnn 资源已打包。"
                    }
                )
            }
        }
    }

    private fun loadBitmap(uri: Uri) {
        contentResolver.openInputStream(uri)?.use { stream ->
            val bitmap = BitmapFactory.decodeStream(stream)
            if (bitmap != null) {
                applySelectedBitmap(bitmap, "相册图片输入")
            }
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
            status = "YOLO-Pose 已完成关键点检测。当前是单图模式，可继续调用 VL 生成总结和海报。"
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
            status = "正在对视频连续抽帧，并由本地动作分析器统计类别和次数… 当前模式：${analysisModeLabel(uiState.analysisMode)}"
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
                    val keyframeLabel = "$sourceLabel · 代表帧 ${result.bestFrameTimeMs}ms / ${result.durationMs}ms · 监测 ${result.sampledFrames} 帧"
                    val diagnosticBitmap = PoseDebugRenderer.render(
                        result.bestFrame,
                        result.poseEstimate,
                        keyframeLabel
                    )
                    val sampledDiagnosticFrames = buildSampledDiagnosticFrames(result.sampledPoseFrames)
                    val timelineBitmap = WorkoutTimelineRenderer.render(result.motionSummary)
                    uiState = uiState.copy(
                        loading = false,
                        selectedBitmap = result.bestFrame,
                        diagnosticBitmap = diagnosticBitmap,
                        sampledDiagnosticFrames = sampledDiagnosticFrames,
                        sampledPoseFrames = result.sampledPoseFrames,
                        frameSamples = result.frameSamples,
                        timelineBitmap = timelineBitmap,
                        poseEstimate = result.poseEstimate,
                        motionSummary = result.motionSummary,
                        sourceLabel = keyframeLabel,
                        status = buildString {
                            append("视频多帧监测完成，本地动作分析器已输出结果。")
                            val sportLabel = when (result.motionSummary.inferredSportType) {
                                "squat" -> "深蹲"
                                "push_up" -> "俯卧撑"
                                else -> "未知动作"
                            }
                            append(" 本地判断：$sportLabel")
                            if (result.motionSummary.repetitionCount > 0) {
                                append(" ${result.motionSummary.repetitionCount} 次")
                            }
                            append("。")
                        }
                    )
                } else {
                    uiState = uiState.copy(
                        loading = false,
                        status = "视频关键帧抽取失败，请换一段更清晰的运动视频。"
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
            uiState = uiState.copy(status = "正在打开系统相机进行视频录制…")
            captureVideoLauncher.launch(intent)
        } catch (t: Throwable) {
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
        uiState = uiState.copy(loading = true, status = "正在基于 YOLO 诊断图调用 VL 总结并生成海报…")
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
                        "运动宣传卡已生成，可以直接保存。"
                    } else {
                        "分析失败，请换一张更清晰的运动图片再试。"
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
            uiState = uiState.copy(status = "当前没有可导出的标注数据，请先选择并分析一段视频。")
            return
        }
        uiState = uiState.copy(
            loading = true,
            status = "正在导出标注数据包，包含时序 JSON 和 YOLO Pose 图片…"
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

    private fun sharePosterToWeChat() {
        val poster = uiState.posterBitmap ?: return
        val uri = PosterComposer.createShareUri(this, poster)
        if (uri == null) {
            uiState = uiState.copy(status = "海报导出失败，请重新生成后再试")
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
            uiState = uiState.copy(status = "已拉起微信，可继续分享到好友或朋友圈")
            return
        } catch (_: Throwable) {
        }

        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "image/*"
            putExtra(Intent.EXTRA_STREAM, uri)
            clipData = ClipData.newUri(contentResolver, "ai_sport_poster", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        grantShareUriPermission(uri, shareIntent)
        val chooserIntent = Intent.createChooser(shareIntent, "分享运动宣传卡")
        try {
            startActivity(chooserIntent)
            uiState = uiState.copy(status = "已拉起系统分享面板，可选择微信或其他应用")
        } catch (_: Throwable) {
            uiState = uiState.copy(status = "当前设备没有可用的分享应用")
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

    private fun sharePosterImage() {
        val poster = uiState.posterBitmap ?: return
        val uri = PosterComposer.createShareUri(this, poster)
        if (uri == null) {
            uiState = uiState.copy(status = "海报导出失败，请重新生成后再试")
            Toast.makeText(this, "海报导出失败", Toast.LENGTH_SHORT).show()
            return
        }

        val shareToWeChat = Intent(Intent.ACTION_SEND).apply {
            type = "image/*"
            putExtra(Intent.EXTRA_STREAM, uri)
            clipData = ClipData.newUri(contentResolver, "ai_sport_poster", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            setPackage("com.tencent.mm")
        }
        grantShareUriPermission(uri, shareToWeChat)
        try {
            startActivity(shareToWeChat)
            uiState = uiState.copy(status = "已拉起微信，请继续分享到好友或朋友圈")
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
            startActivity(Intent.createChooser(genericShare, "分享运动宣传卡"))
            uiState = uiState.copy(status = "已拉起系统分享面板，可选择微信或其他应用")
        } catch (_: Throwable) {
            uiState = uiState.copy(status = "当前设备没有可用的分享应用")
            Toast.makeText(this, "无法拉起分享", Toast.LENGTH_SHORT).show()
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun AiSportScreen() {
        MaterialTheme {
            Scaffold(
                topBar = {
                    TopAppBar(
                        title = { Text("AI Sport") },
                        colors = TopAppBarDefaults.topAppBarColors(
                            containerColor = Color(0xFFF3EEE3),
                            titleContentColor = Color(0xFF0F172A)
                        )
                    )
                }
            ) { paddingValues ->
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues)
                        .padding(16.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    StatusCard(uiState)
                    ActionArea()
                    PreviewCard("代表帧", uiState.selectedBitmap)
                    PreviewCard("YOLO 诊断图", uiState.diagnosticBitmap)
                    SampledFrameGallery(uiState.sampledDiagnosticFrames)
                    PreviewCard("视频监测曲线", uiState.timelineBitmap)
                    PreviewCard("生成的宣传卡", uiState.posterBitmap)
                    AnalysisCard(uiState.analysis, uiState.lastSavedUri)
                }
            }
        }
    }

    @Composable
    private fun StatusCard(state: SportUiState) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("当前状态", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
                Text(state.status, color = Color(0xFF475569))
                Text(
                    "模型：${if (state.modelReady) "已就绪" else "未就绪"}  ·  海报生成：${if (state.posterBitmap != null) "已完成" else "待生成"}",
                    color = Color(0xFF0F766E)
                )
                Text("输入：${state.sourceLabel}", color = Color(0xFF334155))
                state.poseEstimate?.let {
                    Text(
                        "Pose：${it.engineName} · score=${"%.2f".format(it.score)} · ${it.qualityHint} · ${if (it.placeholder) "占位接口" else "真实模型"}",
                        color = Color(0xFF7C3AED)
                    )
                }
                Text("分析模式：${analysisModeLabel(state.analysisMode)}", color = Color(0xFF0369A1))
                state.motionSummary?.let {
                    val sportLabel = when (it.inferredSportType) {
                        "squat" -> "深蹲"
                        "push_up" -> "俯卧撑"
                        else -> "unknown"
                    }
                    Text(
                        "视频动作：$sportLabel · 次数=${it.repetitionCount} · conf=${"%.2f".format(it.confidence)}",
                        color = Color(0xFFB45309)
                    )
                    Text("监测说明：${it.debug}", color = Color(0xFF78716C))
                }
                if (state.loading) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.size(10.dp))
                        Text("正在处理 YOLO / Qwen3-VL-MNN")
                    }
                }
            }
        }
    }

    @Composable
    private fun ActionArea() {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            OutlinedButton(
                onClick = { uiState = uiState.copy(analysisMode = "auto", status = "已切换到自动识别模式。") },
                modifier = Modifier.weight(1f)
            ) {
                Text("自动识别")
            }
            OutlinedButton(
                onClick = { uiState = uiState.copy(analysisMode = "squat", status = "已切换到深蹲计数模式。") },
                modifier = Modifier.weight(1f)
            ) {
                Text("深蹲模式")
            }
            OutlinedButton(
                onClick = { uiState = uiState.copy(analysisMode = "push_up", status = "已切换到俯卧撑计数模式。") },
                modifier = Modifier.weight(1f)
            ) {
                Text("俯卧撑模式")
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = { pickImageLauncher.launch("image/*") },
                modifier = Modifier.weight(1f)
            ) {
                Text("选择运动图")
            }
            Button(
                onClick = { takePicturePreviewLauncher.launch(null) },
                modifier = Modifier.weight(1f)
            ) {
                Text("相机拍照")
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            OutlinedButton(
                onClick = { pickVideoLauncher.launch("video/*") },
                modifier = Modifier.weight(1f)
            ) {
                Text("选择视频")
            }
            OutlinedButton(
                onClick = { captureVideo() },
                modifier = Modifier.weight(1f)
            ) {
                Text("录制视频")
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = { analyzeAndGeneratePoster() },
                enabled = uiState.modelReady && uiState.selectedBitmap != null && !uiState.loading,
                modifier = Modifier.weight(1f)
            ) {
                Text("生成宣传卡")
            }
            OutlinedButton(
                onClick = { savePoster() },
                enabled = uiState.posterBitmap != null && !uiState.loading,
                modifier = Modifier.weight(1f)
            ) {
                Text("保存到相册")
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            OutlinedButton(
                onClick = { exportAnnotationPackage() },
                enabled = uiState.sampledPoseFrames.isNotEmpty() && !uiState.loading,
                modifier = Modifier.weight(1f)
            ) {
                Text("导出标注数据")
            }
            OutlinedButton(
                onClick = { sharePosterImage() },
                enabled = uiState.posterBitmap != null && !uiState.loading,
                modifier = Modifier.weight(1f)
            ) {
                Text("分享微信 / 朋友圈")
            }
        }
    }

    @Composable
    private fun PreviewCard(title: String, bitmap: Bitmap?) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(title, fontWeight = FontWeight.Bold)
                if (bitmap == null) {
                    Text(
                        "还没有内容",
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFFF8FAFC), RoundedCornerShape(18.dp))
                            .padding(24.dp),
                        color = Color(0xFF64748B)
                    )
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
    private fun SampledFrameGallery(frames: List<DiagnosticFramePreview>) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("采样帧诊断图", fontWeight = FontWeight.Bold)
                if (frames.isEmpty()) {
                    Text("视频模式下会在这里横向展示所有采样帧的 YOLO 诊断结果。", color = Color(0xFF64748B))
                } else {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        frames.forEach { frame ->
                            Column(
                                modifier = Modifier.width(320.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Image(
                                    bitmap = frame.bitmap.asImageBitmap(),
                                    contentDescription = "sample_${frame.timeMs}",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(260.dp)
                                        .background(Color(0xFFF8FAFC), RoundedCornerShape(14.dp)),
                                    contentScale = ContentScale.Fit
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
    }

    @Composable
    private fun AnalysisCard(analysis: SportAnalysis?, savedUri: String?) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("结构化结果", fontWeight = FontWeight.Bold)
                if (analysis == null) {
                    Text("生成后会在这里展示运动总结、建议和原始 JSON。", color = Color(0xFF64748B))
                } else {
                    Text("标题：${analysis.summaryTitle.ifBlank { "运动战报" }}")
                    Text("动作：${analysis.sportType.ifBlank { "unknown" }}")
                    Text("次数：${analysis.repetitionCount}")
                    Text("阶段：${analysis.stage.ifBlank { "-" }}")
                    Text("姿态：${analysis.poseQuality}")
                    Text("亮点：${analysis.highlight.ifBlank { "-" }}")
                    Text("建议：${analysis.riskTip.ifBlank { "-" }}")
                    Text("标语：${analysis.slogan.ifBlank { "-" }}")
                    uiState.poseEstimate?.let {
                        Text("关键帧评分：${"%.2f".format(it.score)} · 阶段：${it.stageHint}")
                    }
                    uiState.motionSummary?.let {
                        val sportLabel = when (it.inferredSportType) {
                            "squat" -> "深蹲"
                            "push_up" -> "俯卧撑"
                            else -> it.inferredSportType
                        }
                        Text(
                            "视频推断：$sportLabel · 次数=${it.repetitionCount} · ${it.stageHint}",
                            color = Color(0xFF92400E)
                        )
                    }
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
