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
import androidx.compose.foundation.clickable
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
    val status: String = "Waiting for model and input",
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
    val sourceLabel: String = "Waiting for source",
    val selectedFrameKey: String? = null,
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
    val key: String,
    val timeMs: Long,
    val score: Float,
    val bitmap: Bitmap,
    val sourceBitmap: Bitmap,
    val poseEstimate: PoseEstimate,
    val sourceLabel: String
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
            uiState = uiState.copy(status = "Camera permission denied")
        }
    }

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) loadBitmap(uri)
    }

    private val takePicturePreviewLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) applySelectedBitmap(bitmap, "Camera photo")
    }

    private val pickVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) loadVideo(uri, "Album video")
    }

    private val captureVideoLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val uri = pendingVideoUri
        pendingVideoUri = null
        if (result.resultCode == RESULT_OK && uri != null) {
            loadVideo(uri, "Recorded video")
        } else {
            uiState = uiState.copy(status = "Video recording cancelled")
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
                        llmReady && poseReady && repReady -> "AI Sport is ready for live tracking and report generation"
                        llmReady && poseReady -> "Vision models are ready, rep model is missing"
                        else -> "Model loading failed, check assets files"
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
            sourceLabel = "Live camera",
            selectedFrameKey = null,
            status = "Live workout started"
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
                status = "Live workout ended with no valid result"
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
                                status = "Live tracking: ${sportTypeLabel(metrics.inferredSportType)} ${metrics.repetitionCount} reps"
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
        val candidatePreviews = buildSampledDiagnosticFrames(snapshot.candidateFrames, snapshot.sourceLabel)
        val defaultPreview = candidatePreviews.firstOrNull()
        val selectedBitmap = defaultPreview?.sourceBitmap ?: snapshot.bestFrame
        val selectedPoseEstimate = defaultPreview?.poseEstimate ?: snapshot.bestPoseEstimate
        val selectedLabel = defaultPreview?.sourceLabel ?: snapshot.sourceLabel
        val diagnosticBitmap = PoseDebugRenderer.render(
            selectedBitmap,
            selectedPoseEstimate,
            selectedLabel
        )
        uiState = uiState.copy(
            liveMode = false,
            liveRunning = false,
            selectedBitmap = selectedBitmap,
            diagnosticBitmap = diagnosticBitmap,
            sampledDiagnosticFrames = candidatePreviews,
            sampledPoseFrames = snapshot.candidateFrames,
            frameSamples = snapshot.frameSamples,
            timelineBitmap = WorkoutTimelineRenderer.render(snapshot.motionSummary),
            poseEstimate = selectedPoseEstimate,
            motionSummary = snapshot.motionSummary,
            posterBitmap = null,
            analysis = null,
            lastSavedUri = null,
            sourceLabel = selectedLabel,
            selectedFrameKey = defaultPreview?.key,
            status = "Live workout finished, generating report"
        )
        analyzeAndGeneratePoster()
    }

    private fun loadBitmap(uri: Uri) {
        contentResolver.openInputStream(uri)?.use { stream ->
            val bitmap = BitmapFactory.decodeStream(stream)
            if (bitmap != null) applySelectedBitmap(bitmap, "Album image")
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
            selectedFrameKey = null,
            status = "Pose analysis ready"
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
            selectedFrameKey = null,
            status = "Parsing video: ${analysisModeLabel(uiState.analysisMode)}"
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
                    val galleryBaseLabel = "$sourceLabel ? ${sportTypeLabel(result.motionSummary.inferredSportType)} ? ${result.motionSummary.repetitionCount} reps"
                    val candidatePreviews = buildSampledDiagnosticFrames(result.sampledPoseFrames, galleryBaseLabel)
                    val selectedPreview = candidatePreviews.firstOrNull { it.timeMs == result.bestFrameTimeMs } ?: candidatePreviews.firstOrNull()
                    val selectedBitmap = selectedPreview?.sourceBitmap ?: result.bestFrame
                    val selectedPoseEstimate = selectedPreview?.poseEstimate ?: result.poseEstimate
                    val selectedLabel = selectedPreview?.sourceLabel ?: "$sourceLabel ? best frame ${result.bestFrameTimeMs}ms / ${result.durationMs}ms"
                    val diagnosticBitmap = PoseDebugRenderer.render(
                        selectedBitmap,
                        selectedPoseEstimate,
                        selectedLabel
                    )
                    uiState = uiState.copy(
                        loading = false,
                        selectedBitmap = selectedBitmap,
                        diagnosticBitmap = diagnosticBitmap,
                        sampledDiagnosticFrames = candidatePreviews,
                        sampledPoseFrames = result.sampledPoseFrames,
                        frameSamples = result.frameSamples,
                        timelineBitmap = WorkoutTimelineRenderer.render(result.motionSummary),
                        poseEstimate = selectedPoseEstimate,
                        motionSummary = result.motionSummary,
                        sourceLabel = selectedLabel,
                        selectedFrameKey = selectedPreview?.key,
                        status = "Video analysis done: ${sportTypeLabel(result.motionSummary.inferredSportType)} ${result.motionSummary.repetitionCount} reps"
                    )
                } else {
                    uiState = uiState.copy(
                        loading = false,
                        status = "Video analysis failed"
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
            uiState = uiState.copy(status = "Opening system camera for recording")
            captureVideoLauncher.launch(intent)
        } catch (_: Throwable) {
            pendingVideoUri = null
            uiState = uiState.copy(status = "Failed to open system camera")
            Toast.makeText(this, "Failed to open system camera", Toast.LENGTH_SHORT).show()
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
        uiState = uiState.copy(loading = true, status = "AI is generating report and poster")
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
                        "AI report generated"
                    } else {
                        "Analysis done, poster generation failed"
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
            status = if (uri != null) "Poster saved to gallery" else "Failed to save poster"
        )
        Toast.makeText(this, uiState.status, Toast.LENGTH_SHORT).show()
    }

    private fun selectCandidateFrame(frame: DiagnosticFramePreview) {
        val diagnosticBitmap = PoseDebugRenderer.render(
            frame.sourceBitmap,
            frame.poseEstimate,
            frame.sourceLabel
        )
        uiState = uiState.copy(
            selectedBitmap = frame.sourceBitmap,
            diagnosticBitmap = diagnosticBitmap,
            poseEstimate = frame.poseEstimate,
            posterBitmap = null,
            analysis = null,
            lastSavedUri = null,
            sourceLabel = frame.sourceLabel,
            selectedFrameKey = frame.key,
            status = "Switched to frame ${frame.timeMs}ms, regenerating report"
        )
        analyzeAndGeneratePoster()
    }

    private fun exportAnnotationPackage() {
        if (uiState.sampledPoseFrames.isEmpty() || uiState.frameSamples.isEmpty()) {
            uiState = uiState.copy(status = "No annotation data to export")
            return
        }
        uiState = uiState.copy(
            loading = true,
            status = "Exporting annotation package with JSON and pose data"
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
                        status = "Annotation exported: ${result.frameCount} frames ? ${result.directory.absolutePath}"
                    )
                    Toast.makeText(this, "Annotation export success", Toast.LENGTH_SHORT).show()
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    uiState = uiState.copy(
                        loading = false,
                        status = "Annotation export failed: ${t.message ?: "unknown error"}"
                    )
                    Toast.makeText(this, "Annotation export failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun sharePosterImage() {
        val poster = uiState.posterBitmap ?: return
        val uri = PosterComposer.createShareUri(this, poster)
        if (uri == null) {
            uiState = uiState.copy(status = "Failed to create shareable poster file")
            Toast.makeText(this, "Failed to create shareable poster file", Toast.LENGTH_SHORT).show()
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
            uiState = uiState.copy(status = "WeChat share opened")
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
            startActivity(Intent.createChooser(genericShare, "Share workout report"))
            uiState = uiState.copy(status = "System share sheet opened")
        } catch (_: Throwable) {
            uiState = uiState.copy(status = "Poster share failed")
            Toast.makeText(this, "Poster share failed", Toast.LENGTH_SHORT).show()
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
                        PreviewCard("Source frame", uiState.selectedBitmap)
                        PreviewCard("Pose overlay", uiState.diagnosticBitmap)
                        PreviewCard("Motion timeline", uiState.timelineBitmap)
                        PreviewCard("Workout poster", uiState.posterBitmap)
                        SampledFrameGallery(uiState.sampledDiagnosticFrames, uiState.selectedFrameKey) { frame ->
                            selectCandidateFrame(frame)
                        }
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
                        if (state.liveRunning) "Live workout in progress with realtime rep counting" else "Start camera workout or import photo and video for AI report",
                        color = Color.White,
                        fontWeight = FontWeight.ExtraBold,
                        fontSize = 30.sp,
                        lineHeight = 34.sp
                    )
                    Text(
                        if (state.liveRunning) {
                            "${sportTypeLabel(state.liveSportType)} ? ${state.liveRepCount} reps ? ${WorkoutInsights.formatCalories(state.liveCalories)}"
                        } else {
                            "Realtime tracking, candidate frame switch, AI report and annotation export"
                        },
                        color = Color(0xFFDCE6E0),
                        fontSize = 15.sp
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        HeroBadge(if (state.modelReady) "Model ready" else "Model missing")
                        HeroBadge(if (state.liveRunning) "Live running" else "Idle")
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
                Text("Current Status", fontWeight = FontWeight.ExtraBold, color = Color(0xFF111827), fontSize = 22.sp)
                Text(state.status, color = Color(0xFF475569), lineHeight = 22.sp)
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                    SmallMetricCard("Model", if (state.modelReady) "Ready" else "Missing", Modifier.weight(1f))
                    SmallMetricCard("Mode", analysisModeLabel(state.analysisMode), Modifier.weight(1f))
                    SmallMetricCard("Report", if (state.posterBitmap != null) "Ready" else "Pending", Modifier.weight(1f))
                }
                Text("Source: ${state.sourceLabel}", color = Color(0xFF334155))
                state.poseEstimate?.let {
                    Text(
                        "Pose: ${it.engineName} ? score=${"%.2f".format(it.score)} ? ${it.qualityHint}",
                        color = Color(0xFF6D4CC3)
                    )
                }
                if (state.loading) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(10.dp))
                        Text("Processing frame and pose...", color = Color(0xFF475569))
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
                Text("Controls", fontWeight = FontWeight.ExtraBold, color = Color(0xFF111827), fontSize = 22.sp)
                Text(
                    "Live mode is for realtime demo impact. Photo and video modes are for analysis, annotation export and poster generation. Candidate frames can be switched anytime.",
                    color = Color(0xFF64748B)
                )

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Analysis Mode", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        ModeChip("Auto", uiState.analysisMode == "auto", Modifier.weight(1f)) {
                            uiState = uiState.copy(analysisMode = "auto", status = "Switched to auto recognition mode")
                        }
                        ModeChip("Squat", uiState.analysisMode == "squat", Modifier.weight(1f)) {
                            uiState = uiState.copy(analysisMode = "squat", status = "Switched to squat mode")
                        }
                        ModeChip("Push-up", uiState.analysisMode == "push_up", Modifier.weight(1f)) {
                            uiState = uiState.copy(analysisMode = "push_up", status = "Switched to push-up mode")
                        }
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Live Workout", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        PrimaryActionButton(
                            label = "Start Live",
                            enabled = uiState.modelReady && !uiState.liveRunning,
                            modifier = Modifier.weight(1f),
                            onClick = { ensureCameraPermissionAndStart() }
                        )
                        SecondaryActionButton(
                            label = "Stop",
                            enabled = uiState.liveRunning,
                            modifier = Modifier.weight(1f),
                            onClick = { stopLiveWorkoutMode() }
                        )
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Input", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        PrimaryActionButton("Choose Image", true, Modifier.weight(1f)) {
                            pickImageLauncher.launch("image/*")
                        }
                        PrimaryActionButton("Take Photo", true, Modifier.weight(1f)) {
                            takePicturePreviewLauncher.launch(null)
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        SecondaryActionButton("Choose Video", true, Modifier.weight(1f)) {
                            pickVideoLauncher.launch("video/*")
                        }
                        SecondaryActionButton("Record Video", true, Modifier.weight(1f)) {
                            captureVideo()
                        }
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Output", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        PrimaryActionButton(
                            label = "Generate Report",
                            enabled = uiState.modelReady && uiState.selectedBitmap != null && !uiState.loading,
                            modifier = Modifier.weight(1f),
                            onClick = { analyzeAndGeneratePoster() }
                        )
                        SecondaryActionButton(
                            label = "Save Poster",
                            enabled = uiState.posterBitmap != null && !uiState.loading,
                            modifier = Modifier.weight(1f),
                            onClick = { savePoster() }
                        )
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                        SecondaryActionButton(
                            label = "Export Labels",
                            enabled = uiState.sampledPoseFrames.isNotEmpty() && !uiState.loading,
                            modifier = Modifier.weight(1f),
                            onClick = { exportAnnotationPackage() }
                        )
                        SecondaryActionButton(
                            label = "Share Poster",
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
                Text("Live Workout", fontWeight = FontWeight.ExtraBold, color = Color(0xFF111827), fontSize = 24.sp)
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
                            Text("Reps", color = Color.White.copy(alpha = 0.78f), fontSize = 14.sp)
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
                            Text("Duration  ${WorkoutInsights.formatDuration(state.liveElapsedMs)}", color = Color(0xFF0F172A))
                            Text("Calories  ${WorkoutInsights.formatCalories(state.liveCalories)}", color = Color(0xFFB45309))
                            Text("Pace  ${WorkoutInsights.formatPace(state.liveAverageRepSeconds)}", color = Color(0xFF0F766E))
                            Text("Confidence  ${"%.2f".format(state.liveConfidence)}", color = Color(0xFF475569))
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
                            Text("Pose Skeleton", fontWeight = FontWeight.SemiBold, color = Color(0xFF111827))
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
                                PlaceholderBox("Skeleton preview will appear after live camera starts.", 180)
                            }
                        }
                    }
                    Card(
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(22.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC))
                    ) {
                        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("Motion Curve", fontWeight = FontWeight.SemiBold, color = Color(0xFF111827))
                            LiveWorkoutChart(values = state.liveWaveform)
                            Text(
                                state.liveDebug.ifBlank { "Curve updates in realtime to show rhythm and motion quality." },
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
                Text("AI Workout Report", fontWeight = FontWeight.ExtraBold, fontSize = 24.sp, color = Color(0xFF111827))
                Text(
                    analysis?.summaryTitle?.ifBlank { "This session has been analyzed. Review reps, duration, pace, calories and posture quality below." }
                        ?: "This session has been analyzed. Review reps, duration, pace, calories and posture quality below.",
                    color = Color(0xFF475569)
                )
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    SmallMetricCard("Reps", (motionSummary?.repetitionCount ?: 0).toString(), Modifier.weight(1f))
                    SmallMetricCard("Duration", WorkoutInsights.formatDuration(metrics.durationMs), Modifier.weight(1f))
                    SmallMetricCard("Pace", WorkoutInsights.formatPace(metrics.averageRepSeconds), Modifier.weight(1f))
                    SmallMetricCard("Calories", WorkoutInsights.formatCalories(metrics.calories), Modifier.weight(1f))
                }
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(22.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FAFC))
                ) {
                    Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Posture Advice", fontWeight = FontWeight.SemiBold, color = Color(0xFF111827))
                        Text(
                            analysis?.postureAdvice?.ifBlank { metrics.advice } ?: metrics.advice,
                            color = Color(0xFF475569)
                        )
                        Text(
                            "Best shot: ${analysis?.bestShotLabel?.ifBlank { metrics.bestShotLabel } ?: metrics.bestShotLabel}",
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
                    PlaceholderBox("No image available yet.", 120)
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
    private fun SampledFrameGallery(
        frames: List<DiagnosticFramePreview>,
        selectedFrameKey: String?,
        onSelect: (DiagnosticFramePreview) -> Unit
    ) {
        if (frames.isEmpty()) return
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Candidate Frames", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                Text(
                    "Tap a frame to switch the main shot and regenerate the report.",
                    color = Color(0xFF64748B),
                    style = MaterialTheme.typography.bodySmall
                )
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    frames.forEach { frame ->
                        val selected = frame.key == selectedFrameKey
                        Card(
                            modifier = Modifier
                                .width(280.dp)
                                .clickable { onSelect(frame) },
                            shape = RoundedCornerShape(16.dp),
                            colors = CardDefaults.cardColors(
                                containerColor = if (selected) Color(0xFFEDE9FE) else Color(0xFFF8FAFC)
                            )
                        ) {
                            Column(
                                modifier = Modifier.padding(8.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Image(
                                    bitmap = frame.bitmap.asImageBitmap(),
                                    contentDescription = "sample_${frame.timeMs}",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(220.dp)
                                        .background(Color.White, RoundedCornerShape(12.dp)),
                                    contentScale = ContentScale.Crop
                                )
                                Text(
                                    if (selected) {
                                        "Selected ? ${frame.timeMs}ms ? score=${"%.2f".format(frame.score)}"
                                    } else {
                                        "${frame.timeMs}ms ? score=${"%.2f".format(frame.score)}"
                                    },
                                    color = if (selected) Color(0xFF6D4CC3) else Color(0xFF475569),
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
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
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Structured Result", fontWeight = FontWeight.Bold, color = Color(0xFF111827))
                if (analysis == null) {
                    Text("The generated summary, advice and raw JSON will appear here.", color = Color(0xFF64748B))
                } else {
                    Text("Title: ${analysis.summaryTitle.ifBlank { "Workout Report" }}")
                    Text("Sport: ${sportTypeLabel(analysis.sportType)}")
                    Text("Reps: ${analysis.repetitionCount}")
                    Text("Stage: ${analysis.stage.ifBlank { "-" }}")
                    Text("Pose: ${analysis.poseQuality}")
                    Text("Calories: ${WorkoutInsights.formatCalories(analysis.calories)}")
                    Text("Duration: ${WorkoutInsights.formatDuration(analysis.durationMs)}")
                    Text("Advice: ${analysis.postureAdvice.ifBlank { analysis.riskTip.ifBlank { "-" } }}")
                    if (savedUri != null) {
                        Text("Gallery URI: $savedUri", color = Color(0xFF0F766E))
                    }
                    Text("Raw JSON: ${analysis.rawJson}", color = Color(0xFF64748B))
                }
            }
        }
    }

    private fun analysisModeLabel(mode: String): String = when (mode) {
        "squat" -> "Squat"
        "push_up" -> "Push-up"
        else -> "Auto"
    }

    private fun sportTypeLabel(type: String): String = when (type) {
        "squat" -> "Squat"
        "push_up" -> "Push-up"
        "sit_up" -> "Sit-up"
        else -> "Unknown"
    }

    private fun buildSampledDiagnosticFrames(
        frames: List<SampledPoseFrame>,
        baseLabel: String
    ): List<DiagnosticFramePreview> {
        return frames.map { frame ->
            DiagnosticFramePreview(
                key = buildFrameKey(frame.timeMs),
                timeMs = frame.timeMs,
                score = frame.poseEstimate.score,
                bitmap = PoseDebugRenderer.renderThumbnail(
                    source = frame.bitmap,
                    poseEstimate = frame.poseEstimate,
                    sourceLabel = "${frame.timeMs}ms",
                    withHeader = false
                ),
                sourceBitmap = frame.bitmap,
                poseEstimate = frame.poseEstimate,
                sourceLabel = "$baseLabel ? ${frame.timeMs}ms"
            )
        }
    }

    private fun buildFrameKey(timeMs: Long): String = "frame_$timeMs"
}
