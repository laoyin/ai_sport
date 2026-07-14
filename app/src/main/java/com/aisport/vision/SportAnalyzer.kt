package com.aisport.vision

import android.graphics.Bitmap
import android.util.Log
import com.aisport.engine.MnnEngine
import com.aisport.pose.PoseEstimate
import com.aisport.video.VideoMotionSummary
import com.aisport.workout.WorkoutInsights
import org.json.JSONObject
import java.io.File

data class SportAnalysis(
    val sportType: String = "",
    val summaryTitle: String = "",
    val repetitionCount: Int = 0,
    val stage: String = "",
    val poseQuality: String = "unknown",
    val highlight: String = "",
    val riskTip: String = "",
    val slogan: String = "",
    val confidence: Double = 0.0,
    val durationMs: Long = 0L,
    val avgRepSeconds: Float = 0f,
    val calories: Float = 0f,
    val bestShotLabel: String = "",
    val postureAdvice: String = "",
    val rawJson: String = ""
)

class SportAnalyzer(private val cacheDir: File, private val mnnEngine: MnnEngine) {

    companion object {
        private const val TAG = "SportAnalyzer"

        private const val SPORT_PROMPT = """
You are an AI sports poster assistant.
Analyze the single-person sports image and return JSON only.

Output format:
{
  "sport_type": "squat | jumping_jack | push_up | lunge | plank | running | yoga | unknown",
  "summary_title": "short Chinese title, under 12 chars",
  "repetition_count": 0,
  "stage": "one short phrase about current movement stage",
  "pose_quality": "excellent | good | fair | needs_attention | unknown",
  "highlight": "one short Chinese sentence about the strongest visible point",
  "risk_tip": "one short Chinese sentence about a visible risk or improvement suggestion",
  "slogan": "one energetic Chinese slogan for a shareable poster",
  "confidence": 0.0
}

Rules:
1. Return JSON only.
2. Focus only on what is visible in the image.
3. If repetition count cannot be inferred from a single image, return 1 when a clear action frame is visible, otherwise 0.
4. Do not invent medical diagnosis.
5. Prefer concise Chinese text.
"""
    }

    fun analyze(
        bitmap: Bitmap,
        poseEstimate: PoseEstimate? = null,
        motionSummary: VideoMotionSummary? = null
    ): SportAnalysis? {
        if (!mnnEngine.isReady()) {
            Log.w(TAG, "analyze skipped because MNN engine is not ready")
            return null
        }
        val imageFile = File(cacheDir, "sport_input_${System.currentTimeMillis()}.png")
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, imageFile.outputStream())
        val poseHint = poseEstimate?.let {
            """
            Additional pose context from local pipeline:
            - pose_engine: ${it.engineName}
            - keyframe_score: ${"%.3f".format(it.score)}
            - pose_quality_hint: ${it.qualityHint}
            - stage_hint: ${it.stageHint}
            - placeholder_pose: ${it.placeholder}
            """.trimIndent()
        }.orEmpty()
        val motionHint = motionSummary?.let {
            """
            Additional video motion summary from local multi-frame pipeline:
            - inferred_sport_type: ${it.inferredSportType}
            - inferred_repetition_count: ${it.repetitionCount}
            - inferred_stage_hint: ${it.stageHint}
            - inferred_confidence: ${"%.3f".format(it.confidence)}
            - inferred_debug: ${it.debug}

            Important rules for this request:
            - If local multi-frame pipeline says squat or push_up with confidence >= 0.50, do not output running.
            - Prefer local inferred repetition count over single-image guessing.
            """.trimIndent()
        }.orEmpty()
        val raw = mnnEngine.runInference(
            imageFile.absolutePath,
            (SPORT_PROMPT.trimIndent() + "\n\n" + poseHint + "\n\n" + motionHint).trim()
        ) ?: return null
        return mergeWithHints(parse(raw), poseEstimate, motionSummary)
    }

    private fun parse(raw: String): SportAnalysis {
        val start = raw.indexOf('{')
        val end = raw.lastIndexOf('}')
        if (start < 0 || end <= start) {
            return SportAnalysis(rawJson = raw)
        }
        return try {
            val json = raw.substring(start, end + 1)
            val obj = JSONObject(json)
            SportAnalysis(
                sportType = obj.optString("sport_type"),
                summaryTitle = obj.optString("summary_title"),
                repetitionCount = obj.optInt("repetition_count", 0),
                stage = obj.optString("stage"),
                poseQuality = obj.optString("pose_quality", "unknown"),
                highlight = obj.optString("highlight"),
                riskTip = obj.optString("risk_tip"),
                slogan = obj.optString("slogan"),
                confidence = obj.optDouble("confidence", 0.0),
                rawJson = json
            )
        } catch (t: Throwable) {
            Log.e(TAG, "parse failed", t)
            SportAnalysis(rawJson = raw)
        }
    }

    private fun mergeWithHints(
        analysis: SportAnalysis,
        poseEstimate: PoseEstimate?,
        motionSummary: VideoMotionSummary?
    ): SportAnalysis {
        if (motionSummary == null) {
            return analysis
        }
        val localSport = motionSummary.inferredSportType
        val localResolved = localSport in setOf("squat", "push_up") && motionSummary.confidence >= 0.5f
        if (!localResolved) {
            val metrics = WorkoutInsights.buildSummary(motionSummary, emptyList(), poseEstimate?.qualityHint, analysis.sportType)
            return if (analysis.poseQuality == "unknown" && poseEstimate != null) {
                analysis.copy(
                    poseQuality = poseEstimate.qualityHint,
                    durationMs = metrics.durationMs,
                    avgRepSeconds = metrics.averageRepSeconds,
                    calories = metrics.calories,
                    bestShotLabel = metrics.bestShotLabel,
                    postureAdvice = metrics.advice
                )
            } else {
                analysis.copy(
                    durationMs = metrics.durationMs,
                    avgRepSeconds = metrics.averageRepSeconds,
                    calories = metrics.calories,
                    bestShotLabel = metrics.bestShotLabel,
                    postureAdvice = metrics.advice
                )
            }
        }

        val titleBase = when (localSport) {
            "push_up" -> "俯卧撑"
            else -> "深蹲"
        }
        val title = when {
            analysis.summaryTitle.contains("深蹲") || analysis.summaryTitle.contains("俯卧撑") -> analysis.summaryTitle
            motionSummary.repetitionCount > 1 -> "${titleBase}${motionSummary.repetitionCount}次"
            else -> "${titleBase}动作"
        }
        val highlight = if (analysis.highlight.isBlank()) {
            "视频多帧姿态分析显示这是连续${titleBase}动作。"
        } else {
            analysis.highlight
        }
        val slogan = if (analysis.slogan.isBlank()) {
            if (localSport == "push_up") "稳住节奏，撑出力量。" else "稳住核心，蹲出力量。"
        } else {
            analysis.slogan
        }
        val metrics = WorkoutInsights.buildSummary(motionSummary, emptyList(), poseEstimate?.qualityHint, localSport)
        return analysis.copy(
            sportType = localSport,
            repetitionCount = motionSummary.repetitionCount,
            stage = motionSummary.stageHint.ifBlank { analysis.stage },
            poseQuality = poseEstimate?.qualityHint ?: analysis.poseQuality,
            summaryTitle = title,
            highlight = highlight,
            riskTip = if (analysis.riskTip.isBlank()) metrics.advice else analysis.riskTip,
            slogan = slogan,
            confidence = maxOf(analysis.confidence, motionSummary.confidence.toDouble()),
            durationMs = metrics.durationMs,
            avgRepSeconds = metrics.averageRepSeconds,
            calories = metrics.calories,
            bestShotLabel = metrics.bestShotLabel,
            postureAdvice = metrics.advice
        )
    }
}
