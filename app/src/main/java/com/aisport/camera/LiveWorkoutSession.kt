package com.aisport.camera

import android.graphics.Bitmap
import com.aisport.exercise.ExerciseAnalyzer
import com.aisport.exercise.ExerciseFrameSample
import com.aisport.poster.PoseDebugRenderer
import com.aisport.pose.PoseEstimate
import com.aisport.rep.NativeRepCounter
import com.aisport.rep.RepCounterInference
import com.aisport.video.SampledPoseFrame
import com.aisport.video.VideoMotionSummary
import com.aisport.workout.WorkoutInsights
import kotlin.math.max

data class LiveWorkoutMetrics(
    val repetitionCount: Int,
    val confidence: Float,
    val inferredSportType: String,
    val waveform: List<Float>,
    val elapsedMs: Long,
    val poseEstimate: PoseEstimate,
    val debug: String,
    val calories: Float,
    val averageRepSeconds: Float,
    val overlayBitmap: Bitmap
)

data class LiveWorkoutSnapshot(
    val bestFrame: Bitmap,
    val bestPoseEstimate: PoseEstimate,
    val motionSummary: VideoMotionSummary,
    val frameSamples: List<ExerciseFrameSample>,
    val candidateFrames: List<SampledPoseFrame>,
    val sourceLabel: String
)

class LiveWorkoutSession(
    private val poseEstimator: (Bitmap) -> PoseEstimate,
    private val repCounter: NativeRepCounter?,
    private val forcedSportType: String = "auto"
) {

    companion object {
        private const val MIN_SAMPLE_INTERVAL_MS = 180L
    }

    private val samples = mutableListOf<ExerciseFrameSample>()
    private var startTimestampMs = 0L
    private var lastSampleTimestampMs = 0L
    private var bestQuality = CandidateFrame()
    private var bestPushup = CandidateFrame()
    private var bestSquat = CandidateFrame()
    private var latestSummary = VideoMotionSummary()

    fun processFrame(frame: Bitmap, timestampMs: Long): LiveWorkoutMetrics? {
        if (startTimestampMs == 0L) {
            startTimestampMs = timestampMs
        }
        if (lastSampleTimestampMs != 0L && timestampMs - lastSampleTimestampMs < MIN_SAMPLE_INTERVAL_MS) {
            return null
        }
        lastSampleTimestampMs = timestampMs
        val estimate = poseEstimator(frame)
        val elapsedMs = max(0L, timestampMs - startTimestampMs)
        val sample = ExerciseAnalyzer.buildFrameSample(elapsedMs, estimate)
        samples += sample

        val qualityPriority = estimate.score * 1000f - elapsedMs / 1000f
        bestQuality = bestQuality.promoteIfBetter(qualityPriority, frame, estimate, elapsedMs)

        val squatPriority = estimate.score * 100f + (180f - sample.squatAngle.safeAngle())
        bestSquat = bestSquat.promoteIfBetter(squatPriority, frame, estimate, elapsedMs)

        val pushupPriority = estimate.score * 100f +
            sample.torsoLinearity * 30f +
            sample.torsoHorizontal * 40f +
            (180f - sample.pushupAngle.safeAngle())
        bestPushup = bestPushup.promoteIfBetter(pushupPriority, frame, estimate, elapsedMs)

        latestSummary = RepCounterInference.inferMotionSummary(samples, repCounter, forcedSportType)
        val metrics = WorkoutInsights.buildSummary(latestSummary, samples, estimate.qualityHint)
        return LiveWorkoutMetrics(
            repetitionCount = latestSummary.repetitionCount,
            confidence = latestSummary.confidence,
            inferredSportType = latestSummary.inferredSportType,
            waveform = latestSummary.activeSignal.takeLast(64),
            elapsedMs = elapsedMs,
            poseEstimate = estimate,
            debug = latestSummary.debug,
            calories = metrics.calories,
            averageRepSeconds = metrics.averageRepSeconds,
            overlayBitmap = PoseDebugRenderer.renderThumbnail(
                source = frame,
                poseEstimate = estimate,
                sourceLabel = "${latestSummary.repetitionCount} reps",
                maxWidth = 540,
                withHeader = false
            )
        )
    }

    fun finish(): LiveWorkoutSnapshot? {
        if (samples.isEmpty()) {
            return null
        }
        latestSummary = RepCounterInference.inferMotionSummary(samples, repCounter, forcedSportType)
        val representative = when (latestSummary.inferredSportType) {
            "squat" -> bestSquat.takeIf { it.bitmap != null } ?: bestQuality
            "push_up" -> bestPushup.takeIf { it.bitmap != null } ?: bestQuality
            else -> bestQuality
        }
        val frame = representative.bitmap ?: return null
        val estimate = representative.estimate ?: return null
        val candidateFrames = buildCandidateFrames(representative)
        return LiveWorkoutSnapshot(
            bestFrame = frame,
            bestPoseEstimate = estimate,
            motionSummary = latestSummary,
            frameSamples = samples.toList(),
            candidateFrames = candidateFrames,
            sourceLabel = "实时运动 ${latestSummary.inferredSportType} ${latestSummary.repetitionCount}次"
        )
    }

    private fun buildCandidateFrames(representative: CandidateFrame): List<SampledPoseFrame> {
        val ordered = buildList {
            add(representative)
            add(bestQuality)
            add(bestPushup)
            add(bestSquat)
        }
        val seen = linkedSetOf<String>()
        return ordered.mapNotNull { candidate ->
            val bitmap = candidate.bitmap ?: return@mapNotNull null
            val estimate = candidate.estimate ?: return@mapNotNull null
            val key = "${candidate.timeMs}_${estimate.score}"
            if (!seen.add(key)) return@mapNotNull null
            SampledPoseFrame(
                timeMs = candidate.timeMs,
                bitmap = bitmap,
                poseEstimate = estimate
            )
        }
    }

    private data class CandidateFrame(
        val bitmap: Bitmap? = null,
        val estimate: PoseEstimate? = null,
        val timeMs: Long = 0L,
        val priority: Float = Float.NEGATIVE_INFINITY
    ) {
        fun promoteIfBetter(newPriority: Float, newBitmap: Bitmap, newEstimate: PoseEstimate, newTimeMs: Long): CandidateFrame {
            if (newPriority <= priority) return this
            return CandidateFrame(newBitmap.copy(newBitmap.config ?: Bitmap.Config.ARGB_8888, false), newEstimate, newTimeMs, newPriority)
        }
    }

    private fun Float.safeAngle(): Float = if (isFinite()) coerceIn(0f, 180f) else 180f
}
