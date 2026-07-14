package com.aisport.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.aisport.exercise.ExerciseAnalyzer
import com.aisport.exercise.ExerciseFrameSample
import com.aisport.pose.PoseEstimate
import com.aisport.pose.PoseEstimator
import com.aisport.rep.NativeRepCounter
import com.aisport.rep.RepCounterInference
import kotlin.math.max

data class VideoKeyFrameResult(
    val bestFrame: Bitmap,
    val poseEstimate: PoseEstimate,
    val sampledFrames: Int,
    val bestFrameTimeMs: Long,
    val durationMs: Long,
    val motionSummary: VideoMotionSummary,
    val sampledPoseFrames: List<SampledPoseFrame>,
    val frameSamples: List<ExerciseFrameSample>
)

data class SampledPoseFrame(
    val timeMs: Long,
    val bitmap: Bitmap,
    val poseEstimate: PoseEstimate
)

data class VideoMotionSummary(
    val inferredSportType: String = "unknown",
    val repetitionCount: Int = 0,
    val stageHint: String = "",
    val confidence: Float = 0f,
    val debug: String = "",
    val sampleTimesMs: List<Long> = emptyList(),
    val squatAngles: List<Float> = emptyList(),
    val pushupAngles: List<Float> = emptyList(),
    val activeSignal: List<Float> = emptyList(),
    val representativeAngle: Float = 0f
)

object VideoFrameSampler {

    private const val TAG = "VideoFrameSampler"

    fun extractBestFrame(
        context: Context,
        uri: Uri,
        poseEstimator: PoseEstimator,
        repCounter: NativeRepCounter? = null,
        forcedSportType: String = "auto",
        maxFrames: Int = 120
    ): VideoKeyFrameResult? {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?: 0L
            val sampleCount = when {
                durationMs >= 20_000L -> minOf(maxFrames, 120)
                durationMs >= 12_000L -> minOf(maxFrames, 96)
                durationMs >= 8_000L -> minOf(maxFrames, 72)
                durationMs > 0L -> minOf(maxFrames, 60)
                else -> 20
            }.coerceAtLeast(8)

            val samples = mutableListOf<ExerciseFrameSample>()
            val sampledPoseFrames = mutableListOf<SampledPoseFrame>()
            var bestQuality = CandidateFrame()
            var bestSquat = CandidateFrame()
            var bestPushup = CandidateFrame()

            for (index in 0 until sampleCount) {
                val timeMs = if (durationMs <= 0L) {
                    index * 250L
                } else {
                    (durationMs * index) / max(1, sampleCount - 1)
                }
                val frame = retriever.getFrameAtTime(timeMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                    ?: continue
                val normalized = normalizeFrame(frame)
                val estimate = poseEstimator.estimate(normalized)
                sampledPoseFrames += SampledPoseFrame(
                    timeMs = timeMs,
                    bitmap = createPreviewBitmap(normalized),
                    poseEstimate = estimate
                )
                val sample = ExerciseAnalyzer.buildFrameSample(timeMs, estimate)
                samples += sample

                val qualityPriority = estimate.score * 1000f - timeMs / 1000f
                bestQuality = bestQuality.promoteIfBetter(qualityPriority, normalized, estimate, timeMs)

                val squatPriority = estimate.score * 100f + (180f - sample.squatAngle.safeAngle())
                bestSquat = bestSquat.promoteIfBetter(squatPriority, normalized, estimate, timeMs)

                val pushupPriority = estimate.score * 100f +
                    sample.torsoLinearity * 30f +
                    sample.torsoHorizontal * 40f +
                    (180f - sample.pushupAngle.safeAngle())
                bestPushup = bestPushup.promoteIfBetter(pushupPriority, normalized, estimate, timeMs)

                if (bestQuality.bitmap !== normalized && bestSquat.bitmap !== normalized && bestPushup.bitmap !== normalized) {
                    normalized.recycle()
                }
            }

            if (samples.isEmpty()) {
                return null
            }

            val motionSummary = RepCounterInference.inferMotionSummary(samples, repCounter, forcedSportType)
            val representative = when (motionSummary.inferredSportType) {
                "squat" -> if (bestSquat.bitmap != null) bestSquat else bestQuality
                "push_up" -> if (bestPushup.bitmap != null) bestPushup else bestQuality
                else -> bestQuality
            }

            val bestFrame = representative.bitmap ?: return null
            val bestEstimate = representative.estimate ?: return null
            VideoKeyFrameResult(
                bestFrame = bestFrame,
                poseEstimate = bestEstimate,
                sampledFrames = samples.size,
                bestFrameTimeMs = representative.timeMs,
                durationMs = durationMs,
                motionSummary = motionSummary,
                sampledPoseFrames = sampledPoseFrames,
                frameSamples = samples.toList()
            )
        } catch (t: Throwable) {
            Log.e(TAG, "extractBestFrame failed", t)
            null
        } finally {
            retriever.release()
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
            return CandidateFrame(newBitmap, newEstimate, newTimeMs, newPriority)
        }
    }

    private fun normalizeFrame(frame: Bitmap): Bitmap {
        val maxEdge = 1280
        val longEdge = max(frame.width, frame.height)
        if (longEdge <= maxEdge) {
            return frame
        }
        val scale = maxEdge.toFloat() / longEdge.toFloat()
        return Bitmap.createScaledBitmap(
            frame,
            (frame.width * scale).toInt(),
            (frame.height * scale).toInt(),
            true
        )
    }

    private fun createPreviewBitmap(frame: Bitmap): Bitmap {
        val maxEdge = 480
        val longEdge = max(frame.width, frame.height)
        if (longEdge <= maxEdge) {
            return frame.copy(Bitmap.Config.ARGB_8888, false)
        }
        val scale = maxEdge.toFloat() / longEdge.toFloat()
        return Bitmap.createScaledBitmap(
            frame,
            (frame.width * scale).toInt(),
            (frame.height * scale).toInt(),
            true
        )
    }

    private fun Float.safeAngle(): Float = if (isFinite()) coerceIn(0f, 180f) else 180f
}
