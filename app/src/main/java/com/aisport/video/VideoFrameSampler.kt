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
import com.aisport.pose.PosePoint
import com.aisport.rep.NativeRepCounter
import kotlin.math.max
import kotlin.math.sqrt

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

            val ruleSummary = ExerciseAnalyzer.analyze(samples, forcedSportType)
            val modelSummary = repCounter?.let { inferWithRepCounter(samples, it, forcedSportType) }
            val motionSummary = fuseMotionSummary(ruleSummary, modelSummary, forcedSportType)
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

    private data class ModelMotionSummary(
        val summary: VideoMotionSummary
    )

    private fun inferWithRepCounter(
        samples: List<ExerciseFrameSample>,
        repCounter: NativeRepCounter,
        forcedSportType: String
    ): ModelMotionSummary? {
        if (samples.size < NativeRepCounter.SEQ_LEN) {
            return null
        }

        val featureCache = samples.mapIndexed { index, sample ->
            buildFrameFeature(sample, samples.getOrNull(index - 1))
        }
        val stageVotes = List(samples.size) { IntArray(4) }
        val actionVotes = IntArray(3)
        var confidenceSum = 0f
        var windows = 0

        for (start in 0..(samples.size - NativeRepCounter.SEQ_LEN)) {
            val flat = FloatArray(NativeRepCounter.SEQ_LEN * NativeRepCounter.FEATURE_DIM)
            var cursor = 0
            for (offset in 0 until NativeRepCounter.SEQ_LEN) {
                val feature = featureCache[start + offset]
                for (value in feature) {
                    flat[cursor++] = value
                }
            }
            val result = repCounter.inferWindow(flat) ?: continue
            val actionId = result.actionId().coerceIn(0, 2)
            actionVotes[actionId] += 1
            confidenceSum += softmaxMax(result.actionLogits)
            windows += 1

            val stageIds = result.stageIds()
            stageIds.forEachIndexed { offset, stageId ->
                val frameIndex = start + offset
                stageVotes[frameIndex][stageId.coerceIn(0, 3)] += 1
            }
        }

        if (windows == 0) {
            return null
        }

        val predActionId = argmax(actionVotes)
        val predAction = when (predActionId) {
            1 -> "push_up"
            2 -> "sit_up"
            else -> "background"
        }
        val predStages = stageVotes.map { votes ->
            when (argmax(votes)) {
                1 -> "up"
                2 -> "down"
                3 -> "transition"
                else -> "background"
            }
        }
        val predictedCount = if (predAction == "push_up") countFromStageLabels(predStages) else 0
        val stageHint = when (predStages.lastOrNull()) {
            "up" -> "MNN判断当前接近起始/撑起位置"
            "down" -> "MNN判断当前处于下压最低位"
            "transition" -> "MNN判断当前处于动作过渡阶段"
            else -> "MNN已完成整段时序推理"
        }
        val avgConfidence = (confidenceSum / windows).coerceIn(0f, 1f)
        val inferredSportType = when {
            forcedSportType == "push_up" -> "push_up"
            forcedSportType == "squat" -> "squat"
            predAction == "push_up" -> "push_up"
            predAction == "sit_up" -> "sit_up"
            else -> "unknown"
        }
        val debug = "mnn_action=$predAction windows=$windows count=$predictedCount avg_conf=${"%.2f".format(avgConfidence)}"

        return ModelMotionSummary(
            summary = VideoMotionSummary(
                inferredSportType = inferredSportType,
                repetitionCount = predictedCount,
                stageHint = stageHint,
                confidence = avgConfidence,
                debug = debug,
                sampleTimesMs = samples.map { it.timeMs },
                squatAngles = samples.map { it.squatAngle.safeAngle() },
                pushupAngles = samples.map { it.pushupAngle.safeAngle() },
                activeSignal = predStages.map { stageLabel ->
                    when (stageLabel) {
                        "down" -> 1f
                        "transition" -> 0.5f
                        else -> 0f
                    }
                },
                representativeAngle = samples.minOfOrNull { it.pushupAngle.safeAngle() } ?: 180f
            )
        )
    }

    private fun fuseMotionSummary(
        ruleSummary: VideoMotionSummary,
        modelSummary: ModelMotionSummary?,
        forcedSportType: String
    ): VideoMotionSummary {
        if (modelSummary == null) {
            return ruleSummary
        }

        val model = modelSummary.summary
        val preferModel = when {
            forcedSportType == "push_up" -> true
            model.inferredSportType == "push_up" && model.confidence >= 0.55f -> true
            model.inferredSportType == "push_up" && ruleSummary.inferredSportType == "push_up" -> true
            ruleSummary.repetitionCount == 0 && model.repetitionCount > 0 -> true
            else -> false
        }

        if (!preferModel) {
            return ruleSummary.copy(debug = "${ruleSummary.debug} | ${model.debug}")
        }

        return ruleSummary.copy(
            inferredSportType = if (model.inferredSportType != "unknown") model.inferredSportType else ruleSummary.inferredSportType,
            repetitionCount = model.repetitionCount,
            stageHint = model.stageHint,
            confidence = max(ruleSummary.confidence, model.confidence),
            debug = "${ruleSummary.debug} | ${model.debug}",
            activeSignal = if (model.activeSignal.isNotEmpty()) model.activeSignal else ruleSummary.activeSignal
        )
    }

    private fun buildFrameFeature(
        sample: ExerciseFrameSample,
        prevSample: ExerciseFrameSample?
    ): FloatArray {
        val keypoints = sample.poseEstimate.keypoints
        val shoulder = avgPoint(keypoints, 5, 6)
        val hip = avgPoint(keypoints, 11, 12)
        val torsoScale = max(distance(shoulder, hip), 1e-4f)
        val values = ArrayList<Float>(NativeRepCounter.FEATURE_DIM)

        SELECTED_KEYPOINTS.forEach { index ->
            val point = keypoints.getOrNull(index)
            if (point != null) {
                values += (point.x - hip.first) / torsoScale
                values += (point.y - hip.second) / torsoScale
                values += point.confidence
            } else {
                values += 0f
                values += 0f
                values += 0f
            }
        }

        values += sample.pushupAngle.safeAngle()
        values += sample.squatAngle.safeAngle()
        values += sample.pushupDepth
        values += sample.torsoLinearity
        values += sample.torsoHorizontal
        values += sample.torsoVertical
        values += sample.leftElbowAngle.safeAngle()
        values += sample.rightElbowAngle.safeAngle()
        values += sample.leftKneeAngle.safeAngle()
        values += sample.rightKneeAngle.safeAngle()
        values += sample.hipToShoulderYDiff
        values += sample.hipToWristYDiff

        values += delta(prevSample?.pushupAngle, sample.pushupAngle)
        values += delta(prevSample?.pushupDepth, sample.pushupDepth)
        values += delta(prevSample?.torsoHorizontal, sample.torsoHorizontal)
        values += delta(prevSample?.hipToShoulderYDiff, sample.hipToShoulderYDiff)
        values += delta(prevSample?.leftElbowAngle, sample.leftElbowAngle)
        values += delta(prevSample?.rightElbowAngle, sample.rightElbowAngle)

        values += sample.score
        return values.toFloatArray()
    }

    private fun countFromStageLabels(labels: List<String>): Int {
        var reps = 0
        var inDown = false
        labels.forEach { label ->
            if (label == "down") {
                if (!inDown) {
                    reps += 1
                    inDown = true
                }
            } else {
                inDown = false
            }
        }
        return reps
    }

    private fun argmax(values: IntArray): Int = values.indices.maxByOrNull { values[it] } ?: 0

    private fun softmaxMax(logits: FloatArray): Float {
        if (logits.isEmpty()) {
            return 0f
        }
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { kotlin.math.exp((it - maxLogit).toDouble()) }
        val sum = exps.sum().takeIf { it > 0.0 } ?: return 0f
        val best = exps.maxOrNull() ?: 0.0
        return (best / sum).toFloat()
    }

    private fun delta(prev: Float?, curr: Float): Float {
        val safeCurr = if (curr.isFinite()) curr else 0f
        val safePrev = if (prev != null && prev.isFinite()) prev else safeCurr
        return safeCurr - safePrev
    }

    private fun avgPoint(points: List<PosePoint>, indexA: Int, indexB: Int): Pair<Float, Float> {
        val a = points.getOrNull(indexA)
        val b = points.getOrNull(indexB)
        val ax = a?.x ?: 0f
        val ay = a?.y ?: 0f
        val bx = b?.x ?: 0f
        val by = b?.y ?: 0f
        return Pair((ax + bx) / 2f, (ay + by) / 2f)
    }

    private fun distance(a: Pair<Float, Float>, b: Pair<Float, Float>): Float {
        val dx = a.first - b.first
        val dy = a.second - b.second
        return sqrt(dx * dx + dy * dy)
    }

    private val SELECTED_KEYPOINTS = intArrayOf(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
}
