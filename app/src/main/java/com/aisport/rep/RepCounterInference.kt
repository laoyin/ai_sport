package com.aisport.rep

import com.aisport.exercise.ExerciseAnalyzer
import com.aisport.exercise.ExerciseFrameSample
import com.aisport.pose.PosePoint
import com.aisport.video.VideoMotionSummary
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

object RepCounterInference {

    private val selectedKeypoints = intArrayOf(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    fun inferMotionSummary(
        samples: List<ExerciseFrameSample>,
        repCounter: NativeRepCounter?,
        forcedSportType: String = "auto"
    ): VideoMotionSummary {
        val ruleSummary = ExerciseAnalyzer.analyze(samples, forcedSportType)
        if (repCounter == null || samples.size < NativeRepCounter.SEQ_LEN) {
            return ruleSummary
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
            return ruleSummary
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
        val modelSummary = VideoMotionSummary(
            inferredSportType = inferredSportType,
            repetitionCount = predictedCount,
            stageHint = stageHint,
            confidence = avgConfidence,
            debug = "mnn_action=$predAction windows=$windows count=$predictedCount avg_conf=${"%.2f".format(avgConfidence)}",
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
        return fuseMotionSummary(ruleSummary, modelSummary, forcedSportType)
    }

    fun buildFrameFeature(sample: ExerciseFrameSample, prevSample: ExerciseFrameSample?): FloatArray {
        val keypoints = sample.poseEstimate.keypoints
        val shoulder = avgPoint(keypoints, 5, 6)
        val hip = avgPoint(keypoints, 11, 12)
        val torsoScale = max(distance(shoulder, hip), 1e-4f)
        val values = ArrayList<Float>(NativeRepCounter.FEATURE_DIM)

        selectedKeypoints.forEach { index ->
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

    fun countFromStageLabels(labels: List<String>): Int {
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

    private fun fuseMotionSummary(
        ruleSummary: VideoMotionSummary,
        modelSummary: VideoMotionSummary,
        forcedSportType: String
    ): VideoMotionSummary {
        val preferModel = when {
            forcedSportType == "push_up" -> true
            modelSummary.inferredSportType == "push_up" && modelSummary.confidence >= 0.55f -> true
            modelSummary.inferredSportType == "push_up" && ruleSummary.inferredSportType == "push_up" -> true
            ruleSummary.repetitionCount == 0 && modelSummary.repetitionCount > 0 -> true
            else -> false
        }
        if (!preferModel) {
            return ruleSummary.copy(debug = "${ruleSummary.debug} | ${modelSummary.debug}")
        }
        return ruleSummary.copy(
            inferredSportType = if (modelSummary.inferredSportType != "unknown") modelSummary.inferredSportType else ruleSummary.inferredSportType,
            repetitionCount = modelSummary.repetitionCount,
            stageHint = modelSummary.stageHint,
            confidence = max(ruleSummary.confidence, modelSummary.confidence),
            debug = "${ruleSummary.debug} | ${modelSummary.debug}",
            activeSignal = if (modelSummary.activeSignal.isNotEmpty()) modelSummary.activeSignal else ruleSummary.activeSignal
        )
    }

    private fun argmax(values: IntArray): Int = values.indices.maxByOrNull { values[it] } ?: 0

    private fun softmaxMax(logits: FloatArray): Float {
        if (logits.isEmpty()) return 0f
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp((it - maxLogit).toDouble()) }
        val sum = exps.sum().takeIf { it > 0.0 } ?: return 0f
        return ((exps.maxOrNull() ?: 0.0) / sum).toFloat()
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

    private fun Float.safeAngle(): Float = if (isFinite()) coerceIn(0f, 180f) else 180f
}
