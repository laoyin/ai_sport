package com.aisport.exercise

import com.aisport.pose.PoseEstimate
import com.aisport.pose.PosePoint
import com.aisport.video.VideoMotionSummary
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.pow
import kotlin.math.sqrt

data class ExerciseFrameSample(
    val timeMs: Long,
    val score: Float,
    val poseEstimate: PoseEstimate,
    val squatAngle: Float,
    val pushupAngle: Float,
    val pushupDepth: Float,
    val torsoLinearity: Float,
    val torsoHorizontal: Float,
    val torsoVertical: Float
)

object ExerciseAnalyzer {

    private const val SQUAT_UP_ANGLE = 160f
    private const val SQUAT_DOWN_ANGLE = 105f
    private const val PUSH_UP_UP_ANGLE = 155f
    private const val PUSH_UP_DOWN_ANGLE = 100f

    fun buildFrameSample(timeMs: Long, estimate: PoseEstimate): ExerciseFrameSample {
        return ExerciseFrameSample(
            timeMs = timeMs,
            score = estimate.score,
            poseEstimate = estimate,
            squatAngle = computeAverageKneeAngle(estimate),
            pushupAngle = computeAverageElbowAngle(estimate),
            pushupDepth = computePushupDepthScore(estimate),
            torsoLinearity = computeBodyLinearityScore(estimate),
            torsoHorizontal = computeTorsoHorizontalScore(estimate),
            torsoVertical = computeTorsoVerticalScore(estimate)
        )
    }

    fun analyze(samples: List<ExerciseFrameSample>, forcedSportType: String = "auto"): VideoMotionSummary {
        val valid = samples.filter { it.score >= 0.15f }
        if (valid.size < 4) {
            return VideoMotionSummary(
                inferredSportType = "unknown",
                repetitionCount = 0,
                stageHint = "有效姿态帧不足，暂时无法稳定统计次数",
                confidence = valid.maxOfOrNull { it.score } ?: 0f,
                debug = "valid=${valid.size}"
            )
        }

        val squatStats = analyzeSquat(valid)
        val pushupStats = analyzePushup(valid)
        val winner = when (forcedSportType) {
            "squat" -> squatStats
            "push_up" -> pushupStats
            else -> if (squatStats.confidence >= pushupStats.confidence) squatStats else pushupStats
        }

        val inferred = when (forcedSportType) {
            "squat" -> "squat"
            "push_up" -> "push_up"
            else -> if (winner.confidence >= 0.48f || winner.reps > 0) winner.sportType else "unknown"
        }
        val signal = when (inferred) {
            "squat" -> squatStats.signal
            "push_up" -> pushupStats.signal
            else -> emptyList()
        }

        return VideoMotionSummary(
            inferredSportType = inferred,
            repetitionCount = if (inferred == "unknown") 0 else winner.reps,
            stageHint = if (inferred == "unknown") "已完成多帧检测，但动作类别仍不够稳定" else winner.stageHint,
            confidence = winner.confidence,
            debug = "mode=$forcedSportType | squat=${"%.2f".format(squatStats.confidence)}/${squatStats.reps} | push=${"%.2f".format(pushupStats.confidence)}/${pushupStats.reps}",
            sampleTimesMs = valid.map { it.timeMs },
            squatAngles = valid.map { it.squatAngle.safeAngle() },
            pushupAngles = valid.map { it.pushupAngle.safeAngle() },
            activeSignal = signal,
            representativeAngle = winner.representativeAngle
        )
    }

    data class ExerciseStats(
        val sportType: String,
        val reps: Int,
        val confidence: Float,
        val stageHint: String,
        val signal: List<Float>,
        val representativeAngle: Float
    )

    private fun analyzeSquat(valid: List<ExerciseFrameSample>): ExerciseStats {
        val angles = smooth(valid.map { it.squatAngle.safeAngle() })
        val angleReps = countByStateMachine(
            angles = angles,
            downThreshold = SQUAT_DOWN_ANGLE,
            upThreshold = SQUAT_UP_ANGLE,
            downHoldFrames = 2,
            upHoldFrames = 2,
            minGapFrames = 2
        )
        val bendSignal = angles.map { (180f - it).coerceAtLeast(0f) }
        val signalReps = countByAdaptiveSignal(
            signal = bendSignal,
            lowerRatio = 0.28f,
            upperRatio = 0.62f,
            downHoldFrames = 1,
            upHoldFrames = 1,
            minGapFrames = 3
        )
        val reps = maxOf(angleReps, signalReps)
        val minAngle = angles.minOrNull() ?: 180f
        val maxAngle = angles.maxOrNull() ?: 180f
        val amplitude = maxAngle - minAngle
        val torsoVertical = valid.map { it.torsoVertical }.average().toFloat()
        val currentAngle = angles.lastOrNull() ?: 180f
        val stageHint = when {
            currentAngle <= SQUAT_DOWN_ANGLE -> "当前处于深蹲底部"
            currentAngle >= SQUAT_UP_ANGLE -> "当前接近站直完成位"
            else -> "当前处于深蹲上下过渡阶段"
        }
        val confidence = (
            ((SQUAT_UP_ANGLE - minAngle) / 60f).coerceIn(0f, 1f) * 0.35f +
                (amplitude / 55f).coerceIn(0f, 1f) * 0.25f +
                torsoVertical.coerceIn(0f, 1f) * 0.25f +
                (reps / 4f).coerceIn(0f, 1f) * 0.15f
            ).coerceIn(0f, 1f)

        return ExerciseStats(
            sportType = "squat",
            reps = reps,
            confidence = confidence,
            stageHint = stageHint,
            signal = bendSignal,
            representativeAngle = minAngle
        )
    }

    private fun analyzePushup(valid: List<ExerciseFrameSample>): ExerciseStats {
        val angles = smooth(valid.map { it.pushupAngle.safeAngle() })
        val bentArmSignal = angles.map { (180f - it).coerceAtLeast(0f) }
        val angleReps = countByStateMachine(
            angles = angles,
            downThreshold = PUSH_UP_DOWN_ANGLE,
            upThreshold = PUSH_UP_UP_ANGLE,
            downHoldFrames = 1,
            upHoldFrames = 1,
            minGapFrames = 2
        )
        val signalReps = countByAdaptiveSignal(
            signal = bentArmSignal,
            lowerRatio = 0.30f,
            upperRatio = 0.58f,
            downHoldFrames = 1,
            upHoldFrames = 1,
            minGapFrames = 3
        )
        val reps = maxOf(angleReps, signalReps)
        val minAngle = angles.minOrNull() ?: 180f
        val maxAngle = angles.maxOrNull() ?: 180f
        val amplitude = maxAngle - minAngle
        val torsoLinearity = valid.map { it.torsoLinearity }.average().toFloat()
        val torsoHorizontal = valid.map { it.torsoHorizontal }.average().toFloat()
        val avgDepth = valid.map { it.pushupDepth }.average().toFloat()
        val currentAngle = angles.lastOrNull() ?: 180f
        val stageHint = when {
            currentAngle <= PUSH_UP_DOWN_ANGLE -> "当前处于俯卧撑底部"
            currentAngle >= PUSH_UP_UP_ANGLE -> "当前接近俯卧撑撑起位"
            else -> "当前处于俯卧撑上下过渡阶段"
        }
        val confidence = (
            ((PUSH_UP_UP_ANGLE - minAngle) / 70f).coerceIn(0f, 1f) * 0.25f +
                (amplitude / 65f).coerceIn(0f, 1f) * 0.25f +
                avgDepth.coerceIn(0f, 1f) * 0.20f +
                torsoLinearity.coerceIn(0f, 1f) * 0.10f +
                torsoHorizontal.coerceIn(0f, 1f) * 0.20f +
                (reps / 4f).coerceIn(0f, 1f) * 0.20f
            ).coerceIn(0f, 1f)

        return ExerciseStats(
            sportType = "push_up",
            reps = reps,
            confidence = confidence,
            stageHint = stageHint,
            signal = bentArmSignal,
            representativeAngle = minAngle
        )
    }

    private fun countByStateMachine(
        angles: List<Float>,
        downThreshold: Float,
        upThreshold: Float,
        downHoldFrames: Int,
        upHoldFrames: Int,
        minGapFrames: Int
    ): Int {
        if (angles.size < 5) return 0

        val smoothed = smooth(angles)
        val minAngle = smoothed.minOrNull() ?: return 0
        val maxAngle = smoothed.maxOrNull() ?: return 0
        if (maxAngle - minAngle < 18f) return 0

        var reps = 0
        var stage = if ((smoothed.firstOrNull() ?: 180f) >= upThreshold) "up" else "mid"
        var downStreak = 0
        var upStreak = 0
        var cooldown = 0

        smoothed.forEach { angle ->
            if (cooldown > 0) {
                cooldown -= 1
            }

            if (angle <= downThreshold) {
                downStreak += 1
                upStreak = 0
            } else if (angle >= upThreshold) {
                upStreak += 1
                downStreak = 0
            } else {
                downStreak = 0
                upStreak = 0
            }

            if (downStreak >= downHoldFrames && cooldown == 0) {
                stage = "down"
            }

            if (stage == "down" && upStreak >= upHoldFrames && cooldown == 0) {
                reps += 1
                stage = "up"
                cooldown = minGapFrames
                downStreak = 0
                upStreak = 0
            }
        }

        return reps
    }

    private fun countByAdaptiveSignal(
        signal: List<Float>,
        lowerRatio: Float,
        upperRatio: Float,
        downHoldFrames: Int,
        upHoldFrames: Int,
        minGapFrames: Int
    ): Int {
        if (signal.size < 5) return 0

        val smoothed = smooth(signal)
        val minValue = smoothed.minOrNull() ?: return 0
        val maxValue = smoothed.maxOrNull() ?: return 0
        val amplitude = maxValue - minValue
        if (amplitude < 10f) return 0

        val lower = minValue + amplitude * lowerRatio
        val upper = minValue + amplitude * upperRatio

        var reps = 0
        var stage = if ((smoothed.firstOrNull() ?: 0f) <= lower) "up" else "mid"
        var downStreak = 0
        var upStreak = 0
        var cooldown = 0

        smoothed.forEach { value ->
            if (cooldown > 0) {
                cooldown -= 1
            }

            if (value >= upper) {
                downStreak += 1
                upStreak = 0
            } else if (value <= lower) {
                upStreak += 1
                downStreak = 0
            } else {
                downStreak = 0
                upStreak = 0
            }

            if (downStreak >= downHoldFrames && cooldown == 0) {
                stage = "down"
            }

            if (stage == "down" && upStreak >= upHoldFrames && cooldown == 0) {
                reps += 1
                stage = "up"
                cooldown = minGapFrames
                downStreak = 0
                upStreak = 0
            }
        }

        return reps
    }

    private fun smooth(values: List<Float>): List<Float> {
        if (values.size < 3) return values
        return values.indices.map { index ->
            val prev = values.getOrElse(index - 1) { values[index] }
            val curr = values[index]
            val next = values.getOrElse(index + 1) { values[index] }
            (prev + curr + next) / 3f
        }
    }

    private fun computeAverageKneeAngle(estimate: PoseEstimate): Float {
        val left = computeAngleAt(estimate, 11, 13, 15)
        val right = computeAngleAt(estimate, 12, 14, 16)
        return averageFinite(left, right)
    }

    private fun computeAverageElbowAngle(estimate: PoseEstimate): Float {
        val left = computeAngleAt(estimate, 5, 7, 9)
        val right = computeAngleAt(estimate, 6, 8, 10)
        return averageFinite(left, right)
    }

    private fun computePushupDepthScore(estimate: PoseEstimate): Float {
        val shoulder = averagePoint(estimate.keypoints.getOrNull(5), estimate.keypoints.getOrNull(6)) ?: return 0f
        val wrist = averagePoint(estimate.keypoints.getOrNull(9), estimate.keypoints.getOrNull(10)) ?: return 0f
        val hip = averagePoint(estimate.keypoints.getOrNull(11), estimate.keypoints.getOrNull(12)) ?: return 0f
        val torsoLen = distance(shoulder, hip).coerceAtLeast(1f)
        val verticalGap = (wrist.y - shoulder.y).coerceAtLeast(0f)
        val normalized = 1f - (verticalGap / (torsoLen * 1.25f)).coerceIn(0f, 1f)
        return normalized.coerceIn(0f, 1f)
    }

    private fun averageFinite(a: Float, b: Float): Float {
        return when {
            a.isFinite() && b.isFinite() -> (a + b) / 2f
            a.isFinite() -> a
            b.isFinite() -> b
            else -> 180f
        }
    }

    private fun computeBodyLinearityScore(estimate: PoseEstimate): Float {
        val leftBody = computeAngleAt(estimate, 5, 11, 15)
        val rightBody = computeAngleAt(estimate, 6, 12, 16)
        val bodyAngle = averageFinite(leftBody, rightBody)
        val score = 1f - abs(180f - bodyAngle) / 65f
        return score.coerceIn(0f, 1f)
    }

    private fun computeTorsoHorizontalScore(estimate: PoseEstimate): Float {
        val leftShoulder = estimate.keypoints.getOrNull(5)
        val rightShoulder = estimate.keypoints.getOrNull(6)
        val leftHip = estimate.keypoints.getOrNull(11)
        val rightHip = estimate.keypoints.getOrNull(12)
        val shoulder = averagePoint(leftShoulder, rightShoulder) ?: return 0f
        val hip = averagePoint(leftHip, rightHip) ?: return 0f
        val angleToHorizontal = abs(
            Math.toDegrees(atan2((hip.y - shoulder.y).toDouble(), (hip.x - shoulder.x).toDouble()))
        ).toFloat()
        val normalized = (angleToHorizontal % 180f).let { if (it > 90f) 180f - it else it }
        return (1f - normalized / 90f).coerceIn(0f, 1f)
    }

    private fun computeTorsoVerticalScore(estimate: PoseEstimate): Float {
        val horizontal = computeTorsoHorizontalScore(estimate)
        return (1f - horizontal).coerceIn(0f, 1f)
    }

    private fun averagePoint(a: PosePoint?, b: PosePoint?): PosePoint? {
        return when {
            a != null && b != null && a.confidence >= 0.05f && b.confidence >= 0.05f ->
                PosePoint((a.x + b.x) / 2f, (a.y + b.y) / 2f, (a.confidence + b.confidence) / 2f)
            a != null && a.confidence >= 0.05f -> a
            b != null && b.confidence >= 0.05f -> b
            else -> null
        }
    }

    private fun computeAngleAt(estimate: PoseEstimate, a: Int, b: Int, c: Int): Float {
        val p1 = estimate.keypoints.getOrNull(a) ?: return Float.NaN
        val p2 = estimate.keypoints.getOrNull(b) ?: return Float.NaN
        val p3 = estimate.keypoints.getOrNull(c) ?: return Float.NaN
        if (p1.confidence < 0.05f || p2.confidence < 0.05f || p3.confidence < 0.05f) return Float.NaN

        val v1x = p1.x - p2.x
        val v1y = p1.y - p2.y
        val v2x = p3.x - p2.x
        val v2y = p3.y - p2.y
        val dot = v1x * v2x + v1y * v2y
        val len1 = sqrt(v1x.pow(2) + v1y.pow(2))
        val len2 = sqrt(v2x.pow(2) + v2y.pow(2))
        if (len1 < 1e-3f || len2 < 1e-3f) return Float.NaN

        val cosValue = (dot / (len1 * len2)).coerceIn(-1f, 1f)
        return Math.toDegrees(acos(cosValue).toDouble()).toFloat()
    }

    private fun distance(a: PosePoint, b: PosePoint): Float {
        return sqrt((a.x - b.x).pow(2) + (a.y - b.y).pow(2))
    }

    private fun Float.safeAngle(): Float = if (isFinite()) coerceIn(0f, 180f) else 180f
}
