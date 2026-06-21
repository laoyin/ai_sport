package com.aisport.pose

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class PosePoint(
    val x: Float,
    val y: Float,
    val confidence: Float
)

data class PoseEstimate(
    val score: Float,
    val stageHint: String,
    val qualityHint: String,
    val keypoints: List<PosePoint>,
    val skeletonEdges: List<Pair<Int, Int>>,
    val engineName: String,
    val placeholder: Boolean
)

interface PoseEstimator {
    fun estimate(bitmap: Bitmap): PoseEstimate
}

/**
 * Placeholder bridge for future YOLO-Pose inference.
 * The app already depends on this stable interface so we can later replace the
 * internals with real MNN / ONNX pose output without refactoring the UI flow.
 */
class YoloPoseEstimatorBridge : PoseEstimator {

    override fun estimate(bitmap: Bitmap): PoseEstimate {
        val score = estimateVisualActivityScore(bitmap)
        val quality = when {
            score >= 0.78f -> "excellent"
            score >= 0.62f -> "good"
            score >= 0.48f -> "fair"
            else -> "needs_attention"
        }
        val stage = when {
            bitmap.height > bitmap.width * 1.2f -> "竖幅动作帧更完整"
            score >= 0.72f -> "动作主体集中，适合做封面"
            else -> "检测到有效动作主体"
        }

        return PoseEstimate(
            score = score,
            stageHint = stage,
            qualityHint = quality,
            keypoints = buildSkeletonTemplate(bitmap.width.toFloat(), bitmap.height.toFloat()),
            skeletonEdges = cocoEdges,
            engineName = "YOLO-Pose Bridge (stub)",
            placeholder = true
        )
    }

    private fun estimateVisualActivityScore(bitmap: Bitmap): Float {
        val sampleStepX = max(1, bitmap.width / 24)
        val sampleStepY = max(1, bitmap.height / 24)
        var edgeAccum = 0f
        var brightnessAccum = 0f
        var count = 0
        var y = 0
        while (y < bitmap.height - sampleStepY) {
            var x = 0
            while (x < bitmap.width - sampleStepX) {
                val c1 = bitmap.getPixel(x, y)
                val c2 = bitmap.getPixel(min(bitmap.width - 1, x + sampleStepX), y)
                val c3 = bitmap.getPixel(x, min(bitmap.height - 1, y + sampleStepY))
                val l1 = luminance(c1)
                val l2 = luminance(c2)
                val l3 = luminance(c3)
                edgeAccum += abs(l1 - l2) + abs(l1 - l3)
                brightnessAccum += l1
                count++
                x += sampleStepX
            }
            y += sampleStepY
        }
        if (count == 0) return 0.4f
        val edgeScore = (edgeAccum / count / 255f).coerceIn(0f, 1f)
        val brightnessScore = (brightnessAccum / count / 255f).let {
            1f - abs(it - 0.55f)
        }.coerceIn(0f, 1f)
        return (edgeScore * 0.7f + brightnessScore * 0.3f).coerceIn(0.18f, 0.92f)
    }

    private fun luminance(color: Int): Float {
        return 0.299f * Color.red(color) + 0.587f * Color.green(color) + 0.114f * Color.blue(color)
    }

    private fun buildSkeletonTemplate(width: Float, height: Float): List<PosePoint> {
        val cx = width * 0.5f
        val headY = height * 0.18f
        val shoulderY = height * 0.28f
        val elbowY = height * 0.41f
        val wristY = height * 0.56f
        val hipY = height * 0.53f
        val kneeY = height * 0.73f
        val ankleY = height * 0.9f

        fun p(nx: Float, y: Float) = PosePoint(nx, y, 0.88f)

        return listOf(
            p(cx, headY),
            p(cx - width * 0.04f, headY - height * 0.01f),
            p(cx + width * 0.04f, headY - height * 0.01f),
            p(cx - width * 0.07f, headY + height * 0.01f),
            p(cx + width * 0.07f, headY + height * 0.01f),
            p(cx - width * 0.13f, shoulderY),
            p(cx + width * 0.13f, shoulderY),
            p(cx - width * 0.19f, elbowY),
            p(cx + width * 0.19f, elbowY),
            p(cx - width * 0.22f, wristY),
            p(cx + width * 0.22f, wristY),
            p(cx - width * 0.1f, hipY),
            p(cx + width * 0.1f, hipY),
            p(cx - width * 0.09f, kneeY),
            p(cx + width * 0.09f, kneeY),
            p(cx - width * 0.1f, ankleY),
            p(cx + width * 0.1f, ankleY)
        )
    }

    private val cocoEdges = listOf(
        0 to 1, 0 to 2, 1 to 3, 2 to 4,
        5 to 6, 5 to 7, 7 to 9, 6 to 8, 8 to 10,
        5 to 11, 6 to 12, 11 to 12,
        11 to 13, 13 to 15, 12 to 14, 14 to 16
    )
}
