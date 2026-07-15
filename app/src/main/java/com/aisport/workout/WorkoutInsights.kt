package com.aisport.workout

import com.aisport.exercise.ExerciseFrameSample
import com.aisport.video.VideoMotionSummary
import kotlin.math.max

data class WorkoutSummaryMetrics(
    val durationMs: Long,
    val averageRepSeconds: Float,
    val calories: Float,
    val bestShotLabel: String,
    val advice: String
)

object WorkoutInsights {

    fun buildSummary(
        motionSummary: VideoMotionSummary?,
        samples: List<ExerciseFrameSample>,
        poseQuality: String?,
        sportTypeOverride: String? = null
    ): WorkoutSummaryMetrics {
        val durationMs = when {
            motionSummary?.sampleTimesMs?.isNotEmpty() == true -> motionSummary.sampleTimesMs.last()
            samples.isNotEmpty() -> samples.last().timeMs
            else -> 0L
        }.coerceAtLeast(0L)

        val sportType = sportTypeOverride ?: motionSummary?.inferredSportType ?: "unknown"
        val reps = motionSummary?.repetitionCount ?: 0
        val averageRepSeconds = if (reps > 0 && durationMs > 0L) {
            durationMs / 1000f / reps
        } else {
            0f
        }

        val calories = estimateCalories(
            sportType = sportType,
            reps = reps,
            durationMs = durationMs
        )
        return WorkoutSummaryMetrics(
            durationMs = durationMs,
            averageRepSeconds = averageRepSeconds,
            calories = calories,
            bestShotLabel = buildBestShotLabel(sportType, reps),
            advice = buildAdvice(sportType, poseQuality, averageRepSeconds)
        )
    }

    fun estimateCalories(
        sportType: String,
        reps: Int,
        durationMs: Long
    ): Float {
        val perRep = when (sportType) {
            "push_up" -> 0.45f
            "squat" -> 0.38f
            "sit_up" -> 0.32f
            else -> 0.28f
        }
        val repCalories = reps * perRep
        val timeFloor = (durationMs / 60_000f) * when (sportType) {
            "push_up" -> 4.6f
            "squat" -> 4.2f
            "sit_up" -> 3.8f
            else -> 3.0f
        }
        return max(repCalories, timeFloor).coerceAtLeast(0f)
    }

    fun formatDuration(durationMs: Long): String {
        val totalSeconds = durationMs / 1000L
        val minutes = totalSeconds / 60L
        val seconds = totalSeconds % 60L
        return "%02d:%02d".format(minutes, seconds)
    }

    fun formatPace(secondsPerRep: Float): String {
        if (secondsPerRep <= 0f) return "--"
        return String.format("%.1f 秒/次", secondsPerRep)
    }

    fun formatCalories(calories: Float): String = String.format("%.1f kcal", calories)

    private fun buildBestShotLabel(sportType: String, reps: Int): String {
        val action = when (sportType) {
            "push_up" -> "俯卧撑"
            "squat" -> "深蹲"
            "sit_up" -> "仰卧起坐"
            else -> "运动"
        }
        return if (reps > 0) "最佳 $action 第 ${max(1, reps / 2)} 次" else "最佳动作截图"
    }

    private fun buildAdvice(
        sportType: String,
        poseQuality: String?,
        averageRepSeconds: Float
    ): String {
        if (poseQuality == "excellent") {
            return when (sportType) {
                "push_up" -> "节奏稳定，继续保持核心收紧和手臂发力。"
                "squat" -> "下蹲深度和身体控制都不错，继续保持。"
                "sit_up" -> "动作连贯，腹部发力和节奏控制较好。"
                else -> "动作完成度不错，继续保持当前状态。"
            }
        }
        return when (sportType) {
            "push_up" -> if (averageRepSeconds in 0.1f..1.3f) {
                "节奏偏快，建议放慢下压过程，动作会更标准。"
            } else {
                "保持肩、髋、踝尽量成一直线，提升俯卧撑稳定性。"
            }
            "squat" -> "建议继续保持膝盖与脚尖方向一致，重心更稳定。"
            "sit_up" -> "建议控制起身节奏，避免借力过猛，保持腹部发力。"
            else -> "建议优化拍摄角度，让模型更稳定识别你的动作。"
        }
    }
}
