package com.aisport.poster

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Shader
import com.aisport.video.VideoMotionSummary
import com.aisport.workout.WorkoutInsights
import kotlin.math.max

object WorkoutTimelineRenderer {

    fun render(summary: VideoMotionSummary): Bitmap? {
        if (summary.sampleTimesMs.isEmpty() || summary.activeSignal.isEmpty()) return null

        val width = 1080
        val height = 560
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.parseColor("#F8FAFC"))

        val outer = RectF(24f, 24f, width - 24f, height - 24f)
        canvas.drawRoundRect(outer, 34f, 34f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
        })
        val chart = RectF(70f, 170f, width - 70f, height - 88f)

        canvas.drawText(
            "运动波动曲线 · ${displaySportType(summary.inferredSportType)}",
            62f,
            80f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.parseColor("#0F172A")
                textSize = 38f
                isFakeBoldText = true
            }
        )
        canvas.drawText(
            "次数 ${summary.repetitionCount} · 置信度 ${"%.2f".format(summary.confidence)} · 时长 ${WorkoutInsights.formatDuration(summary.sampleTimesMs.last())}",
            62f,
            122f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.parseColor("#64748B")
                textSize = 26f
            }
        )

        val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#CBD5E1")
            strokeWidth = 3f
        }
        canvas.drawLine(chart.left, chart.bottom, chart.right, chart.bottom, axisPaint)
        canvas.drawLine(chart.left, chart.top, chart.left, chart.bottom, axisPaint)
        repeat(4) { step ->
            val y = chart.top + chart.height() * step / 3f
            canvas.drawLine(chart.left, y, chart.right, y, Paint(axisPaint).apply { alpha = 70 })
        }

        val linePath = Path()
        val fillPath = Path()
        val maxSignal = summary.activeSignal.maxOrNull()?.takeIf { it > 0f } ?: 1f
        summary.activeSignal.forEachIndexed { index, value ->
            val x = chart.left + chart.width() * index / max(1, summary.activeSignal.lastIndex).toFloat()
            val y = chart.bottom - chart.height() * (value / maxSignal)
            if (index == 0) {
                linePath.moveTo(x, y)
                fillPath.moveTo(x, chart.bottom)
                fillPath.lineTo(x, y)
            } else {
                linePath.lineTo(x, y)
                fillPath.lineTo(x, y)
            }
        }
        fillPath.lineTo(chart.right, chart.bottom)
        fillPath.close()
        canvas.drawPath(fillPath, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(
                chart.left,
                chart.top,
                chart.left,
                chart.bottom,
                Color.parseColor("#552563EB"),
                Color.parseColor("#112563EB"),
                Shader.TileMode.CLAMP
            )
            style = Paint.Style.FILL
        })
        canvas.drawPath(linePath, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = if (summary.inferredSportType == "push_up") Color.parseColor("#2563EB") else Color.parseColor("#16A34A")
            style = Paint.Style.STROKE
            strokeWidth = 8f
        })

        val lastIndex = summary.activeSignal.lastIndex
        val lastX = chart.left + chart.width() * lastIndex / max(1, lastIndex).toFloat()
        val lastY = chart.bottom - chart.height() * (summary.activeSignal.last() / maxSignal)
        canvas.drawCircle(lastX, lastY, 12f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#F97316") })

        canvas.drawText("起点", chart.left, height - 42f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#64748B")
            textSize = 24f
        })
        canvas.drawText(
            WorkoutInsights.formatDuration(summary.sampleTimesMs.last()),
            chart.right - 110f,
            height - 42f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.parseColor("#64748B")
                textSize = 24f
            }
        )
        return bitmap
    }

    private fun displaySportType(type: String): String = when (type) {
        "push_up" -> "俯卧撑"
        "squat" -> "深蹲"
        "sit_up" -> "仰卧起坐"
        else -> "未知动作"
    }
}
