package com.aisport.poster

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import com.aisport.video.VideoMotionSummary
import kotlin.math.max

object WorkoutTimelineRenderer {

    fun render(summary: VideoMotionSummary): Bitmap? {
        if (summary.sampleTimesMs.isEmpty() || summary.activeSignal.isEmpty()) return null

        val width = 1080
        val height = 520
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.parseColor("#F8FAFC"))

        val frame = RectF(72f, 96f, width - 48f, height - 70f)
        val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
        }
        canvas.drawRoundRect(RectF(28f, 28f, width - 28f, height - 28f), 28f, 28f, borderPaint)

        val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#0F172A")
            textSize = 38f
            isFakeBoldText = true
        }
        val subPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#475569")
            textSize = 26f
        }
        canvas.drawText(
            "视频监测曲线 · ${summary.inferredSportType.ifBlank { "unknown" }}",
            58f,
            74f,
            titlePaint
        )
        canvas.drawText(
            "次数 ${summary.repetitionCount} · 置信度 ${"%.2f".format(summary.confidence)} · 采样 ${summary.sampleTimesMs.size} 帧",
            58f,
            110f,
            subPaint
        )

        val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#CBD5E1")
            strokeWidth = 3f
        }
        canvas.drawLine(frame.left, frame.bottom, frame.right, frame.bottom, axisPaint)
        canvas.drawLine(frame.left, frame.top, frame.left, frame.bottom, axisPaint)

        repeat(4) { step ->
            val y = frame.top + (frame.height() * step / 3f)
            canvas.drawLine(frame.left, y, frame.right, y, Paint(axisPaint).apply { alpha = 80 })
        }

        val signalPath = buildPath(summary.activeSignal, frame)
        val signalPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = when (summary.inferredSportType) {
                "push_up" -> Color.parseColor("#2563EB")
                else -> Color.parseColor("#16A34A")
            }
            style = Paint.Style.STROKE
            strokeWidth = 7f
        }
        canvas.drawPath(signalPath, signalPaint)

        val markerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#F97316")
            style = Paint.Style.FILL
        }
        val peakIndex = summary.activeSignal.indices.maxByOrNull { summary.activeSignal[it] } ?: 0
        val peakX = frame.left + frame.width() * peakIndex / max(1, summary.activeSignal.lastIndex).toFloat()
        val maxSignal = summary.activeSignal.maxOrNull()?.takeIf { it > 0f } ?: 1f
        val peakY = frame.bottom - frame.height() * (summary.activeSignal[peakIndex] / maxSignal)
        canvas.drawCircle(peakX, peakY, 10f, markerPaint)

        val footerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#64748B")
            textSize = 24f
        }
        canvas.drawText("起点", frame.left, height - 30f, footerPaint)
        canvas.drawText(
            "${summary.sampleTimesMs.lastOrNull()?.div(1000f)?.let { String.format("%.1fs", it) } ?: "--"}",
            frame.right - 90f,
            height - 30f,
            footerPaint
        )
        canvas.drawText(summary.stageHint.take(26), 58f, height - 30f, footerPaint)
        return bitmap
    }

    private fun buildPath(signal: List<Float>, frame: RectF): Path {
        val path = Path()
        val maxSignal = signal.maxOrNull()?.takeIf { it > 0f } ?: 1f
        signal.forEachIndexed { index, value ->
            val x = frame.left + frame.width() * index / max(1, signal.lastIndex).toFloat()
            val y = frame.bottom - frame.height() * (value / maxSignal)
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        return path
    }
}
