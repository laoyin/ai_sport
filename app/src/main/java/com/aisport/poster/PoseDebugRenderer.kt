package com.aisport.poster

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import com.aisport.pose.PoseEstimate

object PoseDebugRenderer {

    fun renderOverlayOnly(source: Bitmap, poseEstimate: PoseEstimate?): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        poseEstimate?.let { drawPoseOverlay(canvas, result.width.toFloat(), result.height.toFloat(), it) }
        return result
    }

    fun render(source: Bitmap, poseEstimate: PoseEstimate?, sourceLabel: String = ""): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        drawHeader(canvas, result.width.toFloat(), poseEstimate, sourceLabel)
        poseEstimate?.let { drawPoseOverlay(canvas, result.width.toFloat(), result.height.toFloat(), it) }
        return result
    }

    fun renderThumbnail(
        source: Bitmap,
        poseEstimate: PoseEstimate?,
        sourceLabel: String = "",
        maxWidth: Int = 420,
        withHeader: Boolean = true
    ): Bitmap {
        val rendered = if (withHeader) {
            render(source, poseEstimate, sourceLabel)
        } else {
            renderOverlayOnly(source, poseEstimate)
        }
        if (rendered.width <= maxWidth) {
            return rendered
        }
        val scaledHeight = (rendered.height * (maxWidth / rendered.width.toFloat())).toInt().coerceAtLeast(1)
        val scaled = Bitmap.createScaledBitmap(rendered, maxWidth, scaledHeight, true)
        if (scaled !== rendered) {
            rendered.recycle()
        }
        return scaled
    }

    private fun drawHeader(
        canvas: Canvas,
        width: Float,
        poseEstimate: PoseEstimate?,
        sourceLabel: String
    ) {
        val headerHeight = 160f
        val headerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(
                0f,
                0f,
                width,
                headerHeight,
                Color.parseColor("#061826"),
                Color.parseColor("#0F766E"),
                Shader.TileMode.CLAMP
            )
            alpha = 220
        }
        canvas.drawRoundRect(RectF(18f, 18f, width - 18f, headerHeight), 28f, 28f, headerPaint)

        val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 34f
            isFakeBoldText = true
        }
        val subPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#D1FAE5")
            textSize = 24f
        }
        val badgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = if (poseEstimate?.placeholder == true) {
                Color.parseColor("#F97316")
            } else {
                Color.parseColor("#22C55E")
            }
        }
        val badgeTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 22f
            isFakeBoldText = true
        }

        canvas.drawText("YOLO Pose Diagnostic", 42f, 64f, titlePaint)
        canvas.drawText(
            "score=${poseEstimate?.score?.let { String.format("%.2f", it) } ?: "--"} · ${poseEstimate?.qualityHint ?: "unknown"}",
            42f,
            102f,
            subPaint
        )
        val label = sourceLabel.ifBlank { "当前输入" }
        canvas.drawText(label.take(40), 42f, 134f, subPaint)

        val badgeText = if (poseEstimate?.placeholder == true) "Stub" else "MNN"
        val badgeRect = RectF(width - 150f, 42f, width - 42f, 92f)
        canvas.drawRoundRect(badgeRect, 24f, 24f, badgePaint)
        canvas.drawText(badgeText, badgeRect.left + 28f, badgeRect.bottom - 16f, badgeTextPaint)
    }

    private fun drawPoseOverlay(
        canvas: Canvas,
        sourceWidth: Float,
        sourceHeight: Float,
        poseEstimate: PoseEstimate
    ) {
        val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#22C55E")
            strokeWidth = (sourceWidth.coerceAtMost(sourceHeight) * 0.008f).coerceAtLeast(5f)
        }
        val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#F97316")
        }
        val pointRadius = (sourceWidth.coerceAtMost(sourceHeight) * 0.012f).coerceAtLeast(6f)

        for ((start, end) in poseEstimate.skeletonEdges) {
            val p1 = poseEstimate.keypoints.getOrNull(start) ?: continue
            val p2 = poseEstimate.keypoints.getOrNull(end) ?: continue
            if (p1.confidence < 0.05f || p2.confidence < 0.05f) continue
            canvas.drawLine(p1.x, p1.y, p2.x, p2.y, linePaint)
        }

        poseEstimate.keypoints.forEachIndexed { index, point ->
            if (point.confidence < 0.05f) return@forEachIndexed
            canvas.drawCircle(point.x, point.y, pointRadius, pointPaint)
            val indexPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.WHITE
                textSize = (pointRadius * 1.8f).coerceAtLeast(18f)
                isFakeBoldText = true
            }
            val bounds = Rect()
            val text = index.toString()
            indexPaint.getTextBounds(text, 0, text.length, bounds)
            canvas.drawText(
                text,
                point.x - bounds.width() / 2f,
                point.y - pointRadius - 8f,
                indexPaint
            )
        }
    }
}
