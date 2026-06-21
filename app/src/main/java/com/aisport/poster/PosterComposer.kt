package com.aisport.poster

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Paint.Style
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import com.aisport.pose.PoseEstimate
import com.aisport.vision.SportAnalysis

object PosterComposer {

    fun createPoster(source: Bitmap, analysis: SportAnalysis, poseEstimate: PoseEstimate? = null, sourceLabel: String = ""): Bitmap {
        val width = 1080
        val height = 1600
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(
                0f,
                0f,
                width.toFloat(),
                height.toFloat(),
                Color.parseColor("#F7F1E5"),
                Color.parseColor("#E8F4F1"),
                Shader.TileMode.CLAMP
            )
        }
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), bgPaint)

        val imageTop = 110
        val imageHeight = 760
        val imageRect = RectF(70f, imageTop.toFloat(), width - 70f, (imageTop + imageHeight).toFloat())
        canvas.drawRoundRect(imageRect, 40f, 40f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
        })
        drawCoverBitmap(canvas, source, imageRect)
        poseEstimate?.let { drawPoseOverlay(canvas, imageRect, it, source.width.toFloat(), source.height.toFloat()) }

        val chipPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#0F766E")
        }
        val chipRect = RectF(90f, 140f, 350f, 220f)
        canvas.drawRoundRect(chipRect, 28f, 28f, chipPaint)
        val chipText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 38f
            isFakeBoldText = true
        }
        canvas.drawText("AI SPORT", 128f, 193f, chipText)

        val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#1F2937")
            textSize = 64f
            isFakeBoldText = true
        }
        val subtitlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#475569")
            textSize = 34f
        }
        val labelTop = 960f
        canvas.drawText(analysis.summaryTitle.ifBlank { "运动战报" }, 82f, labelTop, titlePaint)
        canvas.drawText(
            "动作：${displaySportType(analysis.sportType)}  ·  姿态：${displayQuality(analysis.poseQuality)}",
            84f,
            labelTop + 60f,
            subtitlePaint
        )
        if (sourceLabel.isNotBlank()) {
            canvas.drawText(sourceLabel, 84f, labelTop + 108f, subtitlePaint)
        }

        val statRect = RectF(70f, 1050f, width - 70f, 1245f)
        canvas.drawRoundRect(statRect, 34f, 34f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
        })
        drawStat(canvas, "次数", analysis.repetitionCount.coerceAtLeast(1).toString(), 110f, 1125f)
        drawStat(canvas, "阶段", analysis.stage.ifBlank { "动作帧稳定" }, 390f, 1125f)
        drawStat(canvas, "可信度", "${(analysis.confidence * 100).toInt()}%", 760f, 1125f)

        val sectionTitlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#0F172A")
            textSize = 34f
            isFakeBoldText = true
        }
        val bodyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#334155")
            textSize = 32f
        }
        canvas.drawText("亮点总结", 82f, 1335f, sectionTitlePaint)
        drawParagraph(canvas, analysis.highlight.ifBlank { "动作主体清晰，画面适合生成运动宣传卡。" }, 82f, 1385f, 916f, bodyPaint)
        canvas.drawText("改进建议", 82f, 1465f, sectionTitlePaint)
        drawParagraph(canvas, analysis.riskTip.ifBlank { "继续补充姿态关键点检测后，可进一步给出更细的动作纠错建议。" }, 82f, 1515f, 916f, bodyPaint)

        val sloganPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#F97316")
            textSize = 42f
            isFakeBoldText = true
        }
        drawParagraph(canvas, analysis.slogan.ifBlank { "每一次动作，都值得被记录。" }, 82f, 1580f, 916f, sloganPaint)
        return bitmap
    }

    fun savePoster(context: Context, poster: Bitmap): String? {
        return savePosterUri(context, poster)?.toString()
    }

    fun savePosterUri(context: Context, poster: Bitmap): Uri? {
        val fileName = "ai_sport_${System.currentTimeMillis()}.png"
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/AI Sport")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { out ->
            poster.compress(Bitmap.CompressFormat.PNG, 100, out)
        } ?: return null

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
        }
        return uri
    }

    fun createShareUri(context: Context, poster: Bitmap): Uri? {
        return savePosterUri(context, poster)
    }

    private fun drawCoverBitmap(canvas: Canvas, source: Bitmap, target: RectF) {
        val scale = maxOf(target.width() / source.width, target.height() / source.height)
        val drawWidth = source.width * scale
        val drawHeight = source.height * scale
        val left = target.left + (target.width() - drawWidth) / 2f
        val top = target.top + (target.height() - drawHeight) / 2f
        val src = Rect(0, 0, source.width, source.height)
        val dst = RectF(left, top, left + drawWidth, top + drawHeight)
        canvas.save()
        canvas.clipRect(target)
        canvas.drawBitmap(source, src, dst, Paint(Paint.ANTI_ALIAS_FLAG))
        canvas.restore()
    }

    private fun drawStat(canvas: Canvas, label: String, value: String, x: Float, y: Float) {
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#64748B")
            textSize = 28f
        }
        val valuePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#111827")
            textSize = if (value.length > 8) 28f else 44f
            isFakeBoldText = true
        }
        canvas.drawText(label, x, y, labelPaint)
        canvas.drawText(value, x, y + 56f, valuePaint)
    }

    private fun drawPoseOverlay(
        canvas: Canvas,
        target: RectF,
        poseEstimate: PoseEstimate,
        sourceWidth: Float,
        sourceHeight: Float
    ) {
        val scale = maxOf(target.width() / sourceWidth, target.height() / sourceHeight)
        val drawWidth = sourceWidth * scale
        val drawHeight = sourceHeight * scale
        val left = target.left + (target.width() - drawWidth) / 2f
        val top = target.top + (target.height() - drawHeight) / 2f

        val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#22C55E")
            strokeWidth = 8f
            style = Style.STROKE
        }
        val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#F97316")
            style = Style.FILL
        }
        for ((start, end) in poseEstimate.skeletonEdges) {
            val p1 = poseEstimate.keypoints.getOrNull(start) ?: continue
            val p2 = poseEstimate.keypoints.getOrNull(end) ?: continue
            canvas.drawLine(
                left + p1.x * scale,
                top + p1.y * scale,
                left + p2.x * scale,
                top + p2.y * scale,
                linePaint
            )
        }
        for (point in poseEstimate.keypoints) {
            canvas.drawCircle(left + point.x * scale, top + point.y * scale, 8f, pointPaint)
        }
    }

    private fun drawParagraph(canvas: Canvas, text: String, x: Float, y: Float, maxWidth: Float, paint: Paint) {
        val lines = breakLines(text, paint, maxWidth)
        var currentY = y
        for (line in lines.take(3)) {
            canvas.drawText(line, x, currentY, paint)
            currentY += paint.textSize + 14f
        }
    }

    private fun breakLines(text: String, paint: Paint, maxWidth: Float): List<String> {
        if (text.isBlank()) return listOf("")
        val lines = mutableListOf<String>()
        var current = StringBuilder()
        for (ch in text) {
            val candidate = current.toString() + ch
            if (paint.measureText(candidate) > maxWidth && current.isNotEmpty()) {
                lines += current.toString()
                current = StringBuilder().append(ch)
            } else {
                current.append(ch)
            }
        }
        if (current.isNotEmpty()) {
            lines += current.toString()
        }
        return lines
    }

    private fun displaySportType(type: String): String = when (type) {
        "squat" -> "深蹲"
        "jumping_jack" -> "开合跳"
        "push_up" -> "俯卧撑"
        "lunge" -> "弓步"
        "plank" -> "平板支撑"
        "running" -> "跑步"
        "yoga" -> "瑜伽"
        else -> "运动"
    }

    private fun displayQuality(quality: String): String = when (quality) {
        "excellent" -> "优秀"
        "good" -> "良好"
        "fair" -> "一般"
        "needs_attention" -> "需注意"
        else -> "待判定"
    }
}
