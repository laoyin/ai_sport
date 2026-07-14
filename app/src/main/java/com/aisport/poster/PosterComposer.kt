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
import com.aisport.workout.WorkoutInsights

object PosterComposer {

    fun createPoster(source: Bitmap, analysis: SportAnalysis, poseEstimate: PoseEstimate? = null, sourceLabel: String = ""): Bitmap {
        val width = 1080
        val height = 1680
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        val background = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(
                0f,
                0f,
                width.toFloat(),
                height.toFloat(),
                intArrayOf(
                    Color.parseColor("#F5EEDC"),
                    Color.parseColor("#F8FBFF"),
                    Color.parseColor("#E7F6EE")
                ),
                floatArrayOf(0f, 0.55f, 1f),
                Shader.TileMode.CLAMP
            )
        }
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), background)

        val heroRect = RectF(58f, 96f, width - 58f, 864f)
        canvas.drawRoundRect(heroRect, 44f, 44f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE })
        drawCoverBitmap(canvas, source, heroRect)
        poseEstimate?.let { drawPoseOverlay(canvas, heroRect, it, source.width.toFloat(), source.height.toFloat()) }

        val badgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#111827") }
        val badgeRect = RectF(86f, 126f, 308f, 198f)
        canvas.drawRoundRect(badgeRect, 26f, 26f, badgePaint)
        canvas.drawText("AI SPORT", 122f, 174f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 34f
            isFakeBoldText = true
        })

        val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#111827")
            textSize = 66f
            isFakeBoldText = true
        }
        val subPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#475569")
            textSize = 30f
        }
        canvas.drawText(analysis.summaryTitle.ifBlank { "AI 运动战报" }, 70f, 954f, titlePaint)
        canvas.drawText(
            "${displaySportType(analysis.sportType)} · ${displayQuality(analysis.poseQuality)} · ${WorkoutInsights.formatDuration(analysis.durationMs)}",
            70f,
            1004f,
            subPaint
        )
        if (sourceLabel.isNotBlank()) {
            canvas.drawText(sourceLabel.take(34), 70f, 1044f, subPaint)
        }

        val statsTop = 1082f
        drawStatCard(canvas, RectF(58f, statsTop, 292f, statsTop + 170f), "总次数", analysis.repetitionCount.toString(), "#FFF8F1")
        drawStatCard(canvas, RectF(310f, statsTop, 544f, statsTop + 170f), "热量消耗", WorkoutInsights.formatCalories(analysis.calories), "#F4F8FF")
        drawStatCard(canvas, RectF(562f, statsTop, 796f, statsTop + 170f), "平均节奏", WorkoutInsights.formatPace(analysis.avgRepSeconds), "#F3FBF6")
        drawStatCard(canvas, RectF(814f, statsTop, 1022f, statsTop + 170f), "可信度", "${(analysis.confidence * 100).toInt()}%", "#F9F5FF")

        val sectionTitlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#0F172A")
            textSize = 34f
            isFakeBoldText = true
        }
        val bodyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#334155")
            textSize = 30f
        }
        val sectionRect = RectF(58f, 1280f, width - 58f, 1602f)
        canvas.drawRoundRect(sectionRect, 38f, 38f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE })
        canvas.drawText("姿态建议", 92f, 1346f, sectionTitlePaint)
        drawParagraph(
            canvas,
            analysis.postureAdvice.ifBlank { analysis.riskTip.ifBlank { "继续保持动作节奏与身体稳定性。"} },
            92f,
            1398f,
            sectionRect.width() - 70f,
            bodyPaint
        )
        canvas.drawText("最佳截图", 92f, 1498f, sectionTitlePaint)
        canvas.drawText(analysis.bestShotLabel.ifBlank { "本次动作最佳帧" }, 92f, 1548f, bodyPaint)
        canvas.drawText("高光总结", 520f, 1498f, sectionTitlePaint)
        drawParagraph(
            canvas,
            analysis.highlight.ifBlank { "本次训练节奏稳定，动作识别连续性良好。" },
            520f,
            1548f,
            438f,
            bodyPaint
        )

        val sloganPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#F97316")
            textSize = 42f
            isFakeBoldText = true
        }
        drawParagraph(
            canvas,
            analysis.slogan.ifBlank { "每一次动作，都值得被记录。" },
            82f,
            1650f,
            930f,
            sloganPaint
        )
        return bitmap
    }

    fun savePoster(context: Context, poster: Bitmap): String? = savePosterUri(context, poster)?.toString()

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

    fun createShareUri(context: Context, poster: Bitmap): Uri? = savePosterUri(context, poster)

    private fun drawStatCard(canvas: Canvas, rect: RectF, label: String, value: String, bgColor: String) {
        canvas.drawRoundRect(rect, 30f, 30f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor(bgColor)
        })
        canvas.drawText(label, rect.left + 22f, rect.top + 48f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#64748B")
            textSize = 26f
        })
        canvas.drawText(value, rect.left + 22f, rect.top + 108f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#111827")
            textSize = if (value.length > 8) 30f else 40f
            isFakeBoldText = true
        })
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

    private fun drawPoseOverlay(canvas: Canvas, target: RectF, poseEstimate: PoseEstimate, sourceWidth: Float, sourceHeight: Float) {
        val scale = maxOf(target.width() / sourceWidth, target.height() / sourceHeight)
        val drawWidth = sourceWidth * scale
        val drawHeight = sourceHeight * scale
        val left = target.left + (target.width() - drawWidth) / 2f
        val top = target.top + (target.height() - drawHeight) / 2f

        val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#34D399")
            strokeWidth = 7f
            style = Style.STROKE
        }
        val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#F97316")
            style = Style.FILL
        }
        for ((start, end) in poseEstimate.skeletonEdges) {
            val p1 = poseEstimate.keypoints.getOrNull(start) ?: continue
            val p2 = poseEstimate.keypoints.getOrNull(end) ?: continue
            if (p1.confidence < 0.05f || p2.confidence < 0.05f) continue
            canvas.drawLine(
                left + p1.x * scale,
                top + p1.y * scale,
                left + p2.x * scale,
                top + p2.y * scale,
                linePaint
            )
        }
        for (point in poseEstimate.keypoints) {
            if (point.confidence < 0.05f) continue
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
        if (current.isNotEmpty()) lines += current.toString()
        return lines
    }

    private fun displaySportType(type: String): String = when (type) {
        "squat" -> "深蹲"
        "push_up" -> "俯卧撑"
        "sit_up" -> "仰卧起坐"
        else -> "运动"
    }

    private fun displayQuality(quality: String): String = when (quality) {
        "excellent" -> "优秀"
        "good" -> "良好"
        "fair" -> "一般"
        "needs_attention" -> "需优化"
        else -> "待评估"
    }
}
