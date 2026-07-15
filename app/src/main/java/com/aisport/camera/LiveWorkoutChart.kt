package com.aisport.camera

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

@Composable
fun LiveWorkoutChart(
    values: List<Float>,
    modifier: Modifier = Modifier
) {
    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(140.dp)
            .background(Color(0xFFF8FAFC), RoundedCornerShape(18.dp))
    ) {
        if (values.isEmpty()) {
            drawLine(
                color = Color(0xFFE2E8F0),
                start = Offset(0f, size.height / 2f),
                end = Offset(size.width, size.height / 2f),
                strokeWidth = 4f
            )
            return@Canvas
        }

        val maxValue = values.maxOrNull()?.coerceAtLeast(1f) ?: 1f
        val path = Path()
        val fillPath = Path()
        values.forEachIndexed { index, value ->
            val x = if (values.size == 1) 0f else size.width * index / (values.size - 1).toFloat()
            val normalized = (value / maxValue).coerceIn(0f, 1f)
            val y = size.height - normalized * size.height
            if (index == 0) {
                path.moveTo(x, y)
                fillPath.moveTo(x, size.height)
                fillPath.lineTo(x, y)
            } else {
                path.lineTo(x, y)
                fillPath.lineTo(x, y)
            }
        }
        fillPath.lineTo(size.width, size.height)
        fillPath.close()
        drawPath(
            path = fillPath,
            brush = Brush.verticalGradient(
                colors = listOf(Color(0x446D4CC3), Color(0x116D4CC3), Color.Transparent),
                startY = 0f,
                endY = size.height
            ),
            style = Fill
        )
        drawPath(
            path = path,
            color = Color(0xFF6D4CC3),
            style = Stroke(width = 6f, cap = StrokeCap.Round)
        )
        val lastIndex = values.lastIndex
        val lastValue = values.lastOrNull() ?: 0f
        val cx = if (lastIndex <= 0) 0f else size.width
        val cy = size.height - (lastValue / maxValue).coerceIn(0f, 1f) * size.height
        drawCircle(Color(0x336D4CC3), radius = 18f, center = Offset(cx, cy))
        drawCircle(Color(0xFF6D4CC3), radius = 8f, center = Offset(cx, cy))
    }
}
