package com.aisport.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import kotlin.math.max

class RealtimeFrameAnalyzer(
    private val onFrame: (Bitmap, Long) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        try {
            val bitmap = image.toBitmap() ?: return
            onFrame(bitmap, System.currentTimeMillis())
        } finally {
            image.close()
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        val nv21 = yuv420888ToNv21(this)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val stream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 80, stream)
        val bytes = stream.toByteArray()
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
        val rotated = if (imageInfo.rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }
            Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
        } else {
            decoded
        }
        return downscale(rotated, 960)
    }

    private fun downscale(bitmap: Bitmap, maxEdge: Int): Bitmap {
        val longEdge = max(bitmap.width, bitmap.height)
        if (longEdge <= maxEdge) {
            return bitmap
        }
        val scale = maxEdge.toFloat() / longEdge.toFloat()
        return Bitmap.createScaledBitmap(
            bitmap,
            (bitmap.width * scale).toInt(),
            (bitmap.height * scale).toInt(),
            true
        )
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val ySize = yPlane.buffer.remaining()
        val uSize = uPlane.buffer.remaining()
        val vSize = vPlane.buffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yPlane.buffer.get(nv21, 0, ySize)

        val rowStride = vPlane.rowStride
        val pixelStride = vPlane.pixelStride
        val width = image.width
        val height = image.height
        var offset = ySize

        val vBuffer = vPlane.buffer
        val uBuffer = uPlane.buffer
        val chromaHeight = height / 2
        val chromaWidth = width / 2

        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vuPos = row * rowStride + col * pixelStride
                nv21[offset++] = vBuffer.get(vuPos)
                nv21[offset++] = uBuffer.get(vuPos)
            }
        }
        return nv21
    }
}
