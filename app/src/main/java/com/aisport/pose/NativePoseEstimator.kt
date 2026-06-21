package com.aisport.pose

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONObject
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class NativePoseEstimator(private val context: Context) : PoseEstimator {

    companion object {
        private const val TAG = "NativePoseEstimator"

        init {
            try {
                System.loadLibrary("aisport_native")
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to load pose native library", t)
            }
        }
    }

    private var nativeHandle: Long = 0L
    private var initialized = false

    fun loadModel(assetPath: String = "pose/yolo26n-pose.mnn"): Boolean {
        return try {
            val file = copyAssetToInternal(assetPath) ?: return false
            nativeHandle = nativeInitPose(file.absolutePath)
            initialized = nativeHandle != 0L
            initialized
        } catch (t: Throwable) {
            Log.e(TAG, "loadModel failed", t)
            false
        }
    }

    override fun estimate(bitmap: Bitmap): PoseEstimate {
        if (!initialized || nativeHandle == 0L) {
            return fallback(bitmap, "YOLO-Pose Native unavailable")
        }
        val prepared = letterbox(bitmap, 640)
        val raw = nativeInferPose(nativeHandle, prepared.tensor, prepared.size, prepared.size) ?: return fallback(bitmap, "native null")
        return try {
            val obj = JSONObject(raw)
            if (!obj.optBoolean("ok")) {
                fallback(bitmap, obj.optString("error", "native failed"))
            } else {
                val score = obj.optDouble("score", 0.0).toFloat()
                val keypoints = obj.getJSONArray("keypoints")
                val posePoints = buildList {
                    for (i in 0 until keypoints.length()) {
                        val arr = keypoints.getJSONArray(i)
                        val x = unletterboxX(arr.optDouble(0).toFloat(), prepared)
                        val y = unletterboxY(arr.optDouble(1).toFloat(), prepared)
                        val conf = arr.optDouble(2).toFloat()
                        add(PosePoint(x, y, conf))
                    }
                }
                val quality = when {
                    score >= 0.75f -> "excellent"
                    score >= 0.6f -> "good"
                    score >= 0.45f -> "fair"
                    else -> "needs_attention"
                }
                PoseEstimate(
                    score = score,
                    stageHint = "关键点已由 MNN 模型推理输出",
                    qualityHint = quality,
                    keypoints = posePoints,
                    skeletonEdges = cocoEdges,
                    engineName = "YOLO26n-pose MNN",
                    placeholder = false
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "estimate parse failed", t)
            fallback(bitmap, "parse failed")
        }
    }

    fun release() {
        if (nativeHandle != 0L) {
            try {
                nativeReleasePose(nativeHandle)
            } catch (_: Throwable) {
            }
        }
        nativeHandle = 0L
        initialized = false
    }

    private fun copyAssetToInternal(assetPath: String): File? {
        val fileName = assetPath.substringAfterLast('/')
        val target = File(context.filesDir, fileName)
        if (target.exists() && target.length() > 0L) {
            return target
        }
        return try {
            context.assets.open(assetPath).use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            target
        } catch (t: Throwable) {
            Log.e(TAG, "copyAssetToInternal failed for $assetPath", t)
            null
        }
    }

    private data class PreparedInput(
        val tensor: FloatArray,
        val size: Int,
        val scale: Float,
        val padX: Float,
        val padY: Float
    )

    private fun letterbox(bitmap: Bitmap, targetSize: Int): PreparedInput {
        val scale = min(targetSize / bitmap.width.toFloat(), targetSize / bitmap.height.toFloat())
        val resizedW = max(1, (bitmap.width * scale).roundToInt())
        val resizedH = max(1, (bitmap.height * scale).roundToInt())
        val padX = (targetSize - resizedW) / 2f
        val padY = (targetSize - resizedH) / 2f
        val resized = Bitmap.createScaledBitmap(bitmap, resizedW, resizedH, true)
        val pixels = IntArray(targetSize * targetSize) { 0x727272 }
        val resizedPixels = IntArray(resizedW * resizedH)
        resized.getPixels(resizedPixels, 0, resizedW, 0, 0, resizedW, resizedH)
        val left = padX.roundToInt()
        val top = padY.roundToInt()
        for (y in 0 until resizedH) {
            val srcOffset = y * resizedW
            val dstOffset = (top + y) * targetSize + left
            System.arraycopy(resizedPixels, srcOffset, pixels, dstOffset, resizedW)
        }
        val tensor = FloatArray(3 * targetSize * targetSize)
        val plane = targetSize * targetSize
        for (i in pixels.indices) {
            val c = pixels[i]
            tensor[i] = ((c shr 16) and 0xFF) / 255f
            tensor[plane + i] = ((c shr 8) and 0xFF) / 255f
            tensor[plane * 2 + i] = (c and 0xFF) / 255f
        }
        return PreparedInput(tensor, targetSize, scale, padX, padY)
    }

    private fun unletterboxX(x: Float, prepared: PreparedInput): Float {
        return ((x - prepared.padX) / prepared.scale).coerceAtLeast(0f)
    }

    private fun unletterboxY(y: Float, prepared: PreparedInput): Float {
        return ((y - prepared.padY) / prepared.scale).coerceAtLeast(0f)
    }

    private fun fallback(bitmap: Bitmap, reason: String): PoseEstimate {
        Log.w(TAG, "Falling back to stub pose estimate: $reason")
        return YoloPoseEstimatorBridge().estimate(bitmap)
    }

    private external fun nativeInitPose(modelPath: String): Long
    private external fun nativeInferPose(handle: Long, inputData: FloatArray, inputWidth: Int, inputHeight: Int): String?
    private external fun nativeReleasePose(handle: Long)

    private val cocoEdges = listOf(
        0 to 1, 0 to 2, 1 to 3, 2 to 4,
        5 to 6, 5 to 7, 7 to 9, 6 to 8, 8 to 10,
        5 to 11, 6 to 12, 11 to 12,
        11 to 13, 13 to 15, 12 to 14, 14 to 16
    )
}
