package com.aisport.rep

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class RepCounterResult(
    val actionLogits: FloatArray,
    val stageLogits: Array<FloatArray>
) {
    fun actionId(): Int = actionLogits.indices.maxByOrNull { actionLogits[it] } ?: 0

    fun stageIds(): IntArray = IntArray(stageLogits.size) { index ->
        val logits = stageLogits[index]
        logits.indices.maxByOrNull { logits[it] } ?: 0
    }
}

class NativeRepCounter(private val context: Context) {

    companion object {
        private const val TAG = "NativeRepCounter"

        const val SEQ_LEN = 32
        const val FEATURE_DIM = 55

        init {
            try {
                System.loadLibrary("aisport_native")
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to load rep counter native library", t)
            }
        }
    }

    private var nativeHandle: Long = 0L
    private var initialized = false

    fun loadModel(assetPath: String = "rep/rep_counter.mnn"): Boolean {
        return try {
            val file = copyAssetToInternal(assetPath) ?: return false
            nativeHandle = nativeInitRepCounter(file.absolutePath)
            initialized = nativeHandle != 0L
            initialized
        } catch (t: Throwable) {
            Log.e(TAG, "loadModel failed", t)
            false
        }
    }

    fun inferWindow(
        windowFeatures: FloatArray,
        seqLen: Int = SEQ_LEN,
        featureDim: Int = FEATURE_DIM
    ): RepCounterResult? {
        if (!initialized || nativeHandle == 0L) {
            return null
        }
        if (windowFeatures.size != seqLen * featureDim) {
            Log.w(TAG, "Unexpected feature size=${windowFeatures.size}, expected=${seqLen * featureDim}")
            return null
        }
        val raw = nativeInferRepCounter(nativeHandle, windowFeatures, seqLen, featureDim) ?: return null
        return try {
            val obj = JSONObject(raw)
            if (!obj.optBoolean("ok")) {
                Log.w(TAG, "inferWindow failed: ${obj.optString("error")}")
                null
            } else {
                RepCounterResult(
                    actionLogits = parseFloatArray(obj.getJSONArray("action_logits")),
                    stageLogits = parseMatrix(obj.getJSONArray("stage_logits"))
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "inferWindow parse failed", t)
            null
        }
    }

    fun release() {
        if (nativeHandle != 0L) {
            try {
                nativeReleaseRepCounter(nativeHandle)
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

    private fun parseFloatArray(array: JSONArray): FloatArray {
        return FloatArray(array.length()) { index -> array.optDouble(index, 0.0).toFloat() }
    }

    private fun parseMatrix(array: JSONArray): Array<FloatArray> {
        return Array(array.length()) { rowIndex ->
            parseFloatArray(array.getJSONArray(rowIndex))
        }
    }

    private external fun nativeInitRepCounter(modelPath: String): Long
    private external fun nativeInferRepCounter(handle: Long, inputData: FloatArray, seqLen: Int, featureDim: Int): String?
    private external fun nativeReleaseRepCounter(handle: Long)
}
