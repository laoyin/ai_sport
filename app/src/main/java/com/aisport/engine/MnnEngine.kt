package com.aisport.engine

import android.content.Context
import android.util.Log
import java.io.File

class MnnEngine(private val context: Context) {

    companion object {
        private const val TAG = "AiSportMnnEngine"

        init {
            try {
                System.loadLibrary("aisport_native")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library", e)
            }
        }
    }

    private var nativeHandle: Long = 0L
    private var initialized = false

    fun loadModelBundle(assetDirName: String, configFileName: String = "config.json"): Boolean {
        return try {
            val bundleDir = copyAssetDirectoryToInternalStorage(assetDirName) ?: return false
            val configFile = File(bundleDir, configFileName)
            if (!configFile.exists()) {
                Log.e(TAG, "Missing config file: ${configFile.absolutePath}")
                return false
            }
            nativeHandle = nativeInit(configFile.absolutePath)
            initialized = nativeHandle != 0L
            initialized
        } catch (t: Throwable) {
            Log.e(TAG, "loadModelBundle failed", t)
            false
        }
    }

    fun runInference(imagePath: String, prompt: String): String? {
        if (!initialized || nativeHandle == 0L) {
            return null
        }
        return try {
            nativeRunInference(nativeHandle, imagePath, prompt)
        } catch (t: Throwable) {
            Log.e(TAG, "runInference failed", t)
            null
        }
    }

    fun isReady(): Boolean = initialized && nativeHandle != 0L

    fun release() {
        if (nativeHandle != 0L) {
            try {
                nativeRelease(nativeHandle)
            } catch (t: Throwable) {
                Log.w(TAG, "nativeRelease failed", t)
            }
        }
        nativeHandle = 0L
        initialized = false
    }

    private fun copyAssetDirectoryToInternalStorage(assetDirName: String): File? {
        val targetDir = File(context.filesDir, assetDirName)
        return try {
            copyAssetDirectoryRecursive(assetDirName, targetDir)
            targetDir
        } catch (t: Throwable) {
            Log.e(TAG, "copyAssetDirectoryToInternalStorage failed for $assetDirName", t)
            null
        }
    }

    private fun copyAssetDirectoryRecursive(assetPath: String, dest: File) {
        val children = context.assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            copyAssetToPath(assetPath, dest)
            return
        }

        if (!dest.exists()) {
            dest.mkdirs()
        }
        for (child in children) {
            copyAssetDirectoryRecursive("$assetPath/$child", File(dest, child))
        }
    }

    private fun copyAssetToPath(assetPath: String, dest: File) {
        if (dest.exists() && dest.length() > 0L) {
            return
        }
        dest.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            dest.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }

    private external fun nativeInit(modelPath: String): Long
    private external fun nativeRunInference(handle: Long, imagePath: String, prompt: String): String?
    private external fun nativeRelease(handle: Long)
}
