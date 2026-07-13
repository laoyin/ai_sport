package com.aisport.training

import android.content.Context
import android.graphics.Bitmap
import com.aisport.exercise.ExerciseFrameSample
import com.aisport.poster.PoseDebugRenderer
import com.aisport.video.SampledPoseFrame
import com.aisport.video.VideoMotionSummary
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class AnnotationExportResult(
    val directory: File,
    val frameCount: Int
)

object AnnotationExport {

    private const val PUSH_UP_UP_ANGLE = 155f
    private const val PUSH_UP_DOWN_ANGLE = 100f
    private const val SQUAT_UP_ANGLE = 160f
    private const val SQUAT_DOWN_ANGLE = 105f

    fun exportVideoAnnotationPackage(
        context: Context,
        videoLabel: String,
        analysisMode: String,
        sampledPoseFrames: List<SampledPoseFrame>,
        frameSamples: List<ExerciseFrameSample>,
        motionSummary: VideoMotionSummary?
    ): AnnotationExportResult {
        require(sampledPoseFrames.size == frameSamples.size) {
            "Sampled pose frames and frame samples must have the same size"
        }

        val rootDir = File(context.getExternalFilesDir(null) ?: context.filesDir, "labels")
        val exportId = buildExportId(videoLabel, analysisMode, sampledPoseFrames)
        val exportDir = File(rootDir, exportId)
        val framesDir = File(exportDir, "frames")
        val overlayDir = File(exportDir, "overlay")
        framesDir.mkdirs()
        overlayDir.mkdirs()

        val samplesJson = JSONArray()
        sampledPoseFrames.zip(frameSamples).forEachIndexed { index, (poseFrame, sample) ->
            val baseName = String.format(Locale.US, "%04d_%06dms", index, poseFrame.timeMs)
            val frameFile = File(framesDir, "$baseName.jpg")
            val overlayFile = File(overlayDir, "${baseName}_pose.jpg")

            saveJpeg(poseFrame.bitmap, frameFile)
            saveJpeg(PoseDebugRenderer.renderOverlayOnly(poseFrame.bitmap, poseFrame.poseEstimate), overlayFile)

            samplesJson.put(
                JSONObject().apply {
                    put("video_id", exportId)
                    put("frame_idx", index)
                    put("time_ms", poseFrame.timeMs)
                    put("frame_image", "frames/${frameFile.name}")
                    put("pose_image", "overlay/${overlayFile.name}")
                    put("sport_type", "")
                    put("stage_label", "")
                    put("suggested_sport_type", resolveSuggestedSportType(analysisMode, motionSummary))
                    put("suggested_stage_label", resolveSuggestedStageLabel(sample, analysisMode, motionSummary))
                    put("score", poseFrame.poseEstimate.score)
                    put("quality_hint", poseFrame.poseEstimate.qualityHint)
                    put("engine_name", poseFrame.poseEstimate.engineName)
                    put("keypoints", buildKeypointsJson(poseFrame))
                    put("features", buildFeaturesJson(sample))
                }
            )
        }

        val manifest = JSONObject().apply {
            put("video_id", exportId)
            put("source_label", videoLabel)
            put("analysis_mode", analysisMode)
            put("exported_at", isoNow())
            put("frame_count", sampledPoseFrames.size)
            put("inferred_sport_type", motionSummary?.inferredSportType ?: "unknown")
            put("repetition_count", motionSummary?.repetitionCount ?: 0)
            put("confidence", motionSummary?.confidence ?: 0f)
        }

        File(exportDir, "samples.json").writeText(samplesJson.toString(2), Charsets.UTF_8)
        File(exportDir, "manifest.json").writeText(manifest.toString(2), Charsets.UTF_8)
        return AnnotationExportResult(exportDir, sampledPoseFrames.size)
    }

    private fun buildExportId(
        videoLabel: String,
        analysisMode: String,
        sampledPoseFrames: List<SampledPoseFrame>
    ): String {
        val sourceType = when {
            videoLabel.contains("相机") -> "camera_video"
            videoLabel.contains("视频") -> "album_video"
            videoLabel.contains("图片") -> "album_image"
            else -> "video"
        }
        val mode = when (analysisMode) {
            "push_up" -> "pushup"
            "squat" -> "squat"
            else -> "auto"
        }
        val firstMs = sampledPoseFrames.firstOrNull()?.timeMs ?: 0L
        val lastMs = sampledPoseFrames.lastOrNull()?.timeMs ?: 0L
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return "${sourceType}_${mode}_${firstMs}ms_${lastMs}ms_$ts"
    }

    private fun saveJpeg(bitmap: Bitmap, target: File) {
        FileOutputStream(target).use { output ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output)
        }
    }

    private fun buildKeypointsJson(frame: SampledPoseFrame): JSONArray {
        val result = JSONArray()
        frame.poseEstimate.keypoints.forEach { point ->
            result.put(
                JSONArray().apply {
                    put(point.x)
                    put(point.y)
                    put(point.confidence)
                }
            )
        }
        return result
    }

    private fun buildFeaturesJson(sample: ExerciseFrameSample): JSONObject {
        return JSONObject().apply {
            put("squat_angle", sample.squatAngle)
            put("pushup_angle", sample.pushupAngle)
            put("pushup_depth", sample.pushupDepth)
            put("torso_linearity", sample.torsoLinearity)
            put("torso_horizontal", sample.torsoHorizontal)
            put("torso_vertical", sample.torsoVertical)
            put("left_elbow_angle", sample.leftElbowAngle)
            put("right_elbow_angle", sample.rightElbowAngle)
            put("left_knee_angle", sample.leftKneeAngle)
            put("right_knee_angle", sample.rightKneeAngle)
            put("hip_to_shoulder_y_diff", sample.hipToShoulderYDiff)
            put("hip_to_wrist_y_diff", sample.hipToWristYDiff)
        }
    }

    private fun resolveSuggestedSportType(
        analysisMode: String,
        motionSummary: VideoMotionSummary?
    ): String {
        return when (analysisMode) {
            "push_up" -> "push_up"
            "squat" -> "squat"
            else -> motionSummary?.inferredSportType ?: "unknown"
        }
    }

    private fun resolveSuggestedStageLabel(
        sample: ExerciseFrameSample,
        analysisMode: String,
        motionSummary: VideoMotionSummary?
    ): String {
        return when (resolveSuggestedSportType(analysisMode, motionSummary)) {
            "push_up" -> when {
                sample.pushupAngle <= PUSH_UP_DOWN_ANGLE -> "down"
                sample.pushupAngle >= PUSH_UP_UP_ANGLE -> "up"
                else -> "transition"
            }
            "squat" -> when {
                sample.squatAngle <= SQUAT_DOWN_ANGLE -> "down"
                sample.squatAngle >= SQUAT_UP_ANGLE -> "up"
                else -> "transition"
            }
            else -> "background"
        }
    }

    private fun isoNow(): String {
        return SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date())
    }
}
