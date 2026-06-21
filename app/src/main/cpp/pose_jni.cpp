#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define POSE_LOG_TAG "AI_SPORT_POSE"
#define POSE_LOGI(...) __android_log_print(ANDROID_LOG_INFO, POSE_LOG_TAG, __VA_ARGS__)
#define POSE_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, POSE_LOG_TAG, __VA_ARGS__)

namespace {

std::string EscapeJson(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size() + 16);
    for (char ch : input) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += ch; break;
        }
    }
    return escaped;
}

std::string JStringToStdString(JNIEnv* env, jstring value) {
    if (value == nullptr) return "";
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return "";
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

struct PoseContext {
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
};

std::string BuildErrorJson(const std::string& error) {
    std::ostringstream oss;
    oss << "{"
        << "\"ok\":false,"
        << "\"error\":\"" << EscapeJson(error) << "\""
        << "}";
    return oss.str();
}

struct TensorShapeInfo {
    std::vector<int> dims;
    bool nchw = true;
    int width = 0;
    int height = 0;
    int channels = 0;
};

TensorShapeInfo GetInputInfo(MNN::Tensor* input) {
    TensorShapeInfo info;
    if (input == nullptr) return info;
    info.dims = input->shape();
    if (info.dims.size() == 4) {
        if (info.dims[1] == 3) {
            info.nchw = true;
            info.channels = info.dims[1];
            info.height = info.dims[2];
            info.width = info.dims[3];
        } else if (info.dims[3] == 3) {
            info.nchw = false;
            info.channels = info.dims[3];
            info.height = info.dims[1];
            info.width = info.dims[2];
        }
    }
    return info;
}

int OutputLastDim(const std::vector<int>& dims) {
    return dims.empty() ? 0 : dims.back();
}

int OutputMiddleDim(const std::vector<int>& dims) {
    return dims.size() >= 2 ? dims[1] : 0;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_aisport_pose_NativePoseEstimator_nativeInitPose(
    JNIEnv* env,
    jobject,
    jstring modelPath
) {
    const std::string model = JStringToStdString(env, modelPath);
    if (model.empty()) {
        return 0;
    }
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model.c_str()), MNN::Interpreter::destroy);
    if (!net) {
        POSE_LOGE("createFromFile failed: %s", model.c_str());
        return 0;
    }
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    auto session = net->createSession(config);
    if (!session) {
        POSE_LOGE("createSession failed");
        return 0;
    }
    auto* ctx = new PoseContext();
    ctx->net = net;
    ctx->session = session;
    POSE_LOGI("Pose model initialized: %s", model.c_str());
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jstring JNICALL
Java_com_aisport_pose_NativePoseEstimator_nativeInferPose(
    JNIEnv* env,
    jobject,
    jlong handle,
    jfloatArray inputData,
    jint inputWidth,
    jint inputHeight
) {
    auto* ctx = reinterpret_cast<PoseContext*>(handle);
    if (ctx == nullptr || ctx->net == nullptr || ctx->session == nullptr) {
        return env->NewStringUTF(BuildErrorJson("invalid pose context").c_str());
    }
    auto* deviceInput = ctx->net->getSessionInput(ctx->session, nullptr);
    if (deviceInput == nullptr) {
        return env->NewStringUTF(BuildErrorJson("missing input tensor").c_str());
    }
    const auto inputInfo = GetInputInfo(deviceInput);
    if (inputInfo.width <= 0 || inputInfo.height <= 0 || inputInfo.channels != 3) {
        return env->NewStringUTF(BuildErrorJson("unsupported input shape").c_str());
    }
    const jsize expectedSize = inputInfo.width * inputInfo.height * inputInfo.channels;
    if (env->GetArrayLength(inputData) != expectedSize) {
        return env->NewStringUTF(BuildErrorJson("unexpected input float array size").c_str());
    }

    std::vector<float> hostData(expectedSize);
    env->GetFloatArrayRegion(inputData, 0, expectedSize, hostData.data());

    MNN::Tensor hostTensor(deviceInput, inputInfo.nchw ? MNN::Tensor::CAFFE : MNN::Tensor::TENSORFLOW);
    ::memcpy(hostTensor.host<float>(), hostData.data(), expectedSize * sizeof(float));
    if (!deviceInput->copyFromHostTensor(&hostTensor)) {
        return env->NewStringUTF(BuildErrorJson("copyFromHostTensor failed").c_str());
    }

    ctx->net->runSession(ctx->session);
    auto outputMap = ctx->net->getSessionOutputAll(ctx->session);
    if (outputMap.empty()) {
        return env->NewStringUTF(BuildErrorJson("no output tensors").c_str());
    }

    auto outputIt = outputMap.begin();
    MNN::Tensor* deviceOutput = outputIt->second;
    MNN::Tensor outputHost(deviceOutput, MNN::Tensor::CAFFE);
    deviceOutput->copyToHostTensor(&outputHost);
    const auto outShape = outputHost.shape();
    const float* out = outputHost.host<float>();
    if (out == nullptr) {
        return env->NewStringUTF(BuildErrorJson("output host tensor is null").c_str());
    }

    int numDet = 0;
    int numFeat = 0;
    bool layoutNDF = false;  // [N, D, F] or [D, F]
    if (outShape.size() == 3) {
        if (outShape[2] == 57) {
            numDet = outShape[1];
            numFeat = outShape[2];
            layoutNDF = false;  // [1, numDet, 57]
        } else if (outShape[1] == 57) {
            numDet = outShape[2];
            numFeat = outShape[1];
            layoutNDF = true;   // [1, 57, numDet]
        }
    } else if (outShape.size() == 2 && outShape[1] == 57) {
        numDet = outShape[0];
        numFeat = outShape[1];
        layoutNDF = false;
    }
    if (numDet <= 0 || numFeat != 57) {
        std::ostringstream err;
        err << "unsupported output shape:";
        for (int d : outShape) err << " " << d;
        return env->NewStringUTF(BuildErrorJson(err.str()).c_str());
    }

    auto getValue = [&](int detIndex, int featIndex) -> float {
        if (outShape.size() == 2) {
            return out[detIndex * numFeat + featIndex];
        }
        if (!layoutNDF) {
            return out[detIndex * numFeat + featIndex];
        }
        return out[featIndex * numDet + detIndex];
    };

    int bestIndex = -1;
    float bestScore = -1.0f;
    for (int i = 0; i < numDet; ++i) {
        const float score = getValue(i, 4);
        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }
    if (bestIndex < 0) {
        return env->NewStringUTF(BuildErrorJson("no detection found").c_str());
    }

    std::ostringstream oss;
    oss << "{"
        << "\"ok\":true,"
        << "\"score\":" << bestScore << ","
        << "\"class_id\":" << getValue(bestIndex, 5) << ","
        << "\"box\":["
        << getValue(bestIndex, 0) << ","
        << getValue(bestIndex, 1) << ","
        << getValue(bestIndex, 2) << ","
        << getValue(bestIndex, 3) << "],"
        << "\"keypoints\":[";
    for (int k = 0; k < 17; ++k) {
        const int base = 6 + k * 3;
        if (k > 0) oss << ",";
        oss << "["
            << getValue(bestIndex, base) << ","
            << getValue(bestIndex, base + 1) << ","
            << getValue(bestIndex, base + 2) << "]";
    }
    oss << "],"
        << "\"input_size\":[" << inputWidth << "," << inputHeight << "]"
        << "}";
    return env->NewStringUTF(oss.str().c_str());
}

JNIEXPORT void JNICALL
Java_com_aisport_pose_NativePoseEstimator_nativeReleasePose(
    JNIEnv*,
    jobject,
    jlong handle
) {
    auto* ctx = reinterpret_cast<PoseContext*>(handle);
    delete ctx;
}

}
