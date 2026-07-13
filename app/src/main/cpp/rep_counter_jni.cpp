#include <jni.h>
#include <android/log.h>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define REP_LOG_TAG "AI_SPORT_REP"
#define REP_LOGI(...) __android_log_print(ANDROID_LOG_INFO, REP_LOG_TAG, __VA_ARGS__)
#define REP_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, REP_LOG_TAG, __VA_ARGS__)

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

std::string BuildErrorJson(const std::string& error) {
    std::ostringstream oss;
    oss << "{"
        << "\"ok\":false,"
        << "\"error\":\"" << EscapeJson(error) << "\""
        << "}";
    return oss.str();
}

struct RepContext {
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
};

bool IsStageShape(const std::vector<int>& shape) {
    return shape.size() == 3 && shape[0] == 1 && shape[1] == 32 && shape[2] == 4;
}

bool IsActionShape(const std::vector<int>& shape) {
    return shape.size() == 2 && shape[0] == 1 && shape[1] == 3;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_aisport_rep_NativeRepCounter_nativeInitRepCounter(
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
        REP_LOGE("createFromFile failed: %s", model.c_str());
        return 0;
    }
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    auto session = net->createSession(config);
    if (!session) {
        REP_LOGE("createSession failed");
        return 0;
    }
    auto* ctx = new RepContext();
    ctx->net = net;
    ctx->session = session;
    REP_LOGI("Rep counter initialized: %s", model.c_str());
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jstring JNICALL
Java_com_aisport_rep_NativeRepCounter_nativeInferRepCounter(
    JNIEnv* env,
    jobject,
    jlong handle,
    jfloatArray inputData,
    jint seqLen,
    jint featureDim
) {
    auto* ctx = reinterpret_cast<RepContext*>(handle);
    if (ctx == nullptr || ctx->net == nullptr || ctx->session == nullptr) {
        return env->NewStringUTF(BuildErrorJson("invalid rep context").c_str());
    }

    auto* deviceInput = ctx->net->getSessionInput(ctx->session, nullptr);
    if (deviceInput == nullptr) {
        return env->NewStringUTF(BuildErrorJson("missing input tensor").c_str());
    }

    const auto inputShape = deviceInput->shape();
    if (inputShape.size() != 3 || inputShape[0] != 1 || inputShape[1] != seqLen || inputShape[2] != featureDim) {
        std::ostringstream err;
        err << "unexpected input shape:";
        for (int d : inputShape) err << " " << d;
        return env->NewStringUTF(BuildErrorJson(err.str()).c_str());
    }

    const jsize expectedSize = seqLen * featureDim;
    if (env->GetArrayLength(inputData) != expectedSize) {
        return env->NewStringUTF(BuildErrorJson("unexpected input float array size").c_str());
    }

    std::vector<float> hostData(expectedSize);
    env->GetFloatArrayRegion(inputData, 0, expectedSize, hostData.data());

    MNN::Tensor hostTensor(deviceInput, deviceInput->getDimensionType());
    ::memcpy(hostTensor.host<float>(), hostData.data(), expectedSize * sizeof(float));
    if (!deviceInput->copyFromHostTensor(&hostTensor)) {
        return env->NewStringUTF(BuildErrorJson("copyFromHostTensor failed").c_str());
    }

    ctx->net->runSession(ctx->session);
    auto outputMap = ctx->net->getSessionOutputAll(ctx->session);
    if (outputMap.empty()) {
        return env->NewStringUTF(BuildErrorJson("no output tensors").c_str());
    }

    MNN::Tensor* stageTensor = nullptr;
    MNN::Tensor* actionTensor = nullptr;
    for (const auto& item : outputMap) {
        MNN::Tensor* tensor = item.second;
        const auto shape = tensor->shape();
        if (IsStageShape(shape)) {
            stageTensor = tensor;
        } else if (IsActionShape(shape)) {
            actionTensor = tensor;
        }
    }

    if (stageTensor == nullptr || actionTensor == nullptr) {
        std::ostringstream err;
        err << "failed to locate outputs";
        for (const auto& item : outputMap) {
            err << " [" << item.first << ":";
            for (int d : item.second->shape()) err << " " << d;
            err << "]";
        }
        return env->NewStringUTF(BuildErrorJson(err.str()).c_str());
    }

    MNN::Tensor stageHost(stageTensor, stageTensor->getDimensionType());
    MNN::Tensor actionHost(actionTensor, actionTensor->getDimensionType());
    stageTensor->copyToHostTensor(&stageHost);
    actionTensor->copyToHostTensor(&actionHost);

    const float* stage = stageHost.host<float>();
    const float* action = actionHost.host<float>();
    if (stage == nullptr || action == nullptr) {
        return env->NewStringUTF(BuildErrorJson("output host tensor is null").c_str());
    }

    std::ostringstream oss;
    oss << "{"
        << "\"ok\":true,"
        << "\"action_logits\":[";
    for (int i = 0; i < 3; ++i) {
        if (i > 0) oss << ",";
        oss << action[i];
    }
    oss << "],\"stage_logits\":[";
    for (int t = 0; t < 32; ++t) {
        if (t > 0) oss << ",";
        oss << "[";
        for (int c = 0; c < 4; ++c) {
            if (c > 0) oss << ",";
            oss << stage[t * 4 + c];
        }
        oss << "]";
    }
    oss << "]"
        << "}";
    return env->NewStringUTF(oss.str().c_str());
}

JNIEXPORT void JNICALL
Java_com_aisport_rep_NativeRepCounter_nativeReleaseRepCounter(
    JNIEnv*,
    jobject,
    jlong handle
) {
    auto* ctx = reinterpret_cast<RepContext*>(handle);
    delete ctx;
}

}
