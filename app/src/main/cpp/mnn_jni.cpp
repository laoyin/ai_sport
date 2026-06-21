#include <jni.h>
#include <android/log.h>
#include <sys/stat.h>

#include <cerrno>
#include <memory>
#include <sstream>
#include <string>

#ifdef HAS_MNN
#include <llm/llm.hpp>
#endif

#define LOG_TAG "AI_SPORT_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace {

std::string JStringToStdString(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return "";
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return "";
    }
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

std::string JsonEscape(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size() + 16);
    for (char ch : input) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

std::string BuildErrorJson(const std::string& error) {
    return std::string("{\"sport_type\":\"\",\"summary_title\":\"分析失败\",\"repetition_count\":0,") +
           "\"pose_quality\":\"unknown\",\"highlight\":\"\",\"risk_tip\":\"\",\"confidence\":0.0,\"error\":\"" +
           JsonEscape(error) + "\"}";
}

std::string GetParentDir(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(0, pos);
}

bool EnsureDirectory(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    std::string current;
    current.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        current.push_back(path[i]);
        if (path[i] != '/' || current.size() == 1) {
            continue;
        }
        if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

#ifdef HAS_MNN
std::string BuildRuntimeConfigJson(const std::string& tmpPath) {
    std::ostringstream oss;
    oss << "{"
        << "\"tmp_path\":\"" << JsonEscape(tmpPath) << "\","
        << "\"async\":false,"
        << "\"use_mmap\":false,"
        << "\"backend\":\"cpu\","
        << "\"precision\":\"low\""
        << "}";
    return oss.str();
}

std::string BuildImagePrompt(const std::string& imagePath, const std::string& prompt) {
    std::ostringstream oss;
    oss << "<img>" << imagePath << "</img>\n" << prompt;
    return oss.str();
}
#endif

}  // namespace

#ifdef HAS_MNN
struct MnnContext {
    std::unique_ptr<MNN::Transformer::Llm> llm;
    std::string configPath;
    std::string modelDir;
    std::string tmpDir;
    int maxNewTokens = 768;
};
#else
struct MnnContext {
    bool initialized = false;
};
#endif

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_aisport_engine_MnnEngine_nativeInit(JNIEnv* env, jobject, jstring modelPath) {
    const std::string configPath = JStringToStdString(env, modelPath);
#ifdef HAS_MNN
    if (configPath.empty()) {
        return 0;
    }

    auto* rawLlm = MNN::Transformer::Llm::createLLM(configPath);
    if (rawLlm == nullptr) {
        LOGE("createLLM failed for config: %s", configPath.c_str());
        return 0;
    }

    auto ctx = std::make_unique<MnnContext>();
    ctx->llm.reset(rawLlm);
    ctx->configPath = configPath;
    ctx->modelDir = GetParentDir(configPath);
    ctx->tmpDir = ctx->modelDir + "/tmp";
    EnsureDirectory(ctx->tmpDir);
    ctx->llm->set_config(BuildRuntimeConfigJson(ctx->tmpDir));

    if (!ctx->llm->load()) {
        LOGE("llm->load() failed for config: %s", configPath.c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(ctx.release());
#else
    LOGW("Stub native init for config: %s", configPath.c_str());
    return 0;
#endif
}

JNIEXPORT jstring JNICALL
Java_com_aisport_engine_MnnEngine_nativeRunInference(
    JNIEnv* env,
    jobject,
    jlong handle,
    jstring imagePath,
    jstring prompt
) {
    auto* ctx = reinterpret_cast<MnnContext*>(handle);
    if (ctx == nullptr) {
        return env->NewStringUTF(BuildErrorJson("invalid native handle").c_str());
    }

    const std::string image = JStringToStdString(env, imagePath);
    const std::string promptText = JStringToStdString(env, prompt);
#ifdef HAS_MNN
    if (ctx->llm == nullptr) {
        return env->NewStringUTF(BuildErrorJson("llm context not initialized").c_str());
    }
    if (image.empty()) {
        return env->NewStringUTF(BuildErrorJson("image path is empty").c_str());
    }

    try {
        ctx->llm->reset();
        std::ostringstream output;
        ctx->llm->response(BuildImagePrompt(image, promptText), &output, nullptr, ctx->maxNewTokens);
        const std::string result = output.str();
        if (result.empty()) {
            return env->NewStringUTF(BuildErrorJson("empty llm response").c_str());
        }
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        return env->NewStringUTF(BuildErrorJson(e.what()).c_str());
    } catch (...) {
        return env->NewStringUTF(BuildErrorJson("unknown native exception").c_str());
    }
#else
    return env->NewStringUTF(BuildErrorJson("MNN native layer unavailable").c_str());
#endif
}

JNIEXPORT void JNICALL
Java_com_aisport_engine_MnnEngine_nativeRelease(JNIEnv*, jobject, jlong handle) {
    auto* ctx = reinterpret_cast<MnnContext*>(handle);
    delete ctx;
}

}
