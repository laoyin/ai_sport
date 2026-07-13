plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

import java.io.File

val workspaceRoot = rootDir.parentFile
val qwenVlBundleDir = File(workspaceRoot, "qwen3-vl-mnn")
val generatedAssetRoot = File(buildDir, "generated/ai_sport_assets")
val generatedQwenAssetDir = File(generatedAssetRoot, "qwen3-vl-mnn")
val yoloPoseModelFile = File(rootDir, "yolo26n-pose.mnn")
val generatedPoseAssetDir = File(generatedAssetRoot, "pose")
val repCounterModelFile = File(rootDir, "rep_counter.mnn")
val generatedRepAssetDir = File(generatedAssetRoot, "rep")

val mnnRoot = File(workspaceRoot, "MNN/MNN")
val mnnBuildDirOverride = providers.gradleProperty("mnnBuildDir")
    .orElse(providers.environmentVariable("MNN_BUILD_DIR"))
    .orNull

val mnnBuildDirCandidates = listOf(
    "build-android-arm64-4b-sme2-on",
    "build-android-arm64-4b-sme2-off",
    "build-android-arm64-4b-onnx",
    "build-android-arm64-4b",
    "build-android-arm64"
).map { File(mnnRoot, it) }

fun resolveMnnBuildDir(): File {
    if (!mnnBuildDirOverride.isNullOrBlank()) {
        val override = if (File(mnnBuildDirOverride).isAbsolute) {
            File(mnnBuildDirOverride)
        } else {
            File(mnnRoot, mnnBuildDirOverride)
        }
        return override
    }
    return mnnBuildDirCandidates.firstOrNull { File(it, "libMNN.so").exists() }
        ?: mnnBuildDirCandidates.first()
}

val selectedMnnBuildDir = resolveMnnBuildDir()
val selectedMnnBuildDirPath = selectedMnnBuildDir.absolutePath.replace("\\", "/")

val syncQwenVlAssets by tasks.registering(Copy::class) {
    group = "build setup"
    description = "Sync Qwen3-VL-MNN bundle into generated Android assets."
    from(qwenVlBundleDir)
    into(generatedQwenAssetDir)
    doFirst {
        generatedQwenAssetDir.mkdirs()
        println("Syncing Qwen3-VL assets from: ${qwenVlBundleDir.absolutePath}")
    }
}

val syncPoseAssets by tasks.registering(Copy::class) {
    group = "build setup"
    description = "Sync YOLO26 pose MNN model into generated Android assets."
    from(yoloPoseModelFile)
    into(generatedPoseAssetDir)
    doFirst {
        generatedPoseAssetDir.mkdirs()
        println("Syncing pose model from: ${yoloPoseModelFile.absolutePath}")
    }
}

val syncRepCounterAssets by tasks.registering(Copy::class) {
    group = "build setup"
    description = "Sync rep counter MNN model into generated Android assets."
    from(repCounterModelFile)
    into(generatedRepAssetDir)
    doFirst {
        generatedRepAssetDir.mkdirs()
        println("Syncing rep counter model from: ${repCounterModelFile.absolutePath}")
    }
}

val syncSelectedMnnJniLibs by tasks.registering(Copy::class) {
    group = "build setup"
    description = "Sync libMNN/libMNN_Express/libllm from selected MNN build directory into jniLibs."

    val abiDir = File(projectDir, "src/main/jniLibs/arm64-v8a")
    from(selectedMnnBuildDir) {
        include("libMNN.so")
        include("libMNN_Express.so")
        include("libllm.so")
    }
    into(abiDir)
    doFirst {
        abiDir.mkdirs()
        println("Syncing MNN runtime from: $selectedMnnBuildDirPath")
    }
}

android {
    namespace = "com.aisport"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.aisport"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17")
                arguments += listOf("-DMNN_BUILD_DIR_OVERRIDE=$selectedMnnBuildDirPath")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.11"
    }

    androidResources {
        noCompress += listOf("mnn", "weight", "json", "txt")
    }

    sourceSets {
        getByName("main") {
            assets.srcDir(generatedAssetRoot)
            jniLibs.srcDir("src/main/jniLibs")
        }
    }
}

tasks.matching { it.name == "preBuild" }.configureEach {
    dependsOn(syncQwenVlAssets)
    dependsOn(syncPoseAssets)
    dependsOn(syncRepCounterAssets)
    dependsOn(syncSelectedMnnJniLibs)
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("com.google.android.material:material:1.11.0")
    implementation(platform("androidx.compose:compose-bom:2024.02.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
