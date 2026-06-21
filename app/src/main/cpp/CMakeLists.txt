cmake_minimum_required(VERSION 3.22.1)
project("aisport_native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(
    WORKSPACE_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../"
    ABSOLUTE
)

set(MNN_ROOT "${WORKSPACE_ROOT}/MNN/MNN")
if(DEFINED MNN_BUILD_DIR_OVERRIDE)
    set(MNN_BUILD_DIR "${MNN_BUILD_DIR_OVERRIDE}")
else()
    set(MNN_BUILD_DIR "${MNN_ROOT}/build-android-arm64")
endif()
set(MNN_INCLUDE_DIR "${MNN_ROOT}/include")
set(MNN_LLM_INCLUDE_DIR "${MNN_ROOT}/transformers/llm/engine/include")
set(MNN_LIB_PATH "${MNN_BUILD_DIR}/libMNN.so")
set(MNN_EXPRESS_LIB_PATH "${MNN_BUILD_DIR}/libMNN_Express.so")
set(MNN_LLM_LIB_PATH "${MNN_BUILD_DIR}/libllm.so")

add_library(aisport_native SHARED
    mnn_jni.cpp
    pose_jni.cpp
)

set(MNN_READY TRUE)

if(NOT EXISTS "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp")
    message(WARNING "Missing MNN header: ${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp")
    set(MNN_READY FALSE)
endif()

if(NOT EXISTS "${MNN_LLM_INCLUDE_DIR}/llm/llm.hpp")
    message(WARNING "Missing MNN LLM header: ${MNN_LLM_INCLUDE_DIR}/llm/llm.hpp")
    set(MNN_READY FALSE)
endif()

if(NOT EXISTS "${MNN_LIB_PATH}")
    message(WARNING "Missing libMNN.so: ${MNN_LIB_PATH}")
    set(MNN_READY FALSE)
endif()

if(NOT EXISTS "${MNN_EXPRESS_LIB_PATH}")
    message(WARNING "Missing libMNN_Express.so: ${MNN_EXPRESS_LIB_PATH}")
    set(MNN_READY FALSE)
endif()

if(NOT EXISTS "${MNN_LLM_LIB_PATH}")
    message(WARNING "Missing libllm.so: ${MNN_LLM_LIB_PATH}")
    set(MNN_READY FALSE)
endif()

if(MNN_READY)
    target_include_directories(
        aisport_native
        PRIVATE
        "${MNN_INCLUDE_DIR}"
        "${MNN_LLM_INCLUDE_DIR}"
    )

    target_compile_definitions(aisport_native PRIVATE HAS_MNN)
    target_link_libraries(
        aisport_native
        "${MNN_LIB_PATH}"
        "${MNN_EXPRESS_LIB_PATH}"
        "${MNN_LLM_LIB_PATH}"
        android
        log
    )
else()
    message(WARNING "MNN runtime incomplete. Building STUB mode.")
    target_link_libraries(aisport_native android log)
endif()
