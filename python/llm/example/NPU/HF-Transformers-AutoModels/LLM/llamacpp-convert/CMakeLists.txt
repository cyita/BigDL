# set(TARGET convert-gguf-to-npu)
# add_executable(${TARGET} convert-gguf-to-npu.cpp)
# install(TARGETS ${TARGET} RUNTIME)
# target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(${TARGET} PRIVATE cxx_std_17)

cmake_minimum_required(VERSION 3.10)

project(LLM_GGUF_TO_NPU VERSION 1.0.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)
SET (CMAKE_CXX_STANDARD_REQUIRED True)
find_package(OpenMP REQUIRED)

if(DEFINED ENV{CONDA_ENV_DIR})
    set(ENV_DIR $ENV{CONDA_ENV_DIR})
    set(LIBRARY_DIR ${ENV_DIR}/bigdl-core-npu)
    include_directories(${LIBRARY_DIR}/include)
    set(DLL_DIR ${ENV_DIR}/intel_npu_acceleration_library/lib/Release)
    # set(LLAMA_CPP_LIBRARY_DIR "D:\\yina\\llamacpplibs")
    # include_directories(${LLAMA_CPP_LIBRARY_DIR}/include)
else()
    set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# set(LIBRARY_DIR "D:\\yina\\llamacpplibs")
# include_directories(${LIBRARY_DIR}/include)

add_library(llama STATIC IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${LIBRARY_DIR}/llama.lib)

add_library(common STATIC IMPORTED)
set_target_properties(common PROPERTIES IMPORTED_LOCATION ${LIBRARY_DIR}/common.lib)

add_library(ggml STATIC IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${LIBRARY_DIR}/ggml.lib)


set(TARGET convert-gguf-to-npu)
add_executable(${TARGET} convert-gguf-to-npu.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(PERF quantize-perf)
add_executable(${PERF} quantize-perf.cpp)
target_compile_options(${PERF} PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PERF} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${PERF} PRIVATE cxx_std_17)

add_custom_command(TARGET convert-gguf-to-npu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LIBRARY_DIR}/llama.dll
        ${LIBRARY_DIR}/ggml.dll
        ${CMAKE_BINARY_DIR}/Release/
    COMMENT "Copying npu_llm.dll to build/Release\n"
)

# add_custom_command(TARGET llama-cli-npu POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#         ${DLL_DIR}/
#         ${CMAKE_BINARY_DIR}/Release/
#     COMMENT "Copying dependency to build/Release\n"
# )
