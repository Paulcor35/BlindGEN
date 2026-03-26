cmake_minimum_required(VERSION 3.20)
project(MoaiSealBackend)

set(CMAKE_CXX_STANDARD 17)

# On force SEAL en statique
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZLIB OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/../../SEAL ${CMAKE_BINARY_DIR}/SEAL)

# EXECUTABLES PURE C++
add_executable(moai_gpt2 src/moai_gpt2_engine.cpp)
target_link_libraries(moai_gpt2 PRIVATE SEAL::seal)

add_executable(moai_universal src/moai_universal_engine.cpp)
target_link_libraries(moai_universal PRIVATE SEAL::seal)

# EXTENSION PYTHON (Pybind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Module Legacy
add_library(moai_seal_backend MODULE src/moai_seal_backend.cpp)
target_include_directories(moai_seal_backend PRIVATE ${PYBIND11_INCLUDE_DIR} ${Python3_INCLUDE_DIRS})
target_link_libraries(moai_seal_backend PRIVATE SEAL::seal ${Python3_LIBRARIES})

# Activation OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    foreach(target moai_gpt2 moai_universal moai_seal_backend)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        if(MSVC)
            target_compile_options(${target} PRIVATE /openmp)
        endif()
    endforeach()
endif()

# Installation automatique (pyd / exe)
if(WIN32)
    set_target_properties(moai_seal_backend PROPERTIES PREFIX "" SUFFIX ".pyd")
    add_custom_command(TARGET moai_seal_backend POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moai_seal_backend> ${CMAKE_SOURCE_DIR}/../moai_seal_backend.pyd)
    add_custom_command(TARGET moai_gpt2 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moai_gpt2> ${CMAKE_SOURCE_DIR}/../moai_gpt2.exe)
    add_custom_command(TARGET moai_universal POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moai_universal> ${CMAKE_SOURCE_DIR}/../moai_universal.exe)
else()
    set_target_properties(moai_seal_backend PROPERTIES PREFIX "" SUFFIX ".so")
    # Linux support (omitted for brevity but logic is similar)
endif()
