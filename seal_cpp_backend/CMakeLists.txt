cmake_minimum_required(VERSION 3.13)
project(BlindGEN_Server)

# Find pybind11 using standard CMake search mechanism
# If not found (common for pip-installed pybind11), try using Python's module info
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND python3 -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 REQUIRED)
endif()

# SEAL configuration
set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_DEPS OFF CACHE BOOL "" FORCE)

# IMPORTANT: Path is now relative to the current source directory and points to the root submodule
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../SEAL native/seal)

# Create the Python extension module
pybind11_add_module(blind_engine_sov bindings.cpp)
target_link_libraries(blind_engine_sov PRIVATE seal)

# Optimization and C++ standards
target_compile_features(blind_engine_sov PRIVATE cxx_std_17)

if(MSVC)
    # MSVC specific optimizations
    target_compile_options(blind_engine_sov PRIVATE /O2 /Oi /Ot /GL /MD)
else()
    # GCC/Clang optimizations for Mac/Linux
    target_compile_options(blind_engine_sov PRIVATE -O3 -march=native)
endif()
