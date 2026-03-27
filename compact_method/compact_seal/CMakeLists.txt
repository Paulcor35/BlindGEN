cmake_minimum_required(VERSION 3.20)
project(BlindGEN_Server)

set(CMAKE_CXX_STANDARD 17)

# Configuration SEAL
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZSTD OFF CACHE BOOL "" FORCE)

# Chemin vers SEAL (Submodule à la racine)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../SEAL ${CMAKE_BINARY_DIR}/SEAL)

# Discovery Python 3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Discovery pybind11
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Extension Python : blind_engine_sov
pybind11_add_module(blind_engine_sov bindings.cpp)

target_link_libraries(blind_engine_sov PRIVATE 
    SEAL::seal
)

# Optimisations
if(MSVC)
    target_compile_options(blind_engine_sov PRIVATE /O2 /Oi /Ot /GL /MD)
    set_target_properties(blind_engine_sov PROPERTIES PREFIX "" SUFFIX ".pyd")
    # Copie vers 'compact_method' pour être vu par blind_chat_cpp.py et app.py
    add_custom_command(TARGET blind_engine_sov POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blind_engine_sov> ${CMAKE_CURRENT_SOURCE_DIR}/../blind_engine_sov.pyd)
else()
    target_compile_options(blind_engine_sov PRIVATE -O3 -march=native)
    set_target_properties(blind_engine_sov PROPERTIES PREFIX "" SUFFIX ".so")
    # Copie vers 'compact_method'
    add_custom_command(TARGET blind_engine_sov POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blind_engine_sov> ${CMAKE_CURRENT_SOURCE_DIR}/../blind_engine_sov.so)
endif()
