cmake_minimum_required(VERSION 3.20)
project(MoaiSealBackend)

set(CMAKE_CXX_STANDARD 17)

# Configuration SEAL
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZSTD OFF CACHE BOOL "" FORCE)

# Chemin vers SEAL (Submodule à la racine)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../SEAL ${CMAKE_BINARY_DIR}/SEAL)

# Discovery pybind11
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# EXTENSION PYTHON (Pybind11)
pybind11_add_module(moai_seal_backend src/moai_seal_backend.cpp)
target_link_libraries(moai_seal_backend PRIVATE SEAL::seal)

# EXECUTABLES PURE C++ (Pour tests standalone si besoin)
add_executable(moai_gpt2 src/moai_gpt2_engine.cpp)
target_link_libraries(moai_gpt2 PRIVATE SEAL::seal)

# Activation OpenMP (Crucial pour MOAI car plus d'opérations de rotation)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    foreach(target moai_gpt2 moai_seal_backend)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        if(MSVC)
            target_compile_options(${target} PRIVATE /openmp)
        endif()
    endforeach()
endif()

# Installation automatique (.pyd ou .so)
if(MSVC)
    set_target_properties(moai_seal_backend PROPERTIES PREFIX "" SUFFIX ".pyd")
    # Copie vers 'moai_method' pour être vu par run_moai_universal.py
    add_custom_command(TARGET moai_seal_backend POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moai_seal_backend> ${CMAKE_CURRENT_SOURCE_DIR}/../moai_seal_backend.pyd)
    # On ajoute aussi une copie vers la racine pour la visibilité globale
    add_custom_command(TARGET moai_seal_backend POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moai_seal_backend> ${CMAKE_CURRENT_SOURCE_DIR}/../../moai_seal_backend.pyd)
else()
    set_target_properties(moai_seal_backend PROPERTIES PREFIX "" SUFFIX ".so")
    add_custom_command(TARGET moai_seal_backend POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:moai_seal_backend> ${CMAKE_CURRENT_SOURCE_DIR}/../moai_seal_backend.so)
endif()
