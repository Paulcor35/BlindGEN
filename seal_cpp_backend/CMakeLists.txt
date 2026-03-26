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
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../SEAL ${CMAKE_BINARY_DIR}/SEAL)

# Discovery Python et pybind11 (Méthode robuste Moai)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extension Python : blind_engine_sov
add_library(blind_engine_sov MODULE bindings.cpp)

target_include_directories(blind_engine_sov PRIVATE 
    ${PYBIND11_INCLUDE_DIR} 
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(blind_engine_sov PRIVATE 
    SEAL::seal 
    ${Python3_LIBRARIES}
)

# Optimisations
if(MSVC)
    target_compile_options(blind_engine_sov PRIVATE /O2 /Oi /Ot /GL /MD)
    set_target_properties(blind_engine_sov PROPERTIES PREFIX "" SUFFIX ".pyd")
    # Copie vers la racine pour être vu par app.py
    add_custom_command(TARGET blind_engine_sov POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blind_engine_sov> ${CMAKE_CURRENT_SOURCE_DIR}/../blind_engine_sov.pyd)
else()
    target_compile_options(blind_engine_sov PRIVATE -O3 -march=native)
    set_target_properties(blind_engine_sov PROPERTIES PREFIX "" SUFFIX ".so")
    # Copie vers la racine
    add_custom_command(TARGET blind_engine_sov POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blind_engine_sov> ${CMAKE_CURRENT_SOURCE_DIR}/../blind_engine_sov.so)
endif()
