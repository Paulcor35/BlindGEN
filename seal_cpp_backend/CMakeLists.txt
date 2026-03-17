cmake_minimum_required(VERSION 3.13)
project(BlindGEN_Server)

# Chemins Python (Spécifique à votre environnement Windows)
set(PYBIND11_DIR "C:/Users/willi/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/pybind11/share/cmake/pybind11")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
find_package(pybind11 REQUIRED)

# Inclure Microsoft SEAL depuis le dossier external
# On désactive tout ce qui ralentit la compilation
set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_DEPS OFF CACHE BOOL "" FORCE)

# IMPORTANT : Le chemin doit être relatif au dossier actuel (seal_cpp_backend)
# ../external/SEAL pointe vers c:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/external/SEAL
add_subdirectory(../external/SEAL native/seal)

# Créer l'extension Python
pybind11_add_module(blind_engine_sov bindings.cpp)
target_link_libraries(blind_engine_sov PRIVATE seal)

# Optimisations MSVC
if(MSVC)
    target_compile_options(blind_engine_sov PRIVATE /O2 /Oi /Ot /GL /MD)
endif()
