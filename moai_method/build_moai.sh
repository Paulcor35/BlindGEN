#!/bin/bash
# Script d'automatisation du Build MOAI C++ (SEAL) pour Linux
# Place ce fichier dans moai_method/

echo -e "\e[36m--- Configuration du Build MOAI (Native C++ & SEAL) ---\e[0m"

BUILD_DIR="moai_seal/build"

# Création du dossier build s'il n'existe pas
mkdir -p "$BUILD_DIR"

# 1. Configuration CMake
echo -e "\e[33m[1/2] Configuration CMake...\e[0m"
cd "$BUILD_DIR" || exit
cmake .. -DCMAKE_BUILD_TYPE=Release

# 2. Compilation
echo -e "\e[33m[2/2] Compilation du moteur et de l'extension...\e[0m"
make -j$(nproc)

# Note: Pour Linux, l'extension s'appellera moai_seal_backend.so 
# Le CMakeLists.txt devra gérer l'extension .so au lieu de .pyd si besoin,
# mais par défaut CMake s'adapte.

cd ../..

echo -e "\n\e[32m--- BUILD TERMINE ---\e[0m"
echo "Fichiers disponibles dans moai_method/ :"
echo "  - moai_gpt2 (Moteur d'inférence Natif)"
echo "  - moai_seal_backend.so (Extension Python FHE)"
