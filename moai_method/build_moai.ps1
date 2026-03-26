# Script d'automatisation du Build MOAI C++ (SEAL)
# Place ce fichier dans moai_method/

Write-Host "--- Configuration du Build MOAI (Native C++ & SEAL) ---" -ForegroundColor Cyan

$build_dir = "moai_seal/build"

# Création du dossier build s'il n'existe pas
if (-not (Test-Path $build_dir)) {
    New-Item -ItemType Directory -Path $build_dir | Out-Null
}

# 1. Configuration CMake
Write-Host "[1/2] Configuration CMake..." -ForegroundColor Yellow
cd $build_dir
cmake ..

# 2. Compilation (Release)
Write-Host "[2/2] Compilation du moteur et de l'extension..." -ForegroundColor Yellow
cmake --build . --config Release

# Note: Les fichiers .pyd et .exe sont déjà copiés automatiquement 
# vers moai_method/ grâce aux règles POST_BUILD du CMakeLists.txt.

cd ../..

Write-Host "`n--- BUILD TERMINE ---" -ForegroundColor Green
Write-Host "Fichiers disponibles dans moai_method/ :"
Write-Host "  - moai_gpt2.exe (Moteur d'inférence Natif)"
Write-Host "  - moai_seal_backend.pyd (Extension Python FHE)"
