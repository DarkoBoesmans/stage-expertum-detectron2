#!/bin/bash
# Script om een model gereed te maken voor GitHub release

REPO_DIR=$(pwd)
MODEL_DIR="$REPO_DIR/model_release"
WEIGHTS_DIR="$MODEL_DIR/weights"
OUTPUT_DIR="$REPO_DIR/output"
README="$REPO_DIR/README.md"

# Banner
echo "============================================="
echo "   Model voorbereiden voor GitHub release    "
echo "============================================="

# Controleren of Git LFS is geïnstalleerd
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS is niet geïnstalleerd. Installeer het met:"
    echo "  sudo apt-get install git-lfs"
    echo "  of"
    echo "  brew install git-lfs"
    exit 1
fi

# Controleren of model_release directory bestaat
if [ ! -d "$MODEL_DIR" ]; then
    echo "Maken van model_release directory..."
    mkdir -p "$WEIGHTS_DIR"
fi

# Controleren of er een getraind model is
if [ -f "$OUTPUT_DIR/model_final.pth" ]; then
    echo "Kopiëren van getraind model naar release directory..."
    cp "$OUTPUT_DIR/model_final.pth" "$WEIGHTS_DIR/"
    echo "Model is gekopieerd."
else
    echo "WAARSCHUWING: Geen getraind model gevonden in $OUTPUT_DIR"
    echo "Je moet eerst een model trainen met train.py voordat je deze release kunt maken."
fi

# Controleren of class.names bestaat en kopiëren
if [ -f "$REPO_DIR/class.names" ]; then
    echo "Kopiëren van class.names naar model_release directory..."
    cp "$REPO_DIR/class.names" "$MODEL_DIR/"
    echo "class.names is gekopieerd."
else
    echo "WAARSCHUWING: class.names niet gevonden. Dit bestand is nodig voor het model."
fi

# Controleren of GitHub release instructies in README staan
if grep -q "Git LFS" "$README"; then
    echo "README bevat al Git LFS instructies."
else
    echo "WAARSCHUWING: README bevat mogelijk geen Git LFS instructies."
    echo "Zorg ervoor dat je README instructies bevat voor gebruikers om Git LFS te installeren."
fi

# Git LFS configureren
echo "Git LFS configureren voor grote bestanden..."
git lfs install

# .gitattributes aanmaken/updaten
echo "*.pth filter=lfs diff=lfs merge=lfs -text" > "$REPO_DIR/.gitattributes"
echo "model_release/weights/*.pth filter=lfs diff=lfs merge=lfs -text" >> "$REPO_DIR/.gitattributes"

echo "Git LFS is geconfigureerd voor *.pth bestanden."

echo ""
echo "===== Voltooide acties ====="
echo "1. model_release directory en structuur aangemaakt"
echo "2. .gitattributes bestand geconfigureerd voor Git LFS"
echo ""
echo "===== Volgende stappen ====="
echo "1. Commit de wijzigingen met Git:"
echo "   git add model_release/ .gitattributes"
echo "   git commit -m \"Add pretrained model for release\""
echo ""
echo "2. Push naar GitHub (zorg dat je repository groot genoeg is voor LFS bestanden):"
echo "   git push origin main"
echo ""
echo "3. Maak een release op GitHub voor eenvoudige distributie"
echo ""
echo "Klaar! Je model is nu voorbereid voor GitHub."
