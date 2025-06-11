#!/bin/bash
# Test script om te controleren of de basis functionaliteit werkt

echo "===== DETECTRON2 BASIS TEST ====="
echo "Dit script test of de kern functionaliteit van het project werkt"

# Controleer of Python beschikbaar is
echo "Controleren op Python..."
python --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python niet gevonden"
    exit 1
fi
echo "✓ Python beschikbaar"

# Controleer of de virtuele omgeving bestaat
echo "Controleren op virtuele omgeving..."
if [ ! -d "detectron2-env" ]; then
    echo "WAARSCHUWING: detectron2-env map niet gevonden. Gebruik setup.sh om de omgeving aan te maken."
else
    echo "✓ Virtuele omgeving gevonden"
fi

# Activeer de omgeving als deze bestaat
if [ -f "detectron2-env/bin/activate" ]; then
    echo "Activeren van virtuele omgeving..."
    source detectron2-env/bin/activate
    echo "✓ Virtuele omgeving geactiveerd"
fi

# Controleer benodigde bestanden
echo "Controleren op essentiële bestanden..."
ESSENTIALS=("train.py" "check_model.py" "predict.py" "util.py" "loss.py" "class.names" "requirements.txt")
MISSING=0
for file in "${ESSENTIALS[@]}"; do
    if [ ! -f "$file" ]; then
        echo "FOUT: $file niet gevonden"
        MISSING=1
    fi
done

if [ $MISSING -eq 0 ]; then
    echo "✓ Alle essentiële bestanden gevonden"
else
    echo "WAARSCHUWING: Sommige essentiële bestanden ontbreken"
fi

# Controleer of de virtuele omgeving correct is geïnstalleerd
echo "Controleren op installatie van PyTorch en andere dependencies..."
python -c "import torch; print(f'PyTorch versie: {torch.__version__}')"
if [ $? -ne 0 ]; then
    echo "WAARSCHUWING: PyTorch niet gevonden. Gebruik setup.sh om dependencies te installeren."
else
    echo "✓ PyTorch geïnstalleerd"
fi

python -c "import detectron2; print(f'Detectron2 beschikbaar')"
if [ $? -ne 0 ]; then
    echo "WAARSCHUWING: Detectron2 niet gevonden. Gebruik setup.sh om dependencies te installeren."
else
    echo "✓ Detectron2 geïnstalleerd"
fi

# Controleer of de datastructuur klopt
echo "Controleren van datastructuur..."
if [ -d "data/train/imgs" ] && [ -d "data/train/anns" ] && [ -d "data/val/imgs" ] && [ -d "data/val/anns" ]; then
    echo "✓ Datastructuur is correct"
    TRAIN_IMGS=$(ls data/train/imgs | wc -l)
    TRAIN_ANNS=$(ls data/train/anns | wc -l)
    VAL_IMGS=$(ls data/val/imgs | wc -l)
    VAL_ANNS=$(ls data/val/anns | wc -l)
    echo "  - Training afbeeldingen: $TRAIN_IMGS"
    echo "  - Training annotaties: $TRAIN_ANNS"
    echo "  - Validatie afbeeldingen: $VAL_IMGS"
    echo "  - Validatie annotaties: $VAL_ANNS"
else
    echo "WAARSCHUWING: Datastructuur lijkt niet volledig correct"
fi

# Testafbeeldingen controleren
if [ -d "img" ]; then
    IMG_COUNT=$(ls img | wc -l)
    echo "✓ Testafbeeldingen map gevonden met $IMG_COUNT afbeeldingen"
else
    echo "WAARSCHUWING: Geen img/ map gevonden voor testafbeeldingen"
fi

echo ""
echo "===== TEST AFGEROND ====="
echo ""
echo "Om een model te trainen, gebruik:"
echo "  python train.py --data-dir ./data --class-list ./class.names"
echo ""
echo "Voor geheugen-efficiënte training, gebruik:"
echo "  python check_model.py --data-dir ./data --class-list ./class.names"
echo ""
echo "Voor voorspellingen na training, gebruik:"
echo "  python predict.py --weights ./output/model_final.pth --input ./img --output ./predictions --class-list ./class.names"
echo ""
