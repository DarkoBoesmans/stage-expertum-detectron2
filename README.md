# Detectron2 Training Tool

Dit project biedt een eenvoudige interface om Detectron2 modellen te trainen met data in YOLO-formaat. Het stelt gebruikers in staat om object detectie modellen te trainen met minimale configuratie.

## Overzicht

Deze tool converteert YOLO-formaat annotaties naar een formaat dat compatibel is met Detectron2 en verzorgt het trainen van object detectie modellen. Het ondersteunt verschillende Detectron2 architecturen en biedt flexibiliteit in training parameters.

Het project bevat de volgende hoofdcomponenten:
- `train.py`: Standaard training script
- `check_model.py`: Script om modellen te trainen met geheugenmonitoring en beperkt geheugen
- `predict.py`: Script om voorspellingen te maken met getrainde modellen
- `util.py`: Bevat hulpfuncties voor data verwerking en model configuratie
- `loss.py`: Bevat custom loss functies voor training

## Vereisten

- Python 3.7+
- PyTorch 2.0.0
- torchvision 0.15.1
- Detectron2 (installeerbaar via GitHub)
- OpenCV
- matplotlib
- Overige afhankelijkheden in `requirements.txt`

## Datastructuur

Het systeem verwacht data in de volgende structuur:

```
data-dir
----- train
--------- imgs
------------- filename0001.jpg
------------- filename0002.jpg
------------- ...
--------- anns
------------- filename0001.txt
------------- filename0002.txt
------------- ...
----- val
--------- imgs
------------- filename0001.jpg
------------- ...
--------- anns
------------- filename0001.txt
------------- ...
```

De annotaties moeten in YOLO-formaat zijn:
```
class_id x_center y_center width height
```

## Installatie

### 1. Python virtuele omgeving aanmaken en activeren

```bash
# Maak een virtuele omgeving aan
python -m venv detectron2-env

# Activeer de omgeving
# Op Linux/Mac:
source detectron2-env/bin/activate
# Op Windows:
# detectron2-env\Scripts\activate
```

### 2. Installeer de vereiste packages

```bash
# Installeer dependencies
pip install -r requirements.txt
```

## Gebruik

### Basis training

Train een model met standaard parameters:

```bash
python train.py --data-dir ./data --class-list ./class.names
```

Alle beschikbare opties:

```bash
python train.py --class-list ./class.names \
                --data-dir ./data \
                --output-dir ./output \
                --device cuda \
                --learning-rate 0.00025 \
                --batch-size 4 \
                --iterations 10000 \
                --checkpoint-period 500 \
                --model "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
```

### Training met geheugenmonitoring

Voor machines met beperkt geheugen kan je het `check_model.py` script gebruiken:

```bash
python check_model.py --data-dir ./data \
                      --class-list ./class.names \
                      --device cuda \
                      --mini-batch-size 5
```

### Voorspellen met een getraind model

Nadat je een model hebt getraind, kan je voorspellingen maken:

```bash
python predict.py --weights ./output/model_final.pth \
                 --input ./test_images/ \
                 --output ./predictions/ \
                 --class-list ./class.names
```

## Uitvoer

Het trainingsproces slaat checkpoints en een eindmodel op in de opgegeven output directory:
- `model_final.pth`: Het uiteindelijke, getrainde model
- `model_00XXXX.pth`: Tussentijdse checkpoints (afhankelijk van checkpoint-period)
- Diverse evaluatie logs en metrics

## Project Structuur

```
.
├── train.py                 # Standaard trainingsscript
├── check_model.py           # Training met geheugenmonitoring en beperkt geheugen
├── predict.py               # Voorspellingen maken met getrainde modellen
├── util.py                  # Hulpfuncties voor data en model configuratie
├── loss.py                  # Implementatie van custom loss functies
├── class.names              # Lijst van objectklassen voor detectie
├── requirements.txt         # Benodigde Python packages
├── setup.sh                 # Script voor het opzetten van de omgeving
├── README.md                # Documentatie
├── HANDOVER.md              # Aanvullende documentatie voor overdracht
├── data/                    # Map voor trainings- en validatiedata
│   ├── train/               # Trainingsdata
│   │   ├── imgs/            # Trainingsafbeeldingen
│   │   └── anns/            # Trainingslabels (YOLO-formaat)
│   └── val/                 # Validatiedata (zelfde structuur)
├── img/                     # Map met testafbeeldingen
├── output/                  # Map voor trainingsuitvoer
└── predictions/             # Map voor voorspellingsresultaten
└── output/                  # Map voor trainingsuitvoer
```

## Tips voor het gebruik van dit project

1. **Omgaan met beperkt geheugen**
   - Gebruik `check_model.py` met een klein `mini-batch-size` en lage `batch-size` waarde
   - Voor zeer grote modellen, probeer `--device cpu` om GPU geheugen te vermijden

2. **Beste practices voor data**
   - Zorg ervoor dat alle afbeeldingen in dezelfde resolutie zijn
   - Gebruik consistent gelabelde data
   - Bewaar een goede balans tussen klassen indien mogelijk

3. **Model selectie**
   - Voor snelheid: `COCO-Detection/retinanet_R_50_FPN_1x.yaml`
   - Voor nauwkeurigheid: `COCO-Detection/retinanet_R_101_FPN_3x.yaml`


## Troubleshooting

Common issues and their solutions can be found in the backup directory in TROUBLESHOOTING.md

Key troubleshooting tips:

- Probleem: "CUDA out of memory" fout tijdens training
- Probleem: Langzame training door geheugenlekkage
- Probleem: "No CUDA-capable device is detected"
- Probleem: Inconsistente GPU prestaties
- Probleem: Loss daalt niet of is instabiel
- Probleem: Model detecteert bepaalde klassen niet
- Probleem: Incorrecte data conversie van YOLO naar Detectron2 formaat
- Probleem: Missende of corrupte afbeeldingen/annotaties
- Probleem: Detectron2 installeert niet correct
- Probleem: Incompatibiliteit tussen packages


## Command References

Detailed command references can be found in the backup directory in COMMAND_REFERENCE.md

All scripts support --help to show available options.

