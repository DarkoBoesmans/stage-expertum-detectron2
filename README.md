# Afvalcontainer Detectie Model met Detectron2

Dit project biedt een compleet systeem voor het detecteren van afvalcontainers in afbeeldingen, gebaseerd op Detectron2. Je kunt het systeem vanaf nul opzetten, je eigen model trainen en voorspellingen maken.

> **Gebaseerd op de tutorial:** [Custom Object Detection for Beginners to Masters](https://www.youtube.com/watch?v=I7O4ymSDcGw)

## Volledige Stappenplan van Nul

### 1. Project Setup en Configuratie

```bash
# Eén commando voor complete setup
./project.sh
```

Dit script voert de volgende taken uit:
- Controleert of alle dependencies correct zijn geïnstalleerd
- Configureert de projectstructuur
- Test de Detectron2 installatie
- Bereidt alles voor voor training

Alternatief kun je specifieke stappen uitvoeren:
```bash
./project.sh setup    # Alleen omgeving opzetten
./project.sh test     # Alleen testen of alles werkt
```

### 2. Projectinstallatie (Als je het handmatig wilt doen)

```bash
# Stap 1: Clone de repository
git clone https://github.com/jouw-username/afvalcontainer-detectie.git
cd afvalcontainer-detectie

# Stap 2: Maak een Python virtual environment
python -m venv detectron2-env

# Stap 3: Activeer de virtual environment
source detectron2-env/bin/activate  # Voor Linux/macOS
# OF
detectron2-env\Scripts\activate      # Voor Windows

# Stap 4: Installeer alle dependencies
pip install -r requirements.txt
```

```bash
# Eén commando voor complete setup
./project.sh
```

Dit script voert de volgende taken uit:
- Controleert of alle dependencies correct zijn geïnstalleerd
- Configureert de projectstructuur
- Test de Detectron2 installatie
- Bereidt alles voor voor training

Alternatief kun je specifieke stappen uitvoeren:
```bash
./project.sh setup    # Alleen omgeving opzetten
./project.sh test     # Alleen testen of alles werkt
```

### 3. Data Voorbereiden

Je dataset moet de volgende structuur hebben:
```
data/
  train/
    imgs/      # Trainingsafbeeldingen (.jpg, .png, etc.)
    anns/      # Trainingsannotaties (YOLO-formaat .txt bestanden)
  val/
    imgs/      # Validatieafbeeldingen
    anns/      # Validatieannotaties
```

Zorg ervoor dat je `class.names` bestand alle klassen bevat die je wilt detecteren (één per regel).

### 4. Model Trainen

```bash
# Start de training
python train.py
```

De trainingsopties kunnen worden aangepast in het script of via command line argumenten:
- `--data-dir`: Directory met trainingsdata (standaard: ./data)
- `--output-dir`: Directory voor modeluitvoer (standaard: ./output)
- `--class-list-file`: Bestand met klassenamen (standaard: ./class.names)
- `--device`: "cpu" of "cuda" voor training (standaard: automatisch gedetecteerd)

De training kan afhankelijk van je hardware meerdere uren duren.

### 5. Voorspellingen Maken

Nadat je model is getraind, kun je het gebruiken om voorspellingen te maken op nieuwe afbeeldingen:

```bash
# Basis voorspelling met standaard instellingen
python predict.py

# OF met aangepaste paden
python predict.py --model-path ./output/model_final.pth --image-path ./mijn_fotos --output-path ./mijn_resultaten
```

Standaardinstellingen:
- Model: `./output/model_final.pth` (je getrainde model)
- Invoer afbeeldingen: `./img` directory
- Uitvoer annotaties: `./predictions` directory

Voor meer options:
```bash
python predict.py --help
```

## Project Componenten

### Scripts
- `project.sh`: **Belangrijkste startpunt** - Unified management script voor setup en testen
- `train.py`: Hoofdscript voor het trainen van het model
- `predict.py`: Script voor het maken van voorspellingen op nieuwe afbeeldingen
- `util.py`: Hulpfuncties voor data-conversie en training
- `loss.py`: Aangepaste loss functies en validatiehooks

### Directories
- `data/`: Bevat trainings- en validatiedata
- `img/`: Voorbeeldafbeeldingen voor voorspellingen
- `output/`: Trainingsresultaten en getrainde modellen 
- `predictions/`: Resultaten van voorspellingen

## Tips voor Beste Resultaten

1. **Voor Training**:
   - Gebruik gevarieerde afbeeldingen met verschillende lichtomstandigheden
   - Zorg voor goede annotaties die de hele container omvatten
   - Balanceer je dataset met verschillende soorten containers
   - Volg de [video tutorial](https://www.youtube.com/watch?v=I7O4ymSDcGw) voor gedetailleerde uitleg

2. **Voor Voorspellingen**:
   - Gebruik vergelijkbare beeldkwaliteit als in de training
   - Werkt het beste bij daglicht en duidelijk zichtbare containers
   - Probeer verschillende confidence thresholds als detectie niet optimaal is

## Problemen Oplossen

Als je problemen ondervindt:

1. Controleer of Detectron2 correct is geïnstalleerd: `./project.sh test`
2. Controleer of de datastructuur correct is
3. Bij CUDA fouten, probeer te trainen op CPU: `python train.py --device cpu`
4. Bij voorspellingsproblemen, controleer of het model bestaat in de output directory
