# Detectron2 Training Tool

Dit project biedt een eenvoudige interface om Detectron2 modellen te trainen met data in YOLO-formaat. Het stelt gebruikers in staat om object detectie modellen te trainen met minimale configuratie.

## Overzicht

Deze tool converteert YOLO-formaat annotaties naar een formaat dat compatibel is met Detectron2 en verzorgt het trainen van object detectie modellen. Het ondersteunt verschillende Detectron2 architecturen en biedt flexibiliteit in training parameters.

## Vereisten

- Python 3.7+
- PyTorch
- Detectron2
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

## Gebruik

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

## Uitvoer

Het trainingsproces slaat checkpoints en een eindmodel op in de opgegeven output directory.
