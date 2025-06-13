import argparse
import os
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


def setup_predictor(model_path, device="cpu"):
    """
    Configureer de Detectron2 predictor op basis van het getrainde model.

    Args:
        model_path: Pad naar het getrainde modelbestand (.pth)
        device: "cpu" of "gpu" voor inferentie

    Returns:
        Een Detectron2 DefaultPredictor object
    """
    # Laad de basisconfiguratie
    cfg = get_cfg()

    # Probeer automatisch het juiste model te bepalen
    model_type = None

    # Methode 1: Controleer of metrics.json bestaat voor modeltype informatie
    metrics_path = "./output/metrics.json"
    if os.path.exists(metrics_path):
        try:
            import json

            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                if isinstance(metrics, list) and len(metrics) > 0:
                    if any("retinanet" in key.lower() for key in metrics[0].keys()):
                        model_type = "RetinaNet"
                    else:
                        model_type = "FasterRCNN"
        except Exception as e:
            print(f"Kon metrics niet lezen: {e}")

    # Methode 2: Als metrics niet beschikbaar zijn, probeer het modelpad te controleren
    if model_type is None:
        if "retinanet" in model_path.lower():
            model_type = "RetinaNet"
        else:
            model_type = "FasterRCNN"  # Standaard

    # Configureer op basis van het gedetecteerde modeltype
    print(f"Gedetecteerd model type: {model_type}")
    if model_type == "RetinaNet":
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml")
        )
    else:  # FasterRCNN
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        )

    # Laad de klassen uit class.names
    classes = []
    if os.path.exists("./class.names"):
        with open("./class.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"Geladen klassen: {classes}")

    # Stel de juiste classificatieklassen in
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)

    # Gebruik het model
    cfg.MODEL.WEIGHTS = model_path

    # Stel het juiste apparaat in (CPU/GPU)
    if device.lower() == "gpu" or device.lower() == "cuda":
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"

    # Stel de confidence threshold in
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5

    # Maak en geef de predictor terug
    return DefaultPredictor(cfg), classes


def predict_image(image_path, predictor_data):
    """
    Maak voorspellingen op een enkele afbeelding.

    Args:
        image_path: Pad naar de afbeelding
        predictor_data: Tuple (predictor, classes) van setup_predictor

    Returns:
        Tuple van (afbeelding met annotaties, ruwe outputs)
    """
    predictor, classes = predictor_data

    # Lees de afbeelding
    image = cv2.imread(image_path)
    if image is None:
        print(f"Kan afbeelding niet laden: {image_path}")
        return None, None

    # Maak een kopie voor de resultaten
    result_img = image.copy()

    # Voorspelling maken
    outputs = predictor(image)

    # Als er instances zijn gedetecteerd
    if "instances" in outputs:
        instances = outputs["instances"].to("cpu")

        # Haal de voorspelde velden op
        if len(instances) > 0:
            boxes = (
                instances.pred_boxes.tensor.numpy()
                if instances.has("pred_boxes")
                else []
            )
            scores = instances.scores.numpy() if instances.has("scores") else []
            classes_idx = (
                instances.pred_classes.numpy() if instances.has("pred_classes") else []
            )

            # Teken de bounding boxes en labels
            for box, score, class_idx in zip(boxes, scores, classes_idx):
                # Haal de coördinaten op
                x1, y1, x2, y2 = [int(coord) for coord in box]

                # Maak de label met de klassenaam en score
                class_name = (
                    classes[class_idx]
                    if class_idx < len(classes)
                    else f"Klasse {class_idx}"
                )
                label = f"{class_name}: {score:.2f}"

                # Teken de rechthoek
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Teken de label
                cv2.putText(
                    result_img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    return result_img, outputs


def process_images(image_dir, output_dir, predictor_data):
    """
    Verwerk alle afbeeldingen in een directory.

    Args:
        image_dir: Map met invoer afbeeldingen
        output_dir: Map voor uitvoer afbeeldingen
        predictor_data: Tuple (predictor, classes) van setup_predictor
    """
    os.makedirs(output_dir, exist_ok=True)

    # Haal alle afbeeldingsbestanden op
    images = [
        f
        for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if len(images) == 0:
        print(f"Geen afbeeldingen gevonden in {image_dir}")
        return

    print(f"Verwerken van {len(images)} afbeeldingen...")

    # Verwerk elke afbeelding
    for i, img_file in enumerate(images):
        img_path = os.path.join(image_dir, img_file)
        output_path = os.path.join(output_dir, img_file)

        # Maak voorspelling
        print(f"[{i+1}/{len(images)}] Verwerken van {img_file}...")
        result_img, _ = predict_image(img_path, predictor_data)

        # Sla op als de voorspelling succesvol was
        if result_img is not None:
            cv2.imwrite(output_path, result_img)

    print(f"Alle afbeeldingen verwerkt en opgeslagen in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectron2 voorspellingsscript")

    # Gebruik het standaard modelpad in output directory
    default_model_path = "./output/model_final.pth"

    parser.add_argument(
        "--model-path", default=default_model_path, help="Pad naar het getrainde model"
    )
    parser.add_argument(
        "--image-path",
        default="./img",
        help="Pad naar afbeelding of map met afbeeldingen",
    )
    parser.add_argument(
        "--output-path", default="./predictions", help="Pad naar uitvoermap"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="Apparaat voor inferentie (cpu, gpu, cuda)",
    )

    args = parser.parse_args()

    # Check of het model bestaat
    if not os.path.exists(args.model_path):
        print(f"FOUT: Model niet gevonden op {args.model_path}")
        print("Zorg ervoor dat je eerst een model traint met train.py")
        exit(1)

    # Check of het invoerpad bestaat
    if not os.path.exists(args.image_path):
        print(f"FOUT: Invoerpad niet gevonden: {args.image_path}")
        exit(1)

    print(f"Model: {args.model_path}")
    print(f"Invoer: {args.image_path}")
    print(f"Uitvoer: {args.output_path}")
    print(f"Apparaat: {args.device}")

    # Initialiseer de predictor
    try:
        print("Predictor aan het initialiseren...")
        predictor_data = setup_predictor(args.model_path, device=args.device)

        # Check of invoerpad een map of één afbeelding is
        if os.path.isdir(args.image_path):
            process_images(args.image_path, args.output_path, predictor_data)
        else:
            # Enkele afbeelding
            os.makedirs(args.output_path, exist_ok=True)
            filename = os.path.basename(args.image_path)
            output_path = os.path.join(args.output_path, filename)

            print(f"Voorspelling maken voor enkele afbeelding: {args.image_path}")
            result_img, _ = predict_image(args.image_path, predictor_data)

            if result_img is not None:
                cv2.imwrite(output_path, result_img)
                print(f"Voorspelling opgeslagen als: {output_path}")
            else:
                print("Kan geen voorspelling maken voor deze afbeelding.")

        print("Voorspelling voltooid!")

    except Exception as e:
        print(f"FOUT tijdens voorspelling: {str(e)}")
