import argparse
import os
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


def setup_predictor(model_path, device="cpu"):
    # Load config from a config file
    cfg = get_cfg()

    # Automatisch bepalen welke config te gebruiken op basis van het trainingsproces
    # Controleer modellog
    with open("./output/metrics.json", "r") as f:
        import json

        try:
            metrics = json.load(f)
            if isinstance(metrics, list) and len(metrics) > 0:
                if "validation_retinanet_loss" in metrics[0]:
                    print("Gedetecteerd model type: RetinaNet")
                    cfg.merge_from_file(
                        model_zoo.get_config_file(
                            "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
                        )
                    )
                else:
                    print("Gedetecteerd model type: Faster R-CNN")
                    cfg.merge_from_file(
                        model_zoo.get_config_file(
                            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                        )
                    )
            else:
                # Standaard RetinaNet gebruiken als terugvaloptie
                print(
                    "Geen metrics informatie gevonden, gebruik RetinaNet als standaard"
                )
                cfg.merge_from_file(
                    model_zoo.get_config_file(
                        "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
                    )
                )
        except:
            # Fallback op basis van bestandsnaam
            if "retinanet" in model_path.lower():
                print("Fallback naar RetinaNet op basis van bestandsnaam")
                cfg.merge_from_file(
                    model_zoo.get_config_file(
                        "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
                    )
                )
            else:
                print("Fallback naar Faster R-CNN op basis van bestandsnaam")
                cfg.merge_from_file(
                    model_zoo.get_config_file(
                        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                    )
                )

    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device

    # Confidence threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5

    return DefaultPredictor(cfg)


def predict_image(image_path, predictor):
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Kan afbeelding niet laden: {image_path}")
        return None, None

    # Perform prediction
    outputs = predictor(image)

    # Get predictions
    if len(outputs["instances"]) > 0:
        preds = outputs["instances"].pred_classes.tolist()
        scores = outputs["instances"].scores.tolist()
        bboxes = outputs["instances"].pred_boxes.tensor.tolist()

        # Draw bounding boxes
        result_img = image.copy()
        for j, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            score = scores[j]
            label = f"trash: {score:.2f}"

            # Draw box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
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

    return image, outputs


def process_images(image_dir, output_dir, predictor):
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in directory
    images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    print(f"Verwerken van {len(images)} afbeeldingen...")

    for i, img_file in enumerate(images):
        img_path = os.path.join(image_dir, img_file)
        output_path = os.path.join(output_dir, img_file)

        # Generate prediction
        result_img, _ = predict_image(img_path, predictor)

        if result_img is not None:
            # Save result
            cv2.imwrite(output_path, result_img)

        if (i + 1) % 5 == 0:
            print(f"Verwerkt: {i+1}/{len(images)}")

    print(f"Alle voorspellingen opgeslagen in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectron2 voorspellingsscript")
    parser.add_argument(
        "--model-path", required=True, help="Pad naar het getrainde model"
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Pad naar afbeelding of map met afbeeldingen",
    )
    parser.add_argument("--output-path", required=True, help="Pad naar uitvoermap")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Apparaat voor inferentie",
    )

    args = parser.parse_args()

    # Setup predictor
    predictor = setup_predictor(args.model_path, device=args.device)

    # Check if path is a directory or single file
    if os.path.isdir(args.image_path):
        process_images(args.image_path, args.output_path, predictor)
    else:
        # Single image
        os.makedirs(args.output_path, exist_ok=True)
        filename = os.path.basename(args.image_path)
        output_path = os.path.join(args.output_path, filename)

        result_img, _ = predict_image(args.image_path, predictor)

        if result_img is not None:
            cv2.imwrite(output_path, result_img)
            print(f"Voorspelling opgeslagen als: {output_path}")
        else:
            print("Kan geen voorspelling maken voor deze afbeelding.")
