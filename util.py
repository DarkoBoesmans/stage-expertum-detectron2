import os

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo

from loss import ValidationLoss

import cv2


def get_cfg(
    output_dir,
    learning_rate,
    batch_size,
    iterations,
    checkpoint_period,
    model,
    device,
    nmr_classes,
):
    """
    Create a Detectron2 configuration object and set its attributes.

    Args:
        output_dir (str): The path to the output directory where the trained model and logs will be saved.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size used during training.
        iterations (int): The maximum number of training iterations.
        checkpoint_period (int): The number of iterations between consecutive checkpoints.
        model (str): The name of the model to use, which should be one of the models available in Detectron2's model zoo.
        device (str): The device to use for training, which should be 'cpu' or 'cuda'.
        nmr_classes (int): The number of classes in the dataset.

    Returns:
        The Detectron2 configuration object.
    """
    cfg = _get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model))

    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)

    cfg.DATALOADER.NUM_WORKERS = 4

    # Use a model pre-trained on COCO.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Set the device to use for training.
    if device in ["cpu"]:
        cfg.MODEL.DEVICE = "cpu"

    # Use a smaller batch size if you encounter out-of-memory errors.
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # Set the learning rate for the optimizer.
    cfg.SOLVER.BASE_LR = learning_rate

    # Set the maximum number of training iterations.
    cfg.SOLVER.MAX_ITER = iterations

    # Set the number of iterations between consecutive checkpoints.
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    # Set the number of classes in the dataset.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = nmr_classes

    cfg.OUTPUT_DIR = output_dir

    return cfg


def get_dicts(img_dir, ann_dir):
    """
    Convert the YOLO-format annotations in ann_dir to the format expected by Detectron2.

    Args:
        img_dir (str): The directory containing the image files.
        ann_dir (str): The directory containing the annotation files in YOLO format.

    Returns:
        list: A list of dictionaries, one for each image, where each dictionary contains the following keys:
            - file_name: The path to the image file.
            - image_id: A unique ID for the image.
            - height: The height of the image in pixels.
            - width: The width of the image in pixels.
            - annotations: A list of dictionaries, one for each object in the image, containing the following keys:
                - bbox: A list of four integers [x0, y0, w, h] representing the bounding box of the object in the image,
                        where (x0, y0) is the top-left corner and (w, h) are the width and height of the bounding box,
                        respectively.
                - bbox_mode: A constant from the `BoxMode` class indicating the format of the bounding box coordinates
                            (e.g., `BoxMode.XYWH_ABS` for absolute coordinates in the format [x0, y0, w, h]).
                - category_id: The integer ID of the object's class.
    """
    dataset_dicts = []
    annotation_files = os.listdir(ann_dir)
    annotations_count = 0
    valid_images = 0

    print(f"Loading annotations from {ann_dir} ({len(annotation_files)} files)")

    for idx, file in enumerate(annotation_files):
        # annotations should be provided in yolo format
        record = {}

        # Check for common image extensions
        img_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        img_found = False
        filename = None

        for ext in img_extensions:
            img_path = os.path.join(img_dir, file[:-4] + ext)
            if os.path.exists(img_path):
                filename = img_path
                img_found = True
                break

        if not img_found:
            print(f"Warning: No image found for annotation {file}")
            continue

        img = cv2.imread(filename)
        if img is None:
            print(f"Warning: Could not read image {filename}")
            continue

        height, width = img.shape[:2]
        valid_images += 1

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(os.path.join(ann_dir, file)) as r:
            lines = [l.strip() for l in r.readlines()]

        for _, line in enumerate(lines):
            if len(line) > 2 and not line.startswith("#"):
                try:
                    parts = line.split(" ")
                    if len(parts) != 5:
                        print(f"Warning: Invalid annotation format in {file}: {line}")
                        continue

                    label, cx, cy, w_, h_ = parts

                    obj = {
                        "bbox": [
                            int((float(cx) - (float(w_) / 2)) * width),
                            int((float(cy) - (float(h_) / 2)) * height),
                            int(float(w_) * width),
                            int(float(h_) * height),
                        ],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": int(label),
                    }

                    objs.append(obj)
                    annotations_count += 1

                except ValueError as e:
                    print(f"Warning: Could not parse annotation in {file}: {line}")
                    print(f"Error: {str(e)}")
                    continue

        record["annotations"] = objs
        dataset_dicts.append(record)

    print(
        f"Successfully loaded {valid_images} images with {annotations_count} annotations"
    )
    return dataset_dicts


def register_datasets(root_dir, class_list_file):
    """
    Registers the train and validation datasets and returns the number of classes.
    If the datasets are already registered, it will not register them again.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        class_list_file (str): Path to the file containing the list of class names.

    Returns:
        int: The number of classes in the dataset.
    """
    # Read the list of class names from the class list file.
    with open(class_list_file, "r") as reader:
        classes_ = [l.strip() for l in reader.readlines()]

    # Register the train and validation datasets, deregistering them first if already registered
    for d in ["train", "val"]:
        # De-register if already registered
        if d in DatasetCatalog:
            DatasetCatalog.remove(d)
            print(
                f"Dataset '{d}' was already registered. Re-registering with updated data."
            )

        # Register the dataset
        DatasetCatalog.register(
            d,
            lambda d=d: get_dicts(
                os.path.join(root_dir, d, "imgs"), os.path.join(root_dir, d, "anns")
            ),
        )
        # Set the metadata for the dataset.
        MetadataCatalog.get(d).set(thing_classes=classes_)

    return len(classes_)


def train(
    output_dir,
    data_dir,
    class_list_file,
    learning_rate,
    batch_size,
    iterations,
    checkpoint_period,
    device,
    model,
):
    """
    Train a Detectron2 model on a custom dataset.

    Args:
        output_dir (str): Path to the directory to save the trained model and output files.
        data_dir (str): Path to the directory containing the dataset.
        class_list_file (str): Path to the file containing the list of class names in the dataset.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        iterations (int): Maximum number of training iterations.
        checkpoint_period (int): Number of iterations after which to save a checkpoint of the model.
        device (str): Device to use for training (e.g., 'cpu' or 'cuda').
        model (str): Name of the model configuration to use. Must be a key in the Detectron2 model zoo.

    Returns:
        None
    """

    # Count training and validation images
    train_img_count = (
        len(os.listdir(os.path.join(data_dir, "train", "imgs")))
        if os.path.exists(os.path.join(data_dir, "train", "imgs"))
        else 0
    )
    val_img_count = (
        len(os.listdir(os.path.join(data_dir, "val", "imgs")))
        if os.path.exists(os.path.join(data_dir, "val", "imgs"))
        else 0
    )

    print(
        f"Dataset contains {train_img_count} training images and {val_img_count} validation images"
    )

    # Register the dataset and get the number of classes
    nmr_classes = register_datasets(data_dir, class_list_file)

    # Create the configuration object for the model
    cfg = get_cfg(
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        iterations=iterations,
        checkpoint_period=checkpoint_period,
        model=model,
        device=device,
        nmr_classes=nmr_classes,
    )

    # Create a DefaultTrainer object and train the model
    trainer = ValidationLoss(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
