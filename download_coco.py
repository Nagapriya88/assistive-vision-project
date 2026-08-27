import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["person", "car", "bicycle", "motorcycle", "bus", "truck",
              "bench", "chair", "traffic light", "fire hydrant"],
    max_samples=3000,   # keep it small — adjust up/down as needed
)
dataset.export(
    export_dir="coco_subset_yolo",
    dataset_type=fo.types.YOLOv5Dataset,
)