import os
import argparse
import json
import cv2
from ditod import add_vit_config
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from shapely.geometry import Polygon

def convert_to_labelme_format(class_name, bbox, score):
    labelme_annotation = {
        "label": class_name,
        "points": [
            [bbox[0], bbox[1]],
            [ bbox[2], bbox[3]]
        ],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {"confidence": score},
        "mask": None
    }
    return labelme_annotation

def convert_to_serializable(item):
    if isinstance(item, torch.Tensor):
        return item.tolist()
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, np.generic):
        return item.item()
    elif isinstance(item, np.bool_):
        return bool(item)
    else:
        raise TypeError("Type not serializable")

def calculate_overlapping_boxes(boxes, scores, threshold=0.5):
    """
    Calculate the number of overlapping boxes based on IoU threshold.
    """
    num_boxes = len(boxes)
    overlaps = 0
    for i in range(num_boxes):
        for j in range(i+1, num_boxes):
            box1 = boxes[i]
            box2 = boxes[j]
            if scores[i] > threshold and scores[j] > threshold:
                polygon1 = Polygon([(box1[0], box1[1]), (box1[2], box1[1]), (box1[2], box1[3]), (box1[0], box1[3])])
                polygon2 = Polygon([(box2[0], box2[1]), (box2[2], box2[1]), (box2[2], box2[3]), (box2[0], box2[3])])
                if polygon1.intersects(polygon2):
                    overlaps += 1
    return overlaps
def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--input_folder",
        help="Path to the input image folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        help="Path to the output folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--json_output_folder",
        help="Path to the output JSON folder in LabelMe format",
        type=str,
        default="labelme_output",
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Create output folder if not exists
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.json_output_folder, exist_ok=True)

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)

    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: run inference for each image in the input folder
for image_name in os.listdir(args.input_folder):
        image_path = os.path.join(args.input_folder, image_name)
        img = cv2.imread(image_path)

        md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        md.set(thing_classes=["text", "title", "list", "table", "figure"])
        thing_classes = md.thing_classes if hasattr(md, 'thing_classes') else None

        output = predictor(img)["instances"]

        num_detections = len(output)
        pred_boxes_cpu = output.pred_boxes.tensor.cpu().numpy()
        scores_cpu = output.scores.cpu().numpy()
        overlapping_detections = calculate_overlapping_boxes(pred_boxes_cpu, scores_cpu)

        if num_detections < 3 or overlapping_detections > 10:
            # Save output in LabelMe JSON file
            labelme_output_path = os.path.join(args.json_output_folder, f"{os.path.splitext(image_name)[0]}_output.json")
            labelme_annotations = []
            for i, box in enumerate(output.pred_boxes.to('cpu')):
                class_id = output.pred_classes[i].item() if thing_classes is None else thing_classes[output.pred_classes[i].item()]
                confidence = output.scores[i].item()
                if confidence > 0.5:
                    # Convert to LabelMe format
                    labelme_annotation = convert_to_labelme_format(class_id, bbox=box.numpy().tolist(), score=confidence)
                    labelme_annotations.append(labelme_annotation)

            if len(labelme_annotations) > 0:
                labelme_json = {
                    "version": "5.4.1",
                    "flags": {},
                    "shapes": labelme_annotations,
                    "imagePath": image_path,
                    "imageData": None,
                    "imageHeight": img.shape[0],
                    "imageWidth": img.shape[1]
                }

                with open(labelme_output_path, "w") as json_file:
                    json.dump(labelme_json, json_file, default=convert_to_serializable)

                # Save the output image in the output folder
                result_image_path = os.path.join(args.output_folder, f"{os.path.splitext(image_name)[0]}.jpg")
                cv2.imwrite(result_image_path, img)

    print("Conversion to LabelMe format completed.")

if __name__ == '__main__':
    main()
