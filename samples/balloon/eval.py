"""
Compute VOC style mAP for the Balloon dataset Mask_RCNN model
--
Written by Arvind Iyer
--
  Usage:
    python eval.py --weights=/path/to/weights.h5 --dataset=/path/to/dataset
"""
import os
import sys
import numpy as np
# Root directory of the project
ROOT_DIR = os.getenv("ROOT_DIR", "../../")

from balloon import BalloonConfig, BalloonDataset
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib, utils


class InferenceConfig(BalloonConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def evaluate(model, dataset_val, num_eval_images=25):
    """
    Compute VOC mAP @ IoU=0.5
    Default on 25 images
    """
    config = InferenceConfig()
    config.display()
    image_ids = np.random.choice(dataset_val.image_ids, num_eval_images)
    APs = []

    for image_id in image_ids:
        # load image and annotations
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)
        # molded_images = np.expand_dims(modellib.mold_image(image, config), 0)

        # Run model on image (suppress logging)
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute avg precision
        AP, precision, recall, overlap = utils.compute_ap(
            gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r["masks"]
        )
        print(f"AP @ 0.5 <{image_id}: {AP}")
        APs.append(AP)
    print(f"mAP: {np.mean(APs)}")
    


def load_model_and_weights(weights):
    # load config
    config = InferenceConfig()
    config.display()
    # load model architecture
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=os.path.join(ROOT_DIR, "logs"))

    # Select weights file to load
    if weights.lower() == "coco":
        weights_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    return model


def load_dataset(dataset_path):
    # load validation dataset
    dataset = BalloonDataset()
    dataset.load_balloon(dataset_path, "val")
    dataset.prepare()
    return dataset


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate balloon detection MaskRCNN.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    model = load_model_and_weights(args.weights)
    