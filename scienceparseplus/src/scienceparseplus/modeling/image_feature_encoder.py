import torch
import logging
import time
import weakref
import os
import pandas as pd
import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances


class ImageFeatureEncoder(DefaultPredictor):
    """
    Create an active learning predictor with the given config that runs on
    single device for a single input image.
    """

    def __call__(self, original_image, token_boxes:torch.Tensor):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}

            
            token_boxes[:, ::2] = token_boxes[:, ::2] * (width / 1000)  # x1, x2
            token_boxes[:, 1::2] = token_boxes[:, 1::2] * (height / 1000)  # y1, y2
            # enforce for [CLS]
            token_boxes[0, 0] = 0
            token_boxes[0, 1] = 0
            token_boxes[0, 2] = width
            token_boxes[0, 3] = height

            token_proposals = [
                Instances(image_size=(height, width), proposal_boxes=Boxes(token_boxes)).to(
                    self.model.device
                )
            ]

            images = self.model.preprocess_image([inputs])
            features = self.model.backbone(images.tensor)

            _features = [features[f] for f in self.model.roi_heads.in_features]
            _box_features = self.model.roi_heads.box_pooler(
                _features, [x.proposal_boxes for x in token_proposals]
            )
            box_features = self.model.roi_heads.box_head(_box_features)

            del images, inputs, features, _features, _box_features, token_boxes
            return box_features.detach()


def create_image_feature_encoder(model_path, device = None):

    cfg = get_cfg()
    cfg.merge_from_file(f"{model_path}/config.yaml")
    cfg.MODEL.WEIGHTS = f"{model_path}/model_final.pth"
    if device is not None:
        cfg.MODEL.DEVICE = device
    return ImageFeatureEncoder(cfg)