from abc import abstractmethod
from typing import List, Union, Dict, Any
from dataclasses import dataclass
from copy import copy
from itertools import chain

import layoutparser as lp
import pandas as pd
from pycocotools.coco import COCO

from ..datamodel import *


class BaseBlockPredictor:
    """An abstact class for block predictors.
    The block predictor detects blocks of text from the input
    """

    @abstractmethod
    def detect(self, block: List, tokens: List) -> HierarchicalLayout:
        """Perform block detection based on the input."""


class ObjectDetectionBlockPredictor(BaseBlockPredictor):
    """Use Computer Vision based Object Detection models for
    detecting blocks from a given input
    """

    def __init__(self, config_path, model_path=None, label_map=None, extra_config=[]):

        self.model = lp.Detectron2LayoutModel(
            config_path,
            model_path=model_path,
            label_map=label_map,
            extra_config=extra_config,
        )

    def detect(self, image, tokens: Union[lp.Layout, List]) -> HierarchicalLayout:
        """Detect the blocks based on the input image, group the
        tokens based on the layout, and returns a `HierarchicalLayout`.

        Args:
            image ([type]): The in
            tokens (Union[lp.Layout, List]):
                All the tokens within the page

        """
        blocks = self.model.detect(image)

        return HierarchicalLayout.from_raw_block_token(blocks, tokens)

    __init__.__doc__ = lp.Detectron2LayoutModel.__doc__
    # An ugly hack for copying the docstring
