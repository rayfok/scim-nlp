from typing import List, Union, Dict, Any, Tuple
from abc import abstractmethod
from collections import Counter
import json
import os

from tqdm import tqdm
import pandas as pd

from ..datasets.layout_dataset import BaseLayoutDataset
from ..datamodel import HierarchicalLayout


class BasePredictor:
    def __init__(
        self,
        pdf_extractor: "PDFExtractor",
        block_predictor: "BaseBlockPredictor",
        block_classifier: "BaseBlockClassifier",
        model_name: str,
        visualizer=None,
    ):
        """The template class for pipeline models.

        The pipeline model consists of two stages:

            1. It first detects the blocks using `block_predictor` from the PDF images.
            2. It predicts the block categories using `block_classifier` from PDF tokens.

        When inheriting from this class, you need to define the actual
        `layout_extractor` class.

        Args:
            pdf_extractor (PDFExtractor):
                A PDF extractor defined in scienceparsepluse.pdftools
            block_predictor (BaseBlockPredictor):
                Block predictor(s) defined in .visual_model
            block_classifier (BaseBlockClassifier):
                Block classifier(s) defined in .layoutlm
            model_name (str):
                The name of the Predictor.
            visualizer (optional):
                Layout visualizer used for presenting the model predictions.
                Defaults to None.
        """
        self.pdf_extractor = pdf_extractor
        self.block_predictor = block_predictor
        self.block_classifier = block_classifier

        self.model_name = model_name
        self.visualizer = visualizer

    @abstractmethod
    def predict_page(
        self, page_image: "Image", token_info: Dict, **kwargs
    ) -> "HierarchicalLayout":
        """The main function for running inference on a PDF page,
        Needs to override.

        Args:
            page_image (Image)
            token_info (Dict):
                The page info, including the page height, width,
                and a list of the tokens on the page.

        Returns:
            HierarchicalLayout:
                The layout information stored in HierarchicalLayout.
        """

    def predict_pdf(
        self, pdf_filename: str, save_path: str = None, **kwargs
    ) -> List["HierarchicalLayout"]:
        """Run inference on all pages of a PDF file.

        Args:
            pdf_filename (str):
                The name of the PDF file.
            save_path (str, optional):
                When set, it will save the detected layout to the given folder.
                If self.visualizer is also set, it will save the layout detection
                visualization results in the `viz` folder:

                    The directory structure when visualizer is set:
                        `save_path`/
                        └───`pdf_name`/
                                ├───layout.json
                                └───viz/
                                    └───`page_idx`.jpg

                    The directory structure when visualizer is not set:
                        `save_path`/
                        └───`pdf_name`.json

        Returns:
            List["HierarchicalLayout"]:
                The layout information for each page stored in a list
                ordered the same as the original PDF.
        """
        pdf_page_info, page_images = self.pdf_extractor.load_tokens_and_image(
            pdf_filename, resize_image=True
        )

        if save_path is not None:
            base_save_path = os.path.join(
                save_path, os.path.splitext(os.path.basename(pdf_filename))[0]
            )

            if self.visualizer is not None:
                os.makedirs(f"{base_save_path}/viz/", exist_ok=True)

        results = []
        pbar = tqdm(enumerate(zip(pdf_page_info, page_images)), total=len(page_images))
        for page_idx, (token_info, page_image) in pbar:
            pbar.set_description(f"Working on Page {page_idx}")

            detected_layout = self.predict_page(
                page_image,
                token_info,
                **kwargs,
            )
            results.append(
                {
                    "height": token_info["height"],
                    "width": token_info["width"],
                    "index": page_idx,
                    "layout": detected_layout.to_dict(),
                }
            )

            if self.visualizer is not None and save_path is not None:

                viz = self.visualizer.create_viz(
                    page_image,
                    detected_layout.get_all_blocks_regions(),
                    detected_layout.get_all_tokens(),
                    model_name=self.model_name,
                )

                viz.save(
                    f"{base_save_path}/viz/{page_idx}.png", quality=100, optimized=True
                )

        if save_path is not None:
            if self.visualizer is not None:
                layout_save_path = f"{base_save_path}/layout.json"
            else:
                layout_save_path = f"{base_save_path}.json"

            with open(layout_save_path, "w") as fp:
                json.dump(results, fp)

        return results


class Evaluator:
    def __init__(
        self,
        predictor: BasePredictor,
        visualizer=None,
    ):
        """The evaluator of the pipeline layout detection model.

        Args:
            predictor (BasePredictor):
                The predictor (with the predict_page function overrode)
            visualizer (optional):
                Layout visualizer used for presenting the model predictions.
                Defaults to None.
        """
        self.predictor = predictor
        self.visualizer = visualizer

    @staticmethod
    def calculate_token_accuracy(gt: pd.DataFrame, pred: pd.DataFrame) -> float:
        """Calculate the token-level accuracy based on the gt and pred dataframes.
        The gt and pred dataframes should contain both the token ids and the
        categories ('gt' and 'pred') for each token.

        Returns:
            float: 0.~1. for token level accuracy for this page.
        """
        df = gt[["id", "gt"]].merge(pred[["id", "pred"]])
        df = df[~df["gt"].isna()]
        return (df["gt"] == df["pred"]).mean()

    def eval_page(
        self,
        page_image: "Image",
        token_info: Dict,
        gt_layout: "HierarchicalLayout",
        base_save_path: str = None,
        save_name: str = None,
        **kwargs,
    ):
        """Run evaluation for one PDF page

        Args:
            image (Image):
                The page image
            token_info (Dict):
                The page info, including the page height, width,
                and a list of the tokens on the page.
            gt_layout (HierarchicalLayout):
                The ground-truth layout for this page.
            base_save_path (str, optional):
                The path for saving the prediction results and visualization.
                If not set, both the model prediction and visualization won't be saved.
            save_name (str, optional):
                The filename for saving, intend to be used for differentiating
                different versions of models.
                If not set, will be the same as the model name.

        Returns:
            [type]: [description]
        """
        detected_layout = self.predictor.predict_page(
            page_image, token_info, gt_layout, **kwargs
        )

        pred = detected_layout.to_dataframe()
        gt = gt_layout.to_dataframe()

        save_name = save_name or self.predictor.model_name
        pred.to_csv(f"{base_save_path}/layout/{save_name}.csv", index=None)

        model_acc = self.calculate_token_accuracy(
            gt.rename(columns={"category": "gt"}),
            pred.rename(columns={"category": "pred"}),
        )

        if self.visualizer is not None:
            viz = self.visualizer.create_viz(
                page_image,
                detected_layout.get_all_blocks_regions(),
                detected_layout.get_all_tokens(),
                model_name=self.predictor.model_name,
                model_acc=model_acc,
            )

            if base_save_path is not None:
                viz.save(
                    f"{base_save_path}/viz/{save_name}.png", quality=100, optimized=True
                )
        return pred

    def eval(
        self,
        gt_dataset: BaseLayoutDataset,
        save_path: str,
        save_name: str = None,
        **kwargs,
    ):
        """Run evaluation for the gt_dataset.

        Args:
            gt_dataset (BaseLayoutDataset):
                The ground truth dataset inherited from BaseLayoutDataset.
            save_path (str, optional):
                When set, it will save the detected layout to the given folder.
                If self.visualizer is also set, it will save the layout detection
                visualization results in the `viz` folder:

                    `save_path`/
                     └───`pdf_name`/
                           └───`page_index`/
                                ├───layout/
                                |   └───save_name.csv
                                └───viz/
                                    └───save_name.jpg

            save_name (str, optional):
                The filename for saving, intend to be used for differentiating
                different versions of models.
                If not set, will be the same as the model name.
        """

        for pdf_filename, gt_layouts, page_images, pdf_page_info in tqdm(gt_dataset):

            base_save_path = os.path.join(
                save_path, os.path.splitext(os.path.basename(pdf_filename))[0]
            )

            for page_idx, (gt_layout, page_image, token_info) in enumerate(
                zip(gt_layouts, page_images, pdf_page_info)
            ):

                cur_save_path = f"{base_save_path}/{page_idx}"
                os.makedirs(f"{cur_save_path}/layout/", exist_ok=True)
                os.makedirs(f"{cur_save_path}/viz/", exist_ok=True)

                self.eval_page(
                    page_image,
                    token_info,
                    gt_layout,
                    cur_save_path,
                    save_name,
                    **kwargs,
                )


class V2ModelPredictor(BasePredictor):
    """The Visual Block Detection + Block Text Classification model:

    1. Detects the visual blocks using the `block_predictor`
    2. For each textual block, selects all the tokens inside the blocks
        and predict the block category using the block text classification
        models
    """

    textual_blocks = ["text", "title", "list"]

    def predict_page(
        self, page_image: "Image", token_info: Dict
    ) -> "HierarchicalLayout":

        block_predictorA, block_predictorB = self.block_predictor

        # Detecting Blocks from images
        general_layouts = block_predictorA.detect(page_image, token_info["tokens"])
        equation_layouts = block_predictorB.detect(
            page_image, general_layouts.remaining_tokens
        )

        merged_layouts = HierarchicalLayout(
            bundles=general_layouts.bundles + equation_layouts.bundles,
            remaining_tokens=equation_layouts.remaining_tokens,
        )

        # Predicting block text categories
        block_classifier = self.block_classifier

        for bundle in merged_layouts.bundles:
            if bundle.block.type not in self.textual_blocks:
                continue

            width, height = token_info["width"], token_info["height"]
            detected_tokens = block_classifier.detect(bundle.tokens, width, height)

            bundle.block.type = detected_tokens[0].type
            bundle.tokens = detected_tokens

        merged_layouts.set_tokens_with_block_class()
        return merged_layouts


class V4ModelPredictor(BasePredictor):
    """The Visual Block Detection + Token Classification Voting model:

    1. Predicts the token categories using the `block_classifier`
    2. Detects the visual blocks using the `block_predictor`
    3. For each textual block, relabel the category based on the most
        common token categories.
    """

    textual_blocks = ["text", "title", "list"]

    def predict_page(
        self, page_image: "Image", token_info: Dict
    ) -> "HierarchicalLayout":

        width, height = token_info["width"], token_info["height"]
        tokens = token_info["tokens"]

        tokens = self.block_classifier.detect(tokens, width, height)

        block_predictorA, block_predictorB = self.block_predictor

        # Detecting Blocks from images
        general_layouts = block_predictorA.detect(page_image, tokens)
        equation_layouts = block_predictorB.detect(
            page_image, general_layouts.remaining_tokens
        )

        merged_layouts = HierarchicalLayout(
            bundles=general_layouts.bundles + equation_layouts.bundles,
            remaining_tokens=[
                tok.set(type=None) for tok in equation_layouts.remaining_tokens
            ],
        )

        for bundle in merged_layouts.bundles:
            if bundle.block.type not in self.textual_blocks:
                continue
            most_common_type = Counter(
                [tok.type for tok in bundle.tokens]
            ).most_common()[0][0]
            bundle.block.type = most_common_type

        merged_layouts.set_tokens_with_block_class()

        return merged_layouts


class V6ModelPredictor(BasePredictor):
    """The Visual Block Detection + LayoutLM Block Embedding Model:

    1. Detects the visual blocks using the `block_predictor`
    2. Build the block index using the visual blocks
    3. Predicts the token categories using the `block_classifier`,
        which is based on LayoutLM Block Embedding Model.
    """

    textual_blocks = ["text", "title", "list"]

    def predict_page(
        self, page_image: "Image", token_info: Dict, sort_tokens: bool = False
    ) -> "HierarchicalLayout":
        """
        Args:
            sort_tokens (bool):
                Whether to sort the tokens based on the original PDF
                Extraction orders.
        """
        width, height = token_info["width"], token_info["height"]
        tokens = [ele.set(id=idx) for idx, ele in enumerate(token_info["tokens"])]

        block_predictorA, block_predictorB = self.block_predictor

        # Detecting Blocks from images
        general_layouts = block_predictorA.detect(page_image, tokens)
        equation_layouts = block_predictorB.detect(
            page_image, general_layouts.remaining_tokens
        )

        merged_layouts = HierarchicalLayout(
            bundles=general_layouts.bundles + equation_layouts.bundles,
            remaining_tokens=[
                tok.set(type=None) for tok in equation_layouts.remaining_tokens
            ],
        )

        blocks = [bd.block for bd in merged_layouts.bundles]
        tokens = merged_layouts.get_all_tokens()

        if sort_tokens:
            tokens = sorted(tokens, key=lambda x: x.id)
        tokens = self.block_classifier.detect(tokens, width, height)

        final_layouts = HierarchicalLayout.from_raw_block_token(blocks, tokens)
        for bundle in final_layouts.bundles:
            if bundle.block.type in self.textual_blocks:
                bundle.block.type = bundle.tokens[0].type
        return final_layouts