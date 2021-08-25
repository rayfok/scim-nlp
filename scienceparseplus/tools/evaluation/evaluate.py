import sys
import os
from abc import abstractmethod

import numpy as np
import layoutparser as lp
from tqdm import tqdm

sys.path.append("../../src/")
from scienceparseplus.pdftools import *
from scienceparseplus.modeling.layoutlm import *
from scienceparseplus.modeling.visual_model import *


class COCOGroundTruth(COCOGroundTruthBlockPredictor):
    def __init__(
        self,
        coco_base,
        coco_image_base="images",
        coco_pdf_base="pdfs",
        coco_annotations_name="annotations.json",
        **kwargs,
    ):
        self.coco_base = coco_base
        self.coco_image_base = f"{coco_base}/{coco_image_base}"
        self.coco_pdf_base = f"{coco_base}/{coco_pdf_base}"
        self.coco_path = f"{coco_base}/{coco_annotations_name}"
        super().__init__(self.coco_path, **kwargs)

    @property
    def all_papers(self):
        return self.coco.dataset["papers"]

    def get_paper_filename(self, paper):
        return f"{self.coco_pdf_base}/{paper['paper_sha']}.pdf"

    def get_all_images_for_paper(self, paper):
        return [
            img for img in self.coco.dataset["images"] if img["paper_id"] == paper["id"]
        ]

    def load_gt_for_image_id(self, image_id, pdf_info):

        page_layout = self.detect(image_id, pdf_info["tokens"])
        page_layout.set_tokens_with_block_class()
        return page_layout

    def export_and_visualize(self, pdf_extractor, visualizer, save_path):

        for paper in tqdm(self.all_papers):

            paper_filename = self.get_paper_filename(paper)
            all_images = self.get_all_images_for_paper(paper)

            pdf_page_info, page_images = pdf_extractor.extract(
                paper_filename, resize_image=True
            )
            base_save_path = os.path.join(
                save_path, os.path.splitext(os.path.basename(paper_filename))[0]
            )

            for cur_image in all_images:

                page_idx = cur_image["page_number"]
                cur_save_path = f"{base_save_path}/{page_idx}"
                os.makedirs(f"{cur_save_path}/layout/", exist_ok=True)
                os.makedirs(f"{cur_save_path}/viz/", exist_ok=True)

                pdf_info = pdf_page_info[page_idx]
                page_image = page_images[page_idx]

                gt_layout = self.load_gt_for_image_id(cur_image["id"], pdf_info)

                gt_layout.to_dataframe().to_csv(
                    f"{cur_save_path}/layout/gt.csv", index=None
                )

                viz = visualizer.create_viz(
                    page_image,
                    gt_layout.get_all_blocks_regions(),
                    gt_layout.get_all_tokens(),
                    model_name="gt",
                )
                viz.save(f"{cur_save_path}/viz/gt.png", quality=100, optimized=True)


class BaseModelEvaluator:
    def __init__(
        self,
        pdf_extractor,
        gt_dataset: COCOGroundTruth,
        block_predictor: BaseBlockPredictor,
        block_classifier,
        model_name,
        visualizer=None,
    ):
        self.gt_dataset = gt_dataset
        self.pdf_extractor = pdf_extractor

        self.block_predictor = block_predictor
        self.block_classifier = block_classifier
        self.model_name = model_name
        self.visualizer = visualizer

    @abstractmethod
    def layout_extractor(self, page_image, pdf_info, gt_layout, **kwargs):
        "The layout-extractor should be defined "

    @staticmethod
    def calculate_accuracy(gt, pred):

        df = gt[["id", "gt"]].merge(pred[["id", "pred"]])
        df = df[~df["gt"].isna()]
        return (df["gt"] == df["pred"]).mean()

    def eval_page(
        self,
        page_image,
        pdf_info,
        gt_layout,
        cur_save_path,
        save_name=None,
        **kwargs,
    ):

        detected_layout = self.layout_extractor(
            page_image, pdf_info, gt_layout, **kwargs
        )

        pred = detected_layout.to_dataframe()
        gt = gt_layout.to_dataframe()

        save_name = self.model_name if save_name is None else save_name
        pred.to_csv(f"{cur_save_path}/layout/{save_name}.csv", index=None)

        model_acc = self.calculate_accuracy(
            gt.rename(columns={"category": "gt"}),
            pred.rename(columns={"category": "pred"}),
        )

        if self.visualizer is not None:
            viz = self.visualizer.create_viz(
                page_image,
                detected_layout.get_all_blocks_regions(),
                detected_layout.get_all_tokens(),
                model_name=self.model_name,
                model_acc=model_acc,
            )

            viz.save(
                f"{cur_save_path}/viz/{save_name}.png", quality=100, optimized=True
            )
        return pred

    def eval(self, save_path, save_name=None, **kwargs):

        for paper in tqdm(self.gt_dataset.all_papers):

            paper_filename = self.gt_dataset.get_paper_filename(paper)
            all_images = self.gt_dataset.get_all_images_for_paper(paper)

            pdf_page_info, page_images = self.pdf_extractor.extract(
                paper_filename, resize_image=True
            )
            base_save_path = os.path.join(
                save_path, os.path.splitext(os.path.basename(paper_filename))[0]
            )

            for cur_image in all_images:

                page_idx = cur_image["page_number"]
                cur_save_path = f"{base_save_path}/{page_idx}"
                os.makedirs(f"{cur_save_path}/layout/", exist_ok=True)
                os.makedirs(f"{cur_save_path}/viz/", exist_ok=True)

                pdf_info = pdf_page_info[page_idx]
                page_image = page_images[page_idx]

                gt_layout = self.gt_dataset.load_gt_for_image_id(
                    cur_image["id"], pdf_info
                )

                self.eval_page(
                    page_image,
                    pdf_info,
                    gt_layout,
                    cur_save_path,
                    save_name,
                    **kwargs,
                )