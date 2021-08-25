from typing import List, Union, Dict, Any, Tuple
from abc import abstractmethod
import os

import layoutparser as lp
from pdf2image import convert_from_path
from pycocotools.coco import COCO

from ..datamodel import HierarchicalLayout
from ..pdftools import PDFExtractor


class BaseLayoutDataset:
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __getitem__(
        self, idx
    ) -> Tuple[str, List["HierarchicalLayout"], List["Image"], List[Dict]]:
        """Return the following data for the idx-th pdf_file
        in this dataset:
            1. The pdf filename
            2. A list page layout data ordered by page index
            3. A list of page images that are resized as the page width and height
            4. A list of pdf token information for each page in the pdf.
        """

    @abstractmethod
    def __len__(self):
        """The number of pdfs in this dataset"""

    def export_and_visualize(self, visualizer, save_path: str):
        """Save and visualize the gt dataset to the target directory."""

        for pdf_filename, pdf_layouts, page_images, pdf_page_info in self:

            base_save_path = os.path.join(
                save_path, os.path.splitext(os.path.basename(pdf_filename))[0]
            )

            for page_idx, (page_image, page_layout, page_image) in enumerate(
                (
                    page_images,
                    pdf_layouts,
                    page_images,
                )
            ):
                cur_save_path = f"{base_save_path}/{page_idx}"
                os.makedirs(f"{cur_save_path}/layout/", exist_ok=True)
                os.makedirs(f"{cur_save_path}/viz/", exist_ok=True)

                page_layout.to_dataframe().to_csv(
                    f"{cur_save_path}/layout/gt.csv", index=None
                )

                viz = visualizer.create_viz(
                    page_image,
                    page_layout.get_all_blocks_regions(),
                    page_layout.get_all_tokens(),
                    model_name="gt",
                )
                viz.save(f"{cur_save_path}/viz/gt.png", quality=100, optimized=True)


class COCOLayoutDataset(BaseLayoutDataset):
    def __init__(
        self,
        coco_base_path: str,
        image_folder_name: str = "images",
        pdf_folder_name: str = "pdfs",
        coco_annotation_name: str = "annotations.json",
        allowed_cat_ids: List = [],
        use_numerical_id: bool = False,
        category_name_mapping={},
    ):
        """The COCOLayoutDataset dataloder.

        Args:
            coco_base_path (str):
                The base path for the COCO dataset.
            image_folder_name (str, optional): Defaults to "images".
            pdf_folder_name (str, optional): Defaults to "pdfs".
            coco_annotation_name (str, optional): Defaults to "annotations.json".
            coco_path (str):
                The path to the COCO annotation.

            allowed_cat_ids (List, optional):
                If set, only loads blocks with categories of the given ids.
                Defaults to [].
            use_numerical_id (bool, optional):
                If set, enforce the exported labels are numerical values.
                Defaults to False.
            category_name_mapping (Dict, optional):
                If set, it will convert the names in the COCO dataset according
                to the mapping.
        """
        self.coco_base_path = coco_base_path
        self.coco_image_path = f"{coco_base_path}/{image_folder_name}"
        self.coco_pdf_path = f"{coco_base_path}/{pdf_folder_name}"
        self.coco_anno_path = f"{coco_base_path}/{coco_annotation_name}"
        self.coco = COCO(self.coco_anno_path)

        self.allowed_cat_ids = allowed_cat_ids
        self.use_numerical_id = use_numerical_id
        self.category_name_mapping = category_name_mapping

        self.pdf_extractor = PDFExtractor("pdfplumber")
        # TODO: In the future versions of the COCO dataset, we should also include
        # token level information and the pdf_extractor should be removed

    def load_coco_annotations(self, image_id: int) -> lp.Layout:
        """Helper functions for loading COCO annotations as lp.Layout
        for a given image ids.

        Args:
            image_id (int): The image id in the COCO dataset
        """
        annotations = self.coco.loadAnns(
            self.coco.getAnnIds(imgIds=[image_id], catIds=self.allowed_cat_ids)
        )
        layout = lp.Layout()

        for ele in annotations:

            x, y, w, h = ele["bbox"]

            if self.use_numerical_id:
                cat = ele["category_id"]
            else:
                cat = self.coco.cats[ele["category_id"]]["name"]
                cat = self.category_name_mapping.get(cat, None)

            layout.append(
                lp.TextBlock(
                    block=lp.Rectangle(x, y, w + x, h + y).pad(
                        left=-2, right=-2, top=-2, bottom=-2
                    ),
                    type=cat,
                    id=ele["id"],
                )
            )

        return layout

    def get_all_image_info_for_paper(self, paper_info: Dict) -> List:
        return [
            img
            for img in self.coco.dataset["images"]
            if img["paper_id"] == paper_info["id"]
        ]

    def get_paper_filename(self, paper_info: Dict) -> str:
        return f"{self.coco_pdf_path}/{paper_info['paper_sha']}.pdf"

    def load_gt_layout_for_image_id(self, image_id, token_info) -> HierarchicalLayout:
        blocks = self.load_coco_annotations(image_id)
        gt_layout = HierarchicalLayout.from_raw_block_token(
            blocks, token_info["tokens"]
        )
        gt_layout.set_tokens_with_block_class()
        return gt_layout

    def __len__(self) -> int:
        return len(self.coco.dataset["papers"])

    def __getitem__(
        self, idx
    ) -> Tuple[str, List["HierarchicalLayout"], List["Image"], List[Dict]]:

        paper_info = self.coco.dataset["papers"][idx]

        paper_filename = self.get_paper_filename(paper_info)
        pdf_page_info, page_images = self.pdf_extractor.load_tokens_and_image(
            paper_filename, resize_image=True
        )

        all_image_info = self.get_all_image_info_for_paper(paper_info)
        all_image_info = {
            image_info["page_number"]: image_info["id"] for image_info in all_image_info
        }

        gt_layouts = []
        for page_index, token_info in enumerate(pdf_page_info):

            if page_index not in all_image_info:
                gt_layouts.append(HierarchicalLayout([], []))

            image_id = all_image_info[page_index]
            gt_layouts.append(self.load_gt_layout_for_image_id(image_id, token_info))

        return (paper_filename, gt_layouts, page_images, pdf_page_info)