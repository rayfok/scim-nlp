from typing import List, Union, Dict, Any, Tuple
from glob import glob
import argparse
import os
import json
import re
import sys

import cv2
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import layoutparser as lp
import pandas as pd
import numpy as np

from ..pdftools import *
from ..datamodel import *


class NpEncoder(json.JSONEncoder):
    "A trick learned from https://stackoverflow.com/a/57915246"

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def load_coco_annotations_as_layout(
    coco, image_ids, replace_categories=False
) -> lp.Layout:

    annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_ids))

    layout = lp.Layout()

    for ele in annotations:

        x, y, w, h = ele["bbox"]

        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x, y, w + x, h + y),
                type=ele["category_id"]
                if not replace_categories
                else coco.cats[ele["category_id"]]["name"],
                id=ele["id"],
            )
        )

    return layout


def cvt_token_level_gp(gp):
    row_data = gp.apply(
        lambda row: (row["text"], [row["x_1"], row["y_1"], row["x_2"], row["y_2"]], row["font"]),
        axis=1,
    )
    return {
        "words": [ele[0] for ele in row_data],
        "bbox": [ele[1] for ele in row_data],
        "label": gp.iloc[0]["category"],
        "block_id": int(gp.iloc[0]["parent"]),
        "fonts":  [ele[2] for ele in row_data]
    }


class PubLayNetDataset:
    """Used for processing the original version of the publaynet dataset

    The directory structure is shown as follows:
        `base_path`/
        ├───publaynet/
        │     ├─── `subset`.json
        │     └─── `subset`/
        └───`pdfs`/
            └─── `subset`/

    Subsets: {"train", "dev", "test}
    """

    NON_CATEGORY_TOKEN_TYPE_ID = 6
    """
    """
    FONT_CLEAN_PATTERN = re.compile(r'(\S{6})\+')

    def __init__(
        self, base_path: str, subset: str, pdf_extractor_name: str = "PDFPlumber"
    ):
        self.base_path = base_path
        self.subset = subset

        self.pdf_dir = f"{base_path}/pdfs/{subset}"
        self.annotation_file = f"{base_path}/publaynet/{subset}.json"
        self.image_dir = f"{base_path}/publaynet/{subset}"

        self.coco = COCO(self.annotation_file)
        assert len(self.coco.imgs) == len(os.listdir(self.pdf_dir))
        self.filename2cocoid = {
            val["file_name"].rstrip(".jpg"): val["id"]
            for key, val in self.coco.imgs.items()
        }

        self.categorystr2id = {
            ele["name"]: ele["id"] for ele in self.coco.cats.values()
        }
        self.categorystr2id['empty'] = self.NON_CATEGORY_TOKEN_TYPE_ID

        self.categoryid2str = {
            ele["id"]: ele["name"] for ele in self.coco.cats.values()
        }
        self.categoryid2str[self.NON_CATEGORY_TOKEN_TYPE_ID] = 'empty'

        self.pdf_extractor = PDFExtractor("PDFPlumber")

    def get_image_info_as_df(self) -> pd.DataFrame:
        all_img_info = []
        for img_info in self.coco.imgs.values():
            filename, page_index = img_info["file_name"].rstrip(".jpg").split("_")
            all_img_info.append(
                [
                    img_info["id"],
                    filename,
                    int(page_index),
                    img_info["width"],
                    img_info["height"],
                ]
            )
        df = pd.DataFrame(
            all_img_info, columns=["id", "filename", "index", "width", "height"]
        )
        return df

    def load_image_from_filename(self, filename) -> Image:
        return Image.open(f"{self.image_dir}/{filename}.jpg")

    def iter_all_pdfs(self) -> Tuple[str, str, int]:
        for filepath in sorted(glob(f"{self.pdf_dir}/*.pdf")):
            filename = os.path.basename(filepath).rstrip(".pdf")
            coco_id = self.filename2cocoid[filename]
            yield (filepath, filename, coco_id)

    def load_hierarchical_layout_for_pdf(
        self, filepath, filename, coco_id, replace_categories=False
    ) -> HierarchicalLayout:

        coco_info = self.coco.imgs[coco_id]
        img_width, img_height = coco_info["width"], coco_info["height"]

        # Load the token data
        token_info = self.pdf_extractor.pdf_extractor(filepath)
        if len(token_info) != 1:
            print(f"warning {filename}")
        token_info = token_info[0]

        # Get the size right
        token_width, token_height = token_info["width"], token_info["height"]
        scale_factor = (img_width / token_width, img_height / token_height)
        tokens = token_info["tokens"].scale(scale_factor)

        # Load layout structure
        blocks = load_coco_annotations_as_layout(
            self.coco, [coco_id], replace_categories=replace_categories
        )
        layout = HierarchicalLayout.from_raw_block_token(
            block_layout=blocks, token_layout=tokens
        )

        # Ensure the bundles are appropriately ordered
        layout.bundles = sorted(layout.bundles, key=lambda x: x.tokens[0].id)

        return layout

    def generate_token_mask(self, canvas_size: Tuple, tokens: List) -> np.ndarray:

        width, height = canvas_size
        canvas = np.zeros((height, width), dtype=np.uint8)
        for ele in tokens:
            x1, y1, x2, y2 = ele.coordinates
            type_id = (
                ele.type if ele.type is not None else self.NON_CATEGORY_TOKEN_TYPE_ID
            )
            canvas[int(y1) : int(y2), int(x1) : int(x2)] = type_id
        return canvas

    def generate_token_data(
        self, save_folder, save_token_mask=False, debug=False, overwrite=False,
    ) -> Tuple[Dict, Dict]:

        if debug:
            save_folder = f"{save_folder}/debug"

        token_layout_save_folder = f"{save_folder}/token-layout/{self.subset}"
        grouped_json_save_folder = f"{save_folder}/grouped"
        token_mask_save_folder = f"{save_folder}/token-mask/{self.subset}"

        os.makedirs(token_layout_save_folder, exist_ok=True)
        os.makedirs(grouped_json_save_folder, exist_ok=True)
        if save_token_mask:
            os.makedirs(token_mask_save_folder, exist_ok=True)

        token_level_data = []
        segment_level_data = []

        cnt = 0
        for (filepath, filename, coco_id) in tqdm(
            self.iter_all_pdfs(), total=len(self.coco.imgs)
        ):

            coco_info = self.coco.imgs[coco_id]
            width, height = coco_info["width"], coco_info["height"]

            try:
                layout = self.load_hierarchical_layout_for_pdf(
                    filepath, filename, coco_id
                )
            except KeyboardInterrupt:
                sys.exit()
            except:
                print(f"Unable to read PDF {filename}")
                continue

            if save_token_mask and (overwrite or not os.path.exists(
                f"{token_mask_save_folder}/{filename}.png"
            )):
                token_mask = self.generate_token_mask(
                    (width, height), layout.get_all_tokens(inherit_block_class=True)
                )
                cv2.imwrite(f"{token_mask_save_folder}/{filename}.png", token_mask)

            df = layout.to_dataframe(inherit_block_class=True).drop(
                columns=["confidence"]
            )
            df[["x_1", "x_2"]] = df[["x_1", "x_2"]] * 1000 / width
            df[["y_1", "y_2"]] = df[["y_1", "y_2"]] * 1000 / height
            df[["x_1", "y_1", "x_2", "y_2"]] = df[["x_1", "y_1", "x_2", "y_2"]].astype(
                "int"
            )  # Convert to relative coordinates
            df = df[~df.is_block]
            df["category"] = df["category"].fillna(self.NON_CATEGORY_TOKEN_TYPE_ID).astype("int") 
            df.to_csv(f"{token_layout_save_folder}/{filename}.csv", index=None)

            df['font'] = df['font'].str.replace(self.FONT_CLEAN_PATTERN, '')
            df["parent"] = df["parent"].fillna(-1).astype("int")
            
            # Save the token-level data for grouped JOSN
            df = df.sort_values("id")
            row_item = {
                "words": df["text"].tolist(),
                "bbox": df.apply(
                    lambda row: (row["x_1"], row["y_1"], row["x_2"], row["y_2"]), axis=1
                ).tolist(),
                "labels": df["category"].tolist(),
                "block_ids": df["parent"].tolist(),
                "fonts": df['font'].tolist()
            }
            token_level_data.append(row_item)

            segment_data = (
                df[df['parent']!= -1]
                .groupby("parent")
                .apply(cvt_token_level_gp)
                .tolist()
            )
            segment_level_data.extend(segment_data)
            cnt += 1
            if debug and cnt >= 10:
                break

            del df

        token_level_data = {"data": token_level_data, "labels": self.categoryid2str}
        segment_level_data = {"data": segment_level_data, "labels": self.categoryid2str}

        with open(f"{grouped_json_save_folder}/{self.subset}-token.json", "w") as fp:
            json.dump(token_level_data, fp, cls=NpEncoder)

        with open(f"{grouped_json_save_folder}/{self.subset}-segment.json", "w") as fp:
            json.dump(segment_level_data, fp, cls=NpEncoder)

        return token_level_data, segment_level_data