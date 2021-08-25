from typing import List
from copy import copy
import os
import sys
import logging
import json
import argparse
import configparser
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd
import layoutparser as lp
from pycocotools.coco import COCO

sys.path.append("../src/")
from scienceparseplus import pdftools
from scienceparseplus.modeling.layoutlm import LayoutLMTokenPredictor

parser = argparse.ArgumentParser()

# fmt: off
parser.add_argument('--config', type=str, default='./config.ini')
parser.add_argument('--subset', type=str, default='dev')
# fmt: on


UNSPECIFIED_BLOCK_ID = -1
UNSPECIFIED_BLOCK_TYPE = -1


def is_in(block_a, block_b):
    """A rewrite of the is_in function.
    We will use a soft_margin and center function by default.
    """
    return block_a.is_in(
        block_b, soft_margin={"top": 1, "bottom": 1, "left": 1, "right": 1}, center=True
    )


def find_token_in_block(token_layout: List, block_layout: List):
    """For each token in token_layout, it finds the parent block and
    saves the parent block's id and type."""

    new_token_layout = lp.Layout()

    for token in token_layout:
        find_block = False
        token = copy(token)

        for block in block_layout:
            if is_in(token, block):
                find_block = True
                token.block_type = block.type
                token.block_id = block.id
                continue

        if not find_block:
            token.block_type = UNSPECIFIED_BLOCK_ID
            token.block_id = UNSPECIFIED_BLOCK_TYPE

        new_token_layout.append(token)

    return new_token_layout


def group_tokens(token_layout: List, block_layout: List):
    """Besides the functionality in find_token_in_block,
    it also groups tokens based on the blocks"""

    token_groups = []

    for block in block_layout:
        token_group = []
        remaining_tokens = []
        for token in token_layout:
            if is_in(token, block):
                token = copy(token)
                token.block_type = block.type
                token.block_id = block.id
                token_group.append(token)
            else:
                remaining_tokens.append(token)

        token_groups.append(token_group)
        token_layout = remaining_tokens

    unselected_tokens = []
    for token in remaining_tokens:
        token = copy(token)
        token.block_type = UNSPECIFIED_BLOCK_ID
        token.block_id = UNSPECIFIED_BLOCK_TYPE
        unselected_tokens.append(token)

    return token_groups, remaining_tokens


def load_coco_annotations_as_layout(coco, image_ids, replace_categories=False):

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


class PubLayNetProcessor:
    def __init__(self, configs, subset):

        self.subset = subset

        self.source_pdf_path = os.path.join(configs["source_pdf_path"], subset)
        self.save_base_path = os.path.join(configs["save_base_path"], subset)

        if not os.path.exists(self.save_base_path):
            os.makedirs(self.save_base_path)
            print(f"Creating saving directory {self.save_base_path}")

        self.coco = COCO(os.path.join(configs["coco_path"], f"{subset}.json"))

        pdf_extractor_name = configs.get("pdf_extractor_name", "PDFPlumber")
        if pdf_extractor_name == "PDFPlumber":
            self.pdf_extractor = pdftools.PDFPlumberTokenExtractor()

        self.layoutlm_predictor = LayoutLMTokenPredictor(
            configs["layout_lm_weight_paths"], max_seq_length=512,
            tokenizer_path=configs["layout_lm_weight_paths"]
        )

        self.error_log_file = os.path.join(self.save_base_path, "error_log.csv")

    def obtain_pdf_filepath(self, pdf_info):

        file_name = os.path.splitext(pdf_info["file_name"])[0]
        pdf_name, pdf_page_index = file_name.split("_")

        pdf_filename = f"{self.source_pdf_path}/{file_name}.pdf"

        return pdf_filename, pdf_name, pdf_page_index

    def create_token_dataframe(self, token_layout):

        return pd.DataFrame(
            [
                (*b.coordinates, b.text, b.type, b.style, b.block_id, b.block_type)
                for b in token_layout
            ],
            columns=[
                "x_1",
                "y_1",
                "x_2",
                "y_2",
                "text",
                "token_type",
                "token_style",
                "block_id",
                "block_type",
            ],
        )

    def create_block_dataframe(self, block_layout):

        return pd.DataFrame(
            [(b.id, *b.coordinates, b.type) for b in block_layout],
            columns=["id", "x_1", "y_1", "x_2", "y_2", "type"],
        )

    def record_errors(self, error_type, pdf_name):

        error_log = pd.DataFrame([{"error_type": error_type, "pdf_name": pdf_name}])
        error_log.to_csv(self.error_log_file, mode="a", header=None)

    def process_pages(self, num_cpus=8):

        print("Start Extracting Page Tokens")
        # fmt: off
        with Pool(processes=num_cpus) as p:
            with tqdm(total=len(self.coco.imgs)) as pbar:
                for i, _ in enumerate(p.imap_unordered(self.extract_page_token, self.coco.imgs.items())): 
                    pbar.update()
        # fmt: on

        print("Start LayoutLM Predictions")
        
        pbar = tqdm(self.coco.imgs.items())
        for pdf_id, pdf_info in pbar:
            filename = pdf_info['file_name']
            
            try:
                self.extract_token_category(pdf_id, pdf_info)

            except:
                logging.error(f"Error-5 for {pdf_filename} | Error when processing LayoutLM Predictions")
                self.record_errors(5, pdf_filename)
                continue

            pbar.set_description(f"Processing {filename}...")

    def extract_page_token(self, input_info):

        pdf_id, pdf_info = input_info

        pdf_filename, pdf_name, pdf_page_index = self.obtain_pdf_filepath(pdf_info)

        if not os.path.exists(pdf_filename):
            logging.error(f"Error-1 for {pdf_filename} | Non-existent pdf file")
            self.record_errors(1, pdf_filename)
            return

        try:
            pdf_structures = self.pdf_extractor.extract(
                pdf_filename, include_lines=False
            )
        except:
            logging.error(f"Error-2 for {pdf_filename} | Fail to parse PDF file")
            self.record_errors(2, pdf_filename)
            return

        try:
            pdf_layouts = pdftools.cvt_pdfstructure_to_layout(
                pdf_structures, self.pdf_extractor.NAME
            )
        except:
            logging.error(f"Error-3 for {pdf_filename} | PDF Structure service error")
            self.record_errors(3, pdf_filename)
            return

        if len(pdf_layouts) != 1:
            logging.error(f"Error-4 for {pdf_filename} | Strange documents with more than one page")
            self.record_errors(4, pdf_filename)
            return

        pdf_layout = pdf_layouts[0]
        token_layout = pdf_layout["layout"]

        publaynet_layout = load_coco_annotations_as_layout(self.coco, pdf_id)
        final_token_layout = find_token_in_block(token_layout, publaynet_layout)

        token_df = self.create_token_dataframe(final_token_layout)
        block_df = self.create_block_dataframe(publaynet_layout)
        style_json = pdf_structures.tokens.sources[self.pdf_extractor.NAME].styles

        self.save_page_data(
            token_df,
            block_df,
            style_json,
            pdf_name,
            pdf_page_index,
            pdf_layout["width"],
            pdf_layout["height"],
        )

    def extract_token_category(self, pdf_id, pdf_info):

        pdf_filename, pdf_name, pdf_page_index = self.obtain_pdf_filepath(pdf_info)
        save_path = f"{self.save_base_path}/{pdf_name}/{pdf_page_index}"
        
        if not os.path.exists(f"{save_path}/tokens.csv"):
            print(f"Skip for non-existent token file for {pdf_filename}")
            return 

        token_df = pd.read_csv(f"{save_path}/tokens.csv")
        token_df['text'] = token_df['text'].apply(str)
        size = pd.read_csv(f"{save_path}/size.csv", index_col=0)["0"]

        layout = lp.Layout.from_dataframe(token_df)
        res = self.layoutlm_predictor.detect(layout, size['width'], size['height'])
        token_df['token_type'] = [b.type for b in res]
        token_df.to_csv(f"{save_path}/tokens.csv", index=None)

    def save_page_data(
        self,
        token_df,
        block_df,
        style_json,
        pdf_name,
        pdf_page_index,
        pdf_width,
        pdf_height,
    ):

        save_path = f"{self.save_base_path}/{pdf_name}/{pdf_page_index}"
        if os.path.exists(save_path):
            logging.warning(f"Error-3 | The path {save_path} already exists")
        os.makedirs(save_path, exist_ok=True)
        token_df.to_csv(f"{save_path}/tokens.csv", index=None)
        block_df.to_csv(f"{save_path}/blocks.csv", index=None)
        pd.DataFrame(style_json).T.to_csv(f"{save_path}/styles.csv")
        pd.Series([pdf_width, pdf_height], index=["width", "height"]).to_csv(
            f"{save_path}/size.csv"
        )


if __name__ == "__main__":

    args = parser.parse_args()

    FORMAT = "[%(asctime)-15s | %(levelname)s] %(message)s"
    logging.basicConfig(
        filename=f"./{args.subset}.log", level=logging.WARNING, format=FORMAT
    )

    config = configparser.ConfigParser()
    config.read(args.config)

    processor = PubLayNetProcessor(config["Processor Settings"], args.subset)
    processor.process_pages()
