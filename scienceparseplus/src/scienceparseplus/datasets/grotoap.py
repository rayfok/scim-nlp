from typing import List, Union, Dict, Any, Tuple
from dataclasses import dataclass
from glob import glob
import os
import itertools
from collections import Counter

from tqdm import tqdm
from bs4 import BeautifulSoup
import layoutparser as lp
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from scipy.spatial.distance import cdist

from ..datamodel import HierarchicalLayout
from ..pdftools import PDFPlumberTokenExtractor

DISTANCE_THRESHOLD = 50


class ProgressParallel(Parallel):
    """
    A techinque learned from https://stackoverflow.com/a/61027781
    that enables parallel with tqdm.

    * But actually not used as it doesn't display the total iter
    number correctly and regularly updates when n_dispatched_tasks
    changes.
    """

    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


@dataclass
class PageData:
    blocks: List[lp.TextBlock]
    lines: List[lp.TextBlock]
    words: List[lp.TextBlock]

    def to_dataframe(
        self,
        keep_token_index=True,
        export_font=False,
        normaize_coordinates=False,
        canvas_width=None,
        canvas_height=None,
    ) -> pd.DataFrame:

        if not export_font:
            blocks_to_save = [
                [
                    ele.id,
                    *ele.coordinates,
                    ele.text,
                    ele.type,
                    -1,
                    -1,
                    True,
                    False,
                ]
                for ele in self.blocks
            ]
            lines_to_save = [
                [
                    ele.id,
                    *ele.coordinates,
                    ele.text,
                    ele.type,
                    ele.parent,
                    -1,
                    False,
                    True,
                ]
                for ele in self.lines
            ]
            parent_block_id_for_line_id = {ele.id: ele.parent for ele in self.lines}
            tokens_to_save = [
                [
                    ele.id if keep_token_index else idx,
                    *ele.coordinates,
                    ele.text,
                    ele.type,
                    parent_block_id_for_line_id[ele.parent],  # Cvt to block-level id
                    ele.parent,
                    False,
                    False,
                ]
                for idx, ele in enumerate(self.words, start=len(blocks_to_save))
            ]
            df = pd.DataFrame(
                blocks_to_save + lines_to_save + tokens_to_save,
                columns=[
                    "id",
                    "x_1",
                    "y_1",
                    "x_2",
                    "y_2",
                    "text",
                    "category",
                    "block_id",
                    "line_id",
                    "is_block",
                    "is_line",
                ],
            )
        else:
            blocks_to_save = [
                [
                    ele.id,
                    *ele.coordinates,
                    ele.text,
                    None,
                    ele.type,
                    -1,
                    -1,
                    True,
                    False,
                ]
                for ele in self.blocks
            ]
            lines_to_save = [
                [
                    ele.id,
                    *ele.coordinates,
                    ele.text,
                    None,
                    ele.type,
                    ele.parent,
                    -1,
                    False,
                    True,
                ]
                for ele in self.lines
            ]
            parent_block_id_for_line_id = {ele.id: ele.parent for ele in self.lines}
            tokens_to_save = [
                [
                    ele.id if keep_token_index else idx,
                    *ele.coordinates,
                    ele.text,
                    ele.font,
                    ele.type,
                    parent_block_id_for_line_id[ele.parent],  # Cvt to block-level id
                    ele.parent,
                    False,
                    False,
                ]
                for idx, ele in enumerate(self.words, start=len(blocks_to_save))
            ]
            df = pd.DataFrame(
                blocks_to_save + lines_to_save + tokens_to_save,
                columns=[
                    "id",
                    "x_1",
                    "y_1",
                    "x_2",
                    "y_2",
                    "text",
                    "font",
                    "category",
                    "block_id",
                    "line_id",
                    "is_block",
                    "is_line",
                ],
            )

        if normaize_coordinates:
            assert canvas_width is not None
            assert canvas_height is not None
            df[["x_1", "x_2"]] = (df[["x_1", "x_2"]] / canvas_width * 1000).astype(
                "int"
            )
            df[["y_1", "y_2"]] = (df[["y_1", "y_2"]] / canvas_height * 1000).astype(
                "int"
            )

        return df

    def to_dict(
        self,
        keep_token_index=True,
        export_font=False,
        normaize_coordinates=False,
        canvas_width=None,
        canvas_height=None,
        category_map: Dict = None,
    ) -> Dict:

        df = self.to_dataframe(
            keep_token_index=keep_token_index,
            export_font=export_font,
            normaize_coordinates=normaize_coordinates,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

        # Only select text bocks
        df = df[~df.is_block & ~df.is_line]

        # Filter empty text
        df = df.dropna(axis=0, subset=["text"])
        df = df[~df.text.str.isspace()]

        if len(df) == 0:
            return None

        df["block_id"] = df["block_id"].fillna(-1).astype("int")
        df["line_id"] = df["line_id"].fillna(-1).astype("int")

        row_item = {
            "words": df["text"].tolist(),
            "bbox": df.apply(
                lambda row: (row["x_1"], row["y_1"], row["x_2"], row["y_2"]), axis=1
            ).tolist(),
            "block_ids": df["block_id"].tolist(),
            "line_ids": df["line_id"].tolist(),
        }

        if category_map is not None:
            row_item["labels"] = df["category"].map(category_map).tolist()
        else:
            row_item["labels"] = df["category"].tolist()

        if export_font:
            row_item["fonts"] = df["font"].tolist()

        return row_item


def find_closest_token_and_inherit_font(xml_tokens, pdf_tokens):

    xml_token_centers = np.array([ele.block.center for ele in xml_tokens])
    pdf_token_centers = np.array([ele.block.center for ele in pdf_tokens])

    distances = cdist(xml_token_centers, pdf_token_centers, metric="cityblock")
    matching_indices = distances.argmin(axis=1)

    for idx, word in enumerate(xml_tokens):
        target_token = pdf_tokens[matching_indices[idx]]
        if distances[idx, matching_indices[idx]] > DISTANCE_THRESHOLD:
            print("Warning: Extreme large distances and do not assign fonts")
            word.font = ""
        else:
            word.font = target_token.font

    return xml_tokens


def find_closest_token_and_inherit_font(xml_tokens, pdf_tokens):

    xml_token_centers = np.array([ele.block.center for ele in xml_tokens])
    pdf_token_centers = np.array([ele.block.center for ele in pdf_tokens])

    distances = cdist(xml_token_centers, pdf_token_centers, metric="cityblock")
    matching_indices = distances.argmin(axis=1)

    for idx, word in enumerate(xml_tokens):
        target_token = pdf_tokens[matching_indices[idx]]
        if distances[idx, matching_indices[idx]] > DISTANCE_THRESHOLD:
            print("Warning: Extreme large distances and do not assign fonts")
            word.font = ""
        else:
            word.font = target_token.font

    return xml_tokens


class GrotoapDataset:
    def __init__(self, base_dir: str, dataset_folder_name: str = "dataset"):

        self.base_dir = base_dir
        self.dataset_folder_name = dataset_folder_name
        self.all_xml_files = glob(
            f"{self.base_dir}/{self.dataset_folder_name}/*/*.cxml"
        )

    def load_xml(self, xml_filename: str):
        with open(xml_filename, "r") as fp:
            soup = BeautifulSoup(fp, "lxml")

        pages = soup.find_all("page")

        parsed_page_data = {
            idx: self.parse_page_xml(page) for idx, page in enumerate(pages)
        }

        return parsed_page_data

    def parse_page_xml(self, page: "bs4.element.Tag") -> PageData:

        blocks = []
        lines = []
        words = []

        word_id = 0
        line_id = 0
        all_zones = page.find_all("zone")
        if all_zones is None:
            return PageData()

        for zone_id, zone in enumerate(all_zones):

            words_in_this_block = []
            # Fetch the zone
            v1, v2 = zone.find("zonecorners").find_all("vertex")
            block_type = zone.find("classification").find("category")["value"]
            block = lp.TextBlock(
                lp.Rectangle(
                    float(v1["x"]), float(v1["y"]), float(v2["x"]), float(v2["y"])
                ),
                type=block_type,
                id=zone_id,
            )

            # Fetch lines
            all_lines = zone.find_all("line")
            if all_lines is None:
                continue

            for line in all_lines:

                words_in_this_line = []

                v1, v2 = line.find("linecorners").find_all("vertex")
                current_line = lp.TextBlock(
                    lp.Rectangle(
                        float(v1["x"]),
                        float(v1["y"]),
                        float(v2["x"]),
                        float(v2["y"]),
                    ),
                    type=block_type,
                    parent=zone_id,
                    id=line_id,
                )

                # Fetch words
                all_words = line.find_all("word")
                if all_words is None:
                    continue

                for word in line.find_all("word"):
                    v1, v2 = word.find("wordcorners").find_all("vertex")
                    words_in_this_line.append(
                        lp.TextBlock(
                            lp.Rectangle(
                                float(v1["x"]),
                                float(v1["y"]),
                                float(v2["x"]),
                                float(v2["y"]),
                            ),
                            type=block_type,
                            text="".join(
                                [ele["value"] for ele in word.find_all("gt_text")]
                            ),
                            id=word_id,
                            parent=line_id,
                        )
                    )
                    word_id += 1

                current_line.text = " ".join(ele.text for ele in words_in_this_line)
                line_id += 1
                words_in_this_block.extend(words_in_this_line)
                lines.append(current_line)

            block.text = " ".join(ele.text for ele in words_in_this_block)
            blocks.append(block)
            words.extend(words_in_this_block)

        return PageData(blocks, lines, words)

    def convert_xml_to_page_token(self, xml_filename, export_path):

        savename = "-".join(xml_filename.split("/")[-2:]).rstrip(".cxml")
        parsed_page_data = self.load_xml(xml_filename)
        print(f"Processing {savename}")
        for page_id, page_data in parsed_page_data.items():

            if os.path.exists(f"{export_path}/{savename}-{page_id}.csv"):
                continue

            df = page_data.to_dataframe()
            df.to_csv(f"{export_path}/{savename}-{page_id}.csv", index=None)

    def convert_to_page_token_table(self, export_path: str, n_jobs=20):

        if not os.path.exists(export_path):
            os.makedirs(export_path)
            print(f"Creating the export directory {export_path}")
        else:
            print(f"Overwriting existing exports in {export_path}")

        Parallel(n_jobs=n_jobs)(
            delayed(self.convert_xml_to_page_token)(xml_filename, export_path)
            for xml_filename in tqdm(self.all_xml_files)
        )


class GrotoapDatasetWithFont(GrotoapDataset):
    def __init__(self, base_dir: str, dataset_folder_name: str = "dataset"):

        super().__init__(base_dir, dataset_folder_name)

        self.pdf_extractor = PDFPlumberTokenExtractor()

    def convert_xml_to_page_token(self, xml_filename, export_path):

        savename = "-".join(xml_filename.split("/")[-2:]).rstrip(".cxml")

        print(f"Processing {savename}")

        xml_page_data = self.load_xml(xml_filename)

        pdf_filename = xml_filename.replace("cxml", "pdf")

        page_size = []
        try:
            pdf_token_info = self.pdf_extractor(pdf_filename)
            assert len(pdf_token_info) == len(xml_page_data)
        except:
            pdf_token_info = None
            print(f"Serious Issues for loading PDF for {savename}")

        if pdf_token_info is not None:
            for page_id, page_data in xml_page_data.items():
                if len(page_data.words) == 0:
                    continue  # Do not generate if there's no token

                pdf_token_data = pdf_token_info[page_id]
                page_size.append(
                    [
                        savename,
                        page_id,
                        pdf_token_data["height"],
                        pdf_token_data["width"],
                    ]
                )

                if os.path.exists(f"{export_path}/{savename}-{page_id}.csv"):
                    continue

                find_closest_token_and_inherit_font(
                    page_data.words, pdf_token_data["tokens"]
                )

                df = page_data.to_dataframe(
                    export_font=True,
                    normaize_coordinates=True,
                    canvas_height=pdf_token_data["height"],
                    canvas_width=pdf_token_data["width"],
                )

                df.to_csv(f"{export_path}/{savename}-{page_id}.csv", index=None)

        return page_size

    def convert_to_page_token_table(
        self, export_path: str, n_jobs=20, page_size_table="page_sizes.csv"
    ):

        if not os.path.exists(export_path):
            os.makedirs(export_path)
            print(f"Creating the export directory {export_path}")
        else:
            print(f"Overwriting existing exports in {export_path}")

        page_sizes = Parallel(n_jobs=n_jobs)(
            delayed(self.convert_xml_to_page_token)(xml_filename, export_path)
            for xml_filename in tqdm(self.all_xml_files)
        )
        # page_sizes = []
        # for xml_filename in tqdm(self.all_xml_files):
        #     page_sizes.append(self.convert_xml_to_page_token(xml_filename, export_path))

        page_sizes = [
            ele for ele in itertools.chain.from_iterable(page_sizes) if ele is not None
        ]
        df = pd.DataFrame(
            page_sizes, columns=["savename", "page_id", "height", "width"]
        )
        df.to_csv(f"{export_path}/{page_size_table}", index=None)

        return page_sizes


class CERMINELoader(GrotoapDataset):
    def __init__(self):
        pass

    @staticmethod
    def corner_to_rectangle(corners):
        try:
            corners = corners.find_all("vertex")
            corners = np.array([(float(ele["x"]), float(ele["y"])) for ele in corners])
            x1, y1 = corners.min(axis=0)
            x2, y2 = corners.max(axis=0)
            return lp.Rectangle(x1, y1, x2, y2)
        except:
            print("Error for loading corners")
            return None

    def parse_page_xml(self, page: "bs4.element.Tag") -> PageData:

        blocks = []
        lines = []
        words = []

        word_id = 0
        line_id = 0
        all_zones = page.find_all("zone")
        if all_zones is None:
            return PageData()

        for zone_id, zone in enumerate(all_zones):

            words_in_this_block = []
            # Fetch the zone
            rect = self.corner_to_rectangle(zone.find("zonecorners"))
            if rect is None: continue
            block_type = zone.find("classification").find("category")["value"]
            block = lp.TextBlock(
                rect,
                type=block_type,
                id=zone_id,
            )

            # Fetch lines
            all_lines = zone.find_all("line")
            if all_lines is None:
                continue

            for line in all_lines:

                words_in_this_line = []

                rect = self.corner_to_rectangle(line.find("linecorners"))
                if rect is None: continue
                current_line = lp.TextBlock(
                    rect,
                    type=block_type,
                    parent=zone_id,
                    id=line_id,
                )

                # Fetch words
                all_words = line.find_all("word")
                if all_words is None:
                    continue

                for word in line.find_all("word"):
                    rect = self.corner_to_rectangle(word.find("wordcorners"))
                    if rect is None: continue
                    _word = lp.TextBlock(
                        rect,
                        type=block_type,
                        text="".join(
                            [ele["value"] for ele in word.find_all("gt_text")]
                        ),
                        id=word_id,
                        parent=line_id,
                    )
                    _word.font = Counter(
                        [ele["type"] for ele in word.find_all("font")]
                    ).most_common(n=1)[0][
                        0
                    ]  # The most common font in this word
                    words_in_this_line.append(_word)
                    word_id += 1

                current_line.text = " ".join(ele.text for ele in words_in_this_line)
                line_id += 1
                words_in_this_block.extend(words_in_this_line)
                lines.append(current_line)

            block.text = " ".join(ele.text for ele in words_in_this_block)
            blocks.append(block)
            words.extend(words_in_this_block)

        return PageData(blocks, lines, words)