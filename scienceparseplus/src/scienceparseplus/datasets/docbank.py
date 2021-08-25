from typing import List, Union, Dict, Any, Tuple, Optional
import os
import random
import json
import itertools
from collections import Counter
import cv2

import torch
import pandas as pd
import layoutparser as lp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image


DEFAULT_COLOR_MAP = {
    None: "#C0C0C0",
    "abstract": "#ffb6c1",
    "author": "#02028b",
    "caption": "#00ff03",
    "date": "#0f0000",
    "equation": "#ff0000",
    "figure": "#ff01ff",
    "footer": "#C2C2C2",
    "list": "#302070",
    "paragraph": "#b0c4de",
    "reference": "#309000",
    "section": "#0603ff",
    "table": "#01ffff",
    "title": "#00bfff",
}

DEFAULT_LABEL_MAP = {
    "paragraph": 0,
    "title": 1,
    "equation": 2,
    "reference": 3,
    "section": 4,
    "list": 5,
    "table": 6,
    "caption": 7,
    "author": 8,
    "abstract": 9,  # 522
    "footer": 10,  # 10
    "date": 11,  # 4
    "figure": 12,
}


def _load_json(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def _write_json(data, filename):

    with open(filename, "w") as fp:
        json.dump(data, fp)


class DocBankDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        image_folder_name="DocBank_500K_ori_img",
        text_folder_name="DocBank_500K_txt",
        annotation_loading_format="layout",
        cleanup_annotations=True,
        load_image=True,
    ):
        """A dataloader for the original DocBank Dataset.

        The directory structure is shown as follows:

            `base_path`/
            ├───`image_folder_name`/
            │     └───`name`_ori.jpg
            └───`text_folder_name`/
                └───`name`.txt

        Args:
            path (str): The root dir of the docbank dataset
            image_folder_name (str, optional):
                The folder_name for saving all images in the docbank dataset,
                Defaults to "DocBank_500K_ori_img".
            text_folder_name (str, optional):
                The folder_name for saving all token+category annotations
                Defaults to "DocBank_500K_txt".
            annotation_loading_format (str, optional):
                Load annotation as "layout" or "dataframe"
                Defaults to "layout".
            cleanup_annotations (bool, optional):
                Whether to cleanup the annotation files, i.e., dropping
                "##LTLine##" and "##LTFigure##" from the annotation file
                Defaults to True.
            load_image (bool, optional):
                Whether to load images.
                Defaults to False.
        """
        self.base_path = base_path

        self.image_path = self.base_path + "/" + image_folder_name
        self.text_path = self.base_path + "/" + text_folder_name

        self.all_index = [ele.rstrip(".txt") for ele in os.listdir(self.text_path)]

        self._empty_df = pd.DataFrame(
            columns=["text", "x_1", "y_1", "x_2", "y_2", "R", "G", "B", "font", "type"]
        )

        self.export_layout = annotation_loading_format == "layout"
        self.cleanup_annotations = cleanup_annotations
        self.load_image = load_image

    def load_annotations(
        self, filename: str, cleanup=True, export_layout=True
    ) -> Union[pd.DataFrame, lp.Layout]:
        """Load data annotations as a dataframe

        Args:
            filename (str): the abspath to the txt file
            cleanup (bool, optional):
                Whether to cleanup the annotation files, i.e., dropping
                "##LTLine##" and "##LTFigure##" from the annotation file
                Defaults to True.
            export_layout (bool, optional):
                Whether to convert the output format as lp.Layout format.
                Defaults to True.

        Returns:
            Union[pd.DataFrame, lp.Layout]:
                When export_layout=true, return a lp.Layout including all tokens
                Otherwise return a DataFrame.
                Note: the file could be empty.
        """
        if os.stat(filename).st_size == 0:
            # Check empty file
            if export_layout:
                return lp.Layout([])
            else:
                return self._empty_df

        df = pd.read_csv(filename, sep="\t", header=None, encoding="utf-8", quoting=3)
        df.columns = ["text", "x_1", "y_1", "x_2", "y_2", "R", "G", "B", "font", "type"]
        df["text"] = df["text"].astype("str")
        df = df.reset_index()

        if cleanup:
            # Drop all ltline and ltfigure tokens
            df = df[(df["text"] != "##LTLine##") & (df["text"] != "##LTFigure##")]

        if export_layout:

            def convert_row_to_rectbox(row):
                rectbox = lp.TextBlock(
                    lp.Rectangle(row["x_1"], row["y_1"], row["x_2"], row["y_2"]),
                    text=row["text"],
                    type=row["type"],
                    id=row["index"],
                )
                rectbox.font = row["font"]
                return rectbox

            layout = df.apply(convert_row_to_rectbox, axis=1).tolist()
            return lp.Layout(layout)
        else:
            return df

    def get_text_anno_path(self, name: str) -> str:
        return os.path.join(self.text_path, name + ".txt")

    def get_image_path(self, name: str) -> str:
        return os.path.join(self.image_path, name + "_ori.jpg")

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, Union[pd.DataFrame, lp.Layout], Optional["Image"]]:
        """Return the name, layout, and image for an item in the dataset.

        Returns:
            Tuple[str, Union[pd.DataFrame, lp.Layout], Optional[Image]]:
                the filename of the given item,
                the token info and category annotation
                the image of the file (when self.load_image==True)
        """

        name = self.all_index[idx]

        text_anno_name = self.get_text_anno_path(name)
        text_anno = self.load_annotations(
            text_anno_name, self.cleanup_annotations, self.export_layout
        )

        if not self.load_image:
            return name, text_anno

        image_anno_name = self.get_image_path(name)
        image = Image.open(image_anno_name)
        w, h = image.size
        text_anno = text_anno.scale((w / 1000, h / 1000))
        # scale the text annotaiton to image size
        return name, text_anno, image


class DocBankBlockClassificationDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        subset: str,
        filename: str = None,
        select_n=None,
        select_ratio=None,
        encode_first=False,
    ):
        """A dataloader for the DocBank Block Classification Dataset.

        The directory structure is shown as follows:

            `base_path`
            ├───dev.json
            ├───train.json
            └───test.json

        This dataset is used for training the block text classification model
        in the pipeline method. It is generated by using the blocks predicted by
        visual layout detection models. These models generate block level bounding
        boxes, and we use these blocks for grouping tokens from the docbank dataset.
        It is generated by the xxx.py script, and all the token data is stored in the
        JSON files for better IO performance.

        The JSON contains the following fields:

            {
                "data": a list of block text data, specified below,
                "labels": a dict used (label_id, label_name) for the data,
                "problematic_items": optional, see below
            }

        As for each data item, it is saved as:
            {
                "words": a list of words in the text block,
                "bbox": a list of block bounding boxes for all words,
                "labels": the label_id for this block.
            }


        Note: Sometimes we might have some tokenization bugs from the dataset. To avoid it from
        disrupting the training process, we can identify these item indices before training and
        excluding them from being loaded during training. These indices are stored in the JSON
        as well, under the field "problematic_items".

        Args:
            base_path (str):
                The basepath of the docbank dataset folder.
            subset (str):
                The name of the used subset, in "train", "dev", or "test".
            select_n (int, optional):
                The number of instances will be used during training.
                Defaults to None.
            select_ratio (float, optional):
                The fraction of dataset will be used during training.
                Defaults to None.
            filename (str, optional):
                By default, the loading filename will be the same as the `base_path`/`subset`.json.
                But you could set it specifically to override the default filename: `base_path`/`filename`.json
                Defaults to None.
            encode_first (bool, optional):
                Whether to encode the dataset ahead.
                Defaults to False.
        """

        # TODO: Update the filename and link in the docstring

        self.base_path = base_path

        self.filename = f"{base_path}/{subset}.json"
        if filename is not None:
            self.filename = f"{base_path}/{filename}"
        print(f"Loading from {self.filename}")
        raw_data = _load_json(self.filename)

        _data = raw_data["data"]
        self.labels = raw_data["labels"]

        self._data = []
        for ele in _data:
            if ele != {} and len(ele["words"]) == len(ele["bbox"]):
                ele["words"] = [str(word) for word in ele["words"]]
                self._data.append(ele)

        self._all_indices = list(range(len(self._data)))

        error_idx = self.index_or_load_problematic_items(raw_data)

        print(f"Dropping problematic items {error_idx}")
        for ele in sorted(error_idx, reverse=True):
            # Remove the problematic indices
            # Start from the last to avoid indices shift in the loop
            del self._all_indices[ele]

        if select_n is not None:
            self._all_indices = random.sample(self._all_indices, select_n)
        elif select_ratio is not None:
            self._all_indices = random.sample(
                self._all_indices, int(len(self._all_indices) * select_ratio)
            )

        self.encode_first = encode_first
        self._encoded_data = None
        del raw_data

    def __getitem__(self, idx):
        if not self.encode_first:
            return self._data[self._all_indices[idx]]
        else:
            if self._encoded_data is None:
                raise ValueError("Please run self.encode_data(tokenizer) first")
            return self._encoded_data[idx]

    def __len__(self):
        return len(self._all_indices)

    def index_or_load_problematic_items(self, raw_data: Dict) -> List[int]:
        if "problematic_items" not in raw_data:
            print("problematic_items are not loaded.")
            error_idx = []
        else:
            print("Loading problematic items from file")
            error_idx = raw_data.get("problematic_items", [])
        return error_idx

    def encode_data(self, tokenizer):

        self._encoded_data = []

        for idx in tqdm(self._all_indices):
            self._encoded_data.append(tokenizer.encode_plus([self._data[idx]]))


class DocBankBlockEmbeddingDataset(DocBankBlockClassificationDataset):
    """"""

    LONG_PASSAGE_THRESHOLD = 752
    MAX_SEQ_LEN = 512
    MAX_BLOCK_EMBEDDING_NUMBER = 32

    def __init__(
        self,
        base_path: str,
        subset: str,
        filename: str = None,
        select_n=None,
        select_ratio=None,
        encode_first=False,
        add_class_weight=False,
    ):
        """A dataloader for the DocBank Block Embedding Dataset.

        This dataset is used for training the block embedding LayoutLM model
        It is generated similar to `DocBankBlockClassificationDataset`.

        Different from DocBankBlockClassificationDataset, for each data item,
        it stores all text for a page, and also includes a new field call block_ids:
            {
                "words": a list of words for the whole page,
                "bbox": a list of block bounding boxes for all words,
                "labels": a list of label_ids for all tokens,
                "block_ids": the block ids for each token on this page
            }

        Args:
            base_path (str):
                The basepath of the docbank dataset folder
            subset (str):
                The name of the used subset, in "train", "dev", or "test".
            select_n (int, optional):
                The number of instances will be used during training.
                Defaults to None.
            select_ratio (float, optional):
                The fraction of dataset will be used during training.
                Defaults to None.
            filename (str, optional):
                By default, the loading filename will be the same as the `base_path`/`subset`.json.
                But you could set it specifically to override the default filename: `base_path`/`filename`.json
                Defaults to None.
            encode_first (bool, optional):
                Whether to encode the dataset ahead.
                Defaults to False.
            add_class_weight (bool, optional):
                Whether to encode the dataset ahead.
                Defaults to False.
        """
        self.base_path = base_path

        self.filename = f"{base_path}/{subset}.json"
        if filename is not None:
            self.filename = f"{base_path}/{filename}"
        print(f"Loading from {self.filename}")
        raw_data = _load_json(self.filename)

        self.labels = raw_data["labels"]
        self.files = raw_data.get('files')

        self._data = raw_data["data"]
        self._all_indices = list(range(len(self._data)))

        error_idx = self.index_or_load_problematic_items(raw_data)

        print(f"Dropping problematic items {error_idx}")
        for ele in sorted(error_idx, reverse=True):
            # Remove the problematic indices
            # Start from the last to avoid indices shift in the loop
            del self._all_indices[ele]

        # NEW IN THIS CLASS
        print("Dropping pages of many blocks")
        self._all_indices = [
            ele
            for ele in self._all_indices
            if max(self._data[ele]["block_ids"]) + 1 < self.MAX_BLOCK_EMBEDDING_NUMBER
        ]
        # Because 0 is reserved for "tokens not in any blocks"

        if select_n is not None:
            self._all_indices = random.sample(self._all_indices, select_n)
        elif select_ratio is not None:
            self._all_indices = random.sample(
                self._all_indices, int(len(self._all_indices) * select_ratio)
            )

        self.encode_first = encode_first
        self._encoded_data = None

        self.add_class_weight = add_class_weight
        if self.add_class_weight:
            results = list(
                itertools.chain.from_iterable(
                    [self._data[idx]["labels"] for idx in self._all_indices]
                )
            )
            cnts = Counter(results)
            freq = torch.Tensor([cnts[i] for i in range(len(self.labels))])
            self.class_weight = -torch.log(freq / freq.sum())

            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                self.class_weight = self.class_weight.unsqueeze(0).repeat(n_gpus, 1)

        del raw_data

    def __getitem__(self, idx):

        item = self._data[self._all_indices[idx]]
        word_count = len(item["words"])

        # For longer articles, BERT will only select the first 512 tokens.
        # To expose the model with the tailing text in these longer passages,
        # we randomly sample the starting point.

        if word_count > self.LONG_PASSAGE_THRESHOLD:
            start = random.choice([0, word_count - self.MAX_SEQ_LEN])
            item = {key: val[start:word_count] for key, val in item.items()}

        if self.add_class_weight:
            item["class_weight"] = self.class_weight
        return item

class DocBankImageFeatureDataset(DocBankBlockEmbeddingDataset):

    def __init__(
        self,
        base_path: str,
        subset: str,
        image_directory=str,
        filename: str = None,
        select_n=None,
        select_ratio=None,
        encode_first=False,
        add_class_weight=False,
    ):

        super().__init__(
            base_path = base_path,
            subset = subset,
            filename = filename,
            select_n = select_n,
            select_ratio = select_ratio,
            encode_first = encode_first,
            add_class_weight = add_class_weight,
        )

        self.image_directory = image_directory

    def __getitem__(self, idx):

        item = super().__getitem__(idx)
        
        image_filename = self.files[self._all_indices[idx]].replace('.txt', '_ori.jpg')
        image = cv2.imread(f"{self.image_directory}/{image_filename}")

        item['image'] = image
        return item