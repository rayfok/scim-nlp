from typing import List, Union, Dict, Any, Tuple
import random

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .utils import load_json, write_json

def index_problematic_items_in_dataset(filename: str, tokenizer):
    _raw_data = load_json(filename)
    raw_data = _raw_data['data']

    error_input_index = []
    for idx, item in enumerate(raw_data):
        if (
            (item == {})
            or (len(item["words"]) != len(item["bbox"]))
            or any(type(e) != str for e in item["words"])
        ):
            error_input_index.append(idx)

    error_tokenization_idx = []
    for idx, item in enumerate(tqdm(raw_data)):
        if idx in error_tokenization_idx:
            continue
        try:
            tokenizer.encode_plus([item])
        except Exception as e:
            print(f"{idx}", e)
            error_tokenization_idx.append(idx)

    error_idx = list(set(error_input_index).union(set(error_tokenization_idx)))
    print(f"Saving problematic items to file {filename}")
    _raw_data["problematic_items"] = error_idx
    write_json(_raw_data, filename)


class JSONDataset(Dataset):
    """
    A generalized dataloader for JSON formatted dataset.

    The directory structure is shown as follows:

        `base_path`
            ├───dev.json
            ├───train.json
            └───test.json
            
    The JSON contains the following fields:

        {
            "data": a list of block text data, specified below,
            "labels": a dict used (label_id, label_name) for the data,
            "problematic_items": optional, see below
            "files": a list of string for file names for each data item
        }

    As for each data item, it is saved as:
        {
            "words": a list of words in the text block,
            "bbox": a list of block bounding boxes for all words,
            "labels": the label_id for this block,
            ....
        }

    Note: Sometimes we might have some tokenization bugs from the dataset. To avoid it from
    disrupting the training process, we can identify these item indices before training and
    excluding them from being loaded during training. These indices are stored in the JSON
    as well, under the field "problematic_items", and can be extracted using index_problematic_items_in_dataset. 
    """

    def __init__(
        self,
        base_path: str,
        subset: str,
        filename: str = None,
        select_n=None,
        select_ratio=None,
        encode_first=False,
    ):
        """
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
        self.base_path = base_path

        if filename is not None:
            self.filename = f"{base_path}/{filename}"
        else:
            self.filename = f"{base_path}/{subset}.json"

        print(f"Loading from {self.filename}")
        raw_data = load_json(self.filename)

        self._data = raw_data["data"]
        self.labels = raw_data["labels"]
        self._all_indices = self.get_valid_indices(raw_data)

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

    def get_problematic_items(self, raw_data: Dict) -> List[int]:
        if "problematic_items" not in raw_data:
            print("problematic_items are not loaded.")
            error_idx = []
        else:
            print("Loading problematic items from file")
            error_idx = raw_data.get("problematic_items", [])
        return error_idx

    def get_valid_indices(self, raw_data:Dict) -> List[int]:
        """Currently it only removes problematic items from the index,
        but can be rewritten to support more comprehensive indeices 
        prefiltering functions.
        """
        _all_indices = list(range(len(self._data)))

        error_idx = self.get_problematic_items(raw_data)
        print(f"Dropping problematic items {error_idx}")
        for ele in sorted(error_idx, reverse=True):
            # Remove the problematic indices
            # Start from the last to avoid indices shift in the loop
            del _all_indices[ele]

        return _all_indices


class JSONTokenClassficationDataset(JSONDataset):
    
    LONG_PASSAGE_THRESHOLD = 752
    MAX_SEQ_LEN = 512

    def __getitem__(self, idx):

        item = self._data[self._all_indices[idx]]
        word_count = len(item["words"])

        # For longer articles, BERT will only select the first 512 tokens.
        # To expose the model with the tailing text in these longer passages,
        # we randomly sample the starting point.

        if word_count > self.LONG_PASSAGE_THRESHOLD:
            start = random.choice([0, self.LONG_PASSAGE_THRESHOLD - self.MAX_SEQ_LEN])
            item = {key: val[start:word_count] for key, val in item.items()}

        return item