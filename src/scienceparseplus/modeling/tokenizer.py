from typing import List, Union, Dict, Any, Tuple
from itertools import cycle
import os
import re

import torch
from transformers import LayoutLMTokenizer, AutoTokenizer

from ..pdftools import PDFExtractor
from ..constants import *

DEFAULT_FONT_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../data/font-vocab.txt"
)

DEFAULT_SPECIAL_TOKEN_BOXES = {
    "[PAD]": torch.tensor([0, 0, 0, 0], dtype=torch.long),
    "[CLS]": torch.tensor([0, 0, 0, 0], dtype=torch.long),
    "[SEP]": torch.tensor([1000, 1000, 1000, 1000], dtype=torch.long),
}


class FontTokenizer:
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    SPECIAL_TOKENS_MAP = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }

    PREPROCESS_PARTTERN = re.compile(r"(\S{6})\+")

    def __init__(
        self,
        font_vocab_path=DEFAULT_FONT_VOCAB_PATH,
    ):

        with open(font_vocab_path, "r") as fp:
            font_vocabs = fp.read().split("\n")

        font_vocabs = self.SPECIAL_TOKENS + font_vocabs

        self.font2id = {font: idx for idx, font in enumerate(font_vocabs)}
        self.font2id[None] = self.SPECIAL_TOKENS_MAP["[PAD]"]

    def encode(self, font_names):

        return [
            self.font2id.get(
                (
                    re.sub(self.PREPROCESS_PARTTERN, "", font)
                    if font is not None
                    else None
                ),
                self.SPECIAL_TOKENS_MAP["[UNK]"],
            )
            for font in font_names
        ]


class PDFTokenizer:
    """Tokenize all the tokens for a specific page in a PDF document.
    Though not optimized, it could embed all tokens on the given page.
    """

    def __init__(
        self,
        tokenizer_model_path,
        max_seq_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        font_vocab_path=DEFAULT_FONT_VOCAB_PATH,
        label_list=None,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        cls_token_block_id=0,
        sep_token_block_id=0,
        pad_token_block_id=0,
        cls_token_font_id=2,
        sep_token_font_id=4,
        pad_token_font_id=0,
    ):

        self.font_tokenizer = FontTokenizer(font_vocab_path)
        self.tokenizer = LayoutLMTokenizer.from_pretrained(tokenizer_model_path)
        self.max_seq_length = max_seq_length

        self.label_list = label_list
        if label_list is not None:
            self.label_map = {label: i for i, label in enumerate(label_list)}
        else:
            self.label_map = {}

        self.cls_token_at_end = cls_token_at_end
        self.cls_token = cls_token
        self.cls_token_segment_id = cls_token_segment_id
        self.sep_token = sep_token
        self.sep_token_extra = sep_token_extra
        self.pad_on_left = pad_on_left
        self.pad_token = pad_token

        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box

        self.cls_token_block_id = cls_token_block_id
        self.sep_token_block_id = sep_token_block_id
        self.pad_token_block_id = pad_token_block_id

        self.cls_token_font_id = cls_token_font_id
        self.sep_token_font_id = sep_token_font_id
        self.pad_token_font_id = pad_token_font_id

        self.pad_token_segment_id = pad_token_segment_id
        self.pad_token_label_id = pad_token_label_id
        self.cls_token_label_id = pad_token_label_id

        self.sequence_a_segment_id = sequence_a_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero

    def encode_pdf_page(self, pdf_page_token: Dict, use_labels=False):

        height, width = pdf_page_token["height"], pdf_page_token["width"]
        tokens = pdf_page_token["tokens"]

        assert not any(
            [not (b.text or b.text.isspace()) for b in tokens]
        ), "Please ensure all layout elements have non-empty texts"

        texts = [None] * len(tokens)
        bboxes = [None] * len(tokens)
        font_names = [None] * len(tokens)
        block_ids = [None] * len(tokens)
        label_ids = [None] * len(tokens)

        for idx, token in enumerate(tokens):
            texts[idx] = token.text
            bboxes[idx] = token.scale((1000 / width, 1000 / height)).coordinates
            font_names[idx] = getattr(token, "font", None)
            block_ids[idx] = (
                min(token.parent + 1, 32) if token.parent is not None else 0
            )
            label_ids[idx] = self.label_map.get(token.type)

        font_ids = self.font_tokenizer.encode(font_names)

        return self.encode(
            texts, bboxes, font_ids, block_ids, label_ids if use_labels else None
        )

    def encode(self, texts, bboxes, fonts, blocks, labels=None):
        """Encode a page of texts and bounding boxes.

        Args:
            texts (list): A list of **non-empty** words
            bboxes (list): A list of corresponding boxes for words
            labels (list, optional):
                The labels for each token. Used for creating labels
                during training. If not set, then the return data also
                won't include the label_ids.

        Returns:
            dict: with "input_ids", "bbox", "attention_mask", and
                "label_ids" (optional).
        """

        tokens = []
        token_boxes = []
        font_ids = []
        block_ids = []
        label_ids = []
        labels_iter = labels if labels is not None else cycle([labels])

        token_to_orig_index = []
        orig_to_tok_index = []

        for idx, (word, bbox, font_id, block_id, label_id) in enumerate(
            zip(texts, bboxes, fonts, blocks, labels_iter)
        ):
            orig_to_tok_index.append(len(tokens))

            word_tokens = self.tokenizer.tokenize(word)
            num_tokens = len(word_tokens)
            if num_tokens == 0:
                # In newer versions of transformers (3.5.0), tokenize will not
                # automatically substitute unknown tokens with '[UNK]', and we
                # need to manually replace them.
                word_tokens = [self.tokenizer.unk_token]
                num_tokens += 1

            token_boxes.extend([bbox] * num_tokens)
            block_ids.extend([block_id] * num_tokens)
            font_ids.extend([font_id] * num_tokens)
            # Use the real label id for the first token of the word,
            # and padding ids for the remaining tokens
            label_ids.extend(
                [self.label_map.get(label_id, 0)]
                + [self.pad_token_label_id] * (num_tokens - 1)
            )

            for tok in word_tokens:
                token_to_orig_index.append(idx)
                tokens.append(tok)

        assert (
            len(tokens) == len(token_boxes) == len(label_ids) == len(font_ids)
        ), "Incompatible input sizes"

        special_tokens_count = 3 if self.sep_token_extra else 2
        valid_sequence_length = self.max_seq_length - special_tokens_count

        batched_input_ids = []
        batched_token_boxes = []
        batched_font_ids = []
        batched_block_ids = []
        batched_label_ids = []
        batched_input_mask = []
        batched_segment_ids = []

        ptr = 0
        while len(tokens) - ptr > 0:
            cur_tokens = tokens[ptr : ptr + valid_sequence_length]
            curtoken_boxes = token_boxes[ptr : ptr + valid_sequence_length]
            curfont_ids = font_ids[ptr : ptr + valid_sequence_length]
            curblock_ids = block_ids[ptr : ptr + valid_sequence_length]
            curlabel_ids = label_ids[ptr : ptr + valid_sequence_length]

            (
                encoded_input_ids,
                encoded_token_boxes,
                encoded_font_ids,
                encoded_block_ids,
                encoded_label_ids,
                input_mask,
                segment_ids,
            ) = self.process_tokenized_sequence(
                cur_tokens,
                curtoken_boxes,
                curfont_ids,
                curblock_ids,
                curlabel_ids,
            )

            batched_input_ids.append(encoded_input_ids)
            batched_token_boxes.append(encoded_token_boxes)
            batched_font_ids.append(encoded_font_ids)
            batched_block_ids.append(encoded_block_ids)
            batched_label_ids.append(encoded_label_ids)
            batched_input_mask.append(input_mask)
            batched_segment_ids.append(segment_ids)

            ptr += valid_sequence_length

        encoded_input = {
            "input_ids": torch.tensor(batched_input_ids, dtype=torch.long),
            "bbox": torch.tensor(batched_token_boxes, dtype=torch.long),
            "font_ids": torch.tensor(batched_font_ids, dtype=torch.long),
            "block_ids": torch.tensor(batched_block_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batched_input_mask, dtype=torch.long),
            # "segment_ids":    torch.tensor(batched_segment_ids, dtype=torch.long)
        }

        if labels is not None:
            encoded_input["labels"] = torch.tensor(batched_label_ids, dtype=torch.long)

        return encoded_input, token_to_orig_index

    def process_tokenized_sequence(
        self, tokens, token_boxes, font_ids, block_ids, label_ids
    ):

        # Add end of sentence token
        tokens.append(self.sep_token)
        token_boxes.append(self.sep_token_box)
        font_ids.append(self.pad_token_font_id)
        block_ids.append(self.pad_token_block_id)
        label_ids.append(self.pad_token_label_id)

        if self.sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens.append(self.sep_token)
            token_boxes.append(self.sep_token_box)
            font_ids.append(self.sep_token_font_id)
            block_ids.append(self.sep_token_block_id)
            label_ids.append(self.pad_token_label_id)

        segment_ids = [self.sequence_a_segment_id] * len(tokens)

        # Add cls token
        if self.cls_token_at_end:
            tokens.append(self.cls_token)
            token_boxes.append(self.cls_token_box)
            font_ids.append(self.cls_token_font_id)
            block_ids.append(self.cls_token_block_id)
            label_ids.append(self.cls_token_label_id)
            segment_ids.append(self.cls_token_segment_id)
        else:
            tokens = [self.cls_token] + tokens
            token_boxes = [self.cls_token_box] + token_boxes
            font_ids = [self.cls_token_font_id] + font_ids
            block_ids = [self.cls_token_block_id] + block_ids
            label_ids = [self.cls_token_label_id] + label_ids
            segment_ids = [self.cls_token_segment_id] + segment_ids

        # Convert token to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            if self.pad_on_left:
                input_ids = ([self.pad_token] * padding_length) + input_ids
                input_mask = (
                    [0 if self.mask_padding_with_zero else 1] * padding_length
                ) + input_mask
                segment_ids = (
                    [self.pad_token_segment_id] * padding_length
                ) + segment_ids
                label_ids = ([self.pad_token_label_id] * padding_length) + label_ids
                token_boxes = ([self.pad_token_box] * padding_length) + token_boxes
                font_ids = ([self.pad_token_font_id] * padding_length) + font_ids
                block_ids = ([self.pad_token_block_id] * padding_length) + block_ids
            else:
                input_ids += [self.pad_token] * padding_length
                input_mask += [0 if self.mask_padding_with_zero else 1] * padding_length
                segment_ids += [self.pad_token_segment_id] * padding_length
                label_ids += [self.pad_token_label_id] * padding_length
                token_boxes += [self.pad_token_box] * padding_length
                font_ids += [self.pad_token_font_id] * padding_length
                block_ids += [self.pad_token_block_id] * padding_length

        assert len(input_ids) == self.max_seq_length, f"{len(input_ids)}"
        assert len(input_mask) == self.max_seq_length, f"{len(input_mask)}"
        assert len(segment_ids) == self.max_seq_length, f"{len(segment_ids)}"
        assert len(label_ids) == self.max_seq_length, f"{len(label_ids)}"
        assert len(token_boxes) == self.max_seq_length, f"{len(token_boxes)}"
        assert len(block_ids) == self.max_seq_length, f"{len(block_ids)}"
        assert len(font_ids) == self.max_seq_length, f"{len(font_ids)}"

        return (
            input_ids,
            token_boxes,
            font_ids,
            block_ids,
            label_ids,
            input_mask,
            segment_ids,
        )

    def process_logits(
        self, logits: torch.Tensor, token_to_text_map: List
    ) -> List[List]:

        start_idx, end_idx = 1, -1
        predictions = logits.max(dim=-1)
        scores = predictions.values
        indices = predictions.indices
        # Current prediction only supports LayoutLM so this is fine.
        # But we may need ot change that in the future to support other model.

        flatten_scores = scores[:, start_idx:end_idx].reshape(-1)
        flatten_indices = indices[:, start_idx:end_idx].reshape(-1)
        # Remove all the [cls] and [sep] for each sentences and flatten

        filtered_scores = flatten_scores[: len(token_to_text_map)].tolist()
        filtered_indices = flatten_indices[: len(token_to_text_map)].tolist()
        # It's a hack to get all the non-blank texts:
        # Equivalent to:
        # attention_mask[:, -1] = 0
        # attention_mask[1:, 0] = 0
        # pred[attention_mask==1][1:-1]

        # Regroup all the tokens to original words
        prediction_groups = [[] for _ in range(max(token_to_text_map) + 1)]
        for idx, (score, index) in enumerate(zip(filtered_scores, filtered_indices)):
            prediction_groups[token_to_text_map[idx]].append((score, index))

        return prediction_groups


class JSONDatasetTokenizer:
    def __init__(
        self,
        tokenizer_name,
        use_layout=False,
        max_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        special_token_boxes=DEFAULT_SPECIAL_TOKEN_BOXES,
    ):

        if "layout" in tokenizer_name:
            self.tokenizer = LayoutLMTokenizerFast.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.use_layout = use_layout
        self.max_length = max_length

        self.token_to_id_map = {
            token: idx
            for idx, token in zip(
                self.tokenizer.all_special_ids, self.tokenizer.all_special_tokens
            )
        }

        self.special_token_boxes = {
            self.token_to_id_map[token]: box for token, box in special_token_boxes.items()
        }

    def encode_plus(self, features):

        first = features[0]

        all_words = [f["words"] for f in features]
        encoded_inputs = self.tokenizer.batch_encode_plus(
            all_words,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        bz, sequence_length = encoded_inputs["input_ids"].shape
        encoded_bbox = torch.zeros(bz, sequence_length, 4, dtype=torch.long)
        encoded_labels = torch.zeros(bz, sequence_length, dtype=torch.long)

        offset_mapping = encoded_inputs.pop("offset_mapping")
        special_tokens_mask = encoded_inputs.pop("special_tokens_mask")

        for idx, (offset, token_mask, f, input_ids) in enumerate(
            zip(
                offset_mapping,
                special_tokens_mask,
                features,
                encoded_inputs["input_ids"],
            )
        ):
            token_mask = token_mask.bool()
            if False in token_mask:
                mapping = (offset[~token_mask, 0] == 0).cumsum(dim=0) - 1
                encoded_bbox[idx, ~token_mask, :] = torch.tensor(
                    [f["bbox"][idx] for idx in mapping], dtype=torch.long
                )
                encoded_labels[idx, ~token_mask] = torch.tensor(
                    [f["labels"][idx] for idx in mapping], dtype=torch.long
                )

            special_token = 102
            # hard coded for current mode - only SEP token has special bbox

            special_token_idx = torch.where(input_ids == special_token)[0]
            encoded_bbox[idx, special_token_idx, :] = self.special_token_boxes[
                special_token
            ]

        encoded_inputs["labels"] = encoded_labels

        if self.use_layout:
            encoded_inputs["bbox"] = encoded_bbox

        if "class_weight" in first:
            encoded_inputs["class_weight"] = first["class_weight"]

        return encoded_inputs


class JSONDatasetWithFontTokenizer(JSONDatasetTokenizer):
    def __init__(
        self,
        tokenizer_name,
        font_tokenizer,
        use_layout=False,
        max_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        special_token_boxes=DEFAULT_SPECIAL_TOKEN_BOXES,
    ):

        super().__init__(
            tokenizer_name,
            use_layout=use_layout,
            max_length=max_length,
            special_token_boxes=special_token_boxes,
        )

        self.font_tokenizer = font_tokenizer

    def encode_plus(self, features):

        first = features[0]

        all_words = [f["words"] for f in features]

        encoded_inputs = self.tokenizer.batch_encode_plus(
            all_words,
            max_length=512,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        bz, sequence_length = encoded_inputs["input_ids"].shape
        encoded_bbox = torch.zeros(bz, sequence_length, 4, dtype=torch.long)
        encoded_labels = torch.zeros(bz, sequence_length, dtype=torch.long)
        encoded_fonts = torch.zeros(bz, sequence_length, dtype=torch.long)

        offset_mapping = encoded_inputs.pop("offset_mapping")
        special_tokens_mask = encoded_inputs.pop("special_tokens_mask")

        for idx, (offset, token_mask, f, input_ids) in enumerate(
            zip(
                offset_mapping,
                special_tokens_mask,
                features,
                encoded_inputs["input_ids"],
            )
        ):
            token_mask = token_mask.bool()
            if False in token_mask:
                mapping = (offset[~token_mask, 0] == 0).cumsum(dim=0) - 1
                encoded_bbox[idx, ~token_mask, :] = torch.tensor(
                    [f["bbox"][idx] for idx in mapping], dtype=torch.long
                )
                encoded_labels[idx, ~token_mask] = torch.tensor(
                    [f["labels"][idx] for idx in mapping], dtype=torch.long
                )
                encoded_fonts[idx, ~token_mask] = torch.tensor(
                    self.font_tokenizer.encode([f["fonts"][idx] for idx in mapping]),
                    dtype=torch.long,
                )

            special_token = 102
            # hard coded for current mode - only SEP token has special bbox

            special_token_idx = torch.where(input_ids == special_token)[0]
            encoded_bbox[idx, special_token_idx, :] = self.special_token_boxes[
                special_token
            ]

            for special_token, token_idx in self.token_to_id_map.items():
                special_token_idx = torch.where(input_ids == token_idx)[0]
                encoded_fonts[
                    idx, special_token_idx
                ] = self.font_tokenizer.SPECIAL_TOKENS_MAP[special_token]

        encoded_inputs["labels"] = encoded_labels
        encoded_inputs["font_ids"] = encoded_fonts

        # Because there are some negative blocks
        # 0 -> token not in any blocks
        # >=1 -> regular
        if self.use_layout:
            encoded_inputs["bbox"] = encoded_bbox

        return encoded_inputs