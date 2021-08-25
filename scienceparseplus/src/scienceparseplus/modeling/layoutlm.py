from itertools import cycle
import gc

import pandas as pd
import numpy as np
import layoutparser as lp

import torch
import torch
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.models.layoutlm.modeling_layoutlm import *
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMTokenizer,
    LayoutLMForTokenClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from ..constants import IMAGE_FEATURE_SIZE

DEFAULT_SPECIAL_TOKEN_BOXES = {
    "[PAD]": torch.tensor([0, 0, 0, 0], dtype=torch.long),
    "[CLS]": torch.tensor([0, 0, 0, 0], dtype=torch.long),
    "[SEP]": torch.tensor([1000, 1000, 1000, 1000], dtype=torch.long),
}


class LayoutLMPageTokenizer:
    """An old-fashioned tokenizer for LayoutLM for both encoding tokens and coordinates.
    The implementation is based on older version of transformers (2.9) and the original
    LayoutLM code implementations.
    """

    def __init__(
        self,
        tokenizer_model_path,
        max_seq_length,
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
    ):

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
        self.pad_token_segment_id = pad_token_segment_id
        self.pad_token_label_id = pad_token_label_id
        self.sequence_a_segment_id = sequence_a_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero

    def encode(self, texts, bboxes, labels=None):
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
        label_ids = []
        labels_iter = labels if labels is not None else cycle([labels])

        tok_to_orig_index = []
        orig_to_tok_index = []

        for idx, (word, bbox, label_id) in enumerate(zip(texts, bboxes, labels_iter)):
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

            # Use the real label id for the first token of the word,
            # and padding ids for the remaining tokens
            label_ids.extend(
                [self.label_map.get(label_id, 0)]
                + [self.pad_token_label_id] * (num_tokens - 1)
            )

            for tok in word_tokens:
                tok_to_orig_index.append(idx)
                tokens.append(tok)

        assert (
            len(tokens) == len(token_boxes) == len(label_ids)
        ), "Incompatible input sizes"

        special_tokens_count = 3 if self.sep_token_extra else 2
        valid_sequence_length = self.max_seq_length - special_tokens_count

        batched_input_ids = []
        batched_token_boxes = []
        batched_label_ids = []
        batched_input_mask = []
        batched_segment_ids = []

        ptr = 0
        while len(tokens) - ptr > 0:
            cur_tokens = tokens[ptr : ptr + valid_sequence_length]
            curtoken_boxes = token_boxes[ptr : ptr + valid_sequence_length]
            curlabel_ids = label_ids[ptr : ptr + valid_sequence_length]

            (
                encoded_input_ids,
                encoded_token_boxes,
                encoded_label_ids,
                input_mask,
                segment_ids,
            ) = self.process_tokenized_sequence(
                cur_tokens, curtoken_boxes, curlabel_ids
            )

            batched_input_ids.append(encoded_input_ids)
            batched_token_boxes.append(encoded_token_boxes)
            batched_label_ids.append(encoded_label_ids)
            batched_input_mask.append(input_mask)
            batched_segment_ids.append(segment_ids)

            ptr += valid_sequence_length

        encoded_input = {
            "input_ids": torch.tensor(batched_input_ids, dtype=torch.long),
            "bbox": torch.tensor(batched_token_boxes, dtype=torch.long),
            "attention_mask": torch.tensor(batched_input_mask, dtype=torch.long),
            # "segment_ids":    torch.tensor(batched_segment_ids, dtype=torch.long)
        }

        if labels is not None:
            encoded_input["labels"] = torch.tensor(batched_label_ids, dtype=torch.long)

        return encoded_input, tok_to_orig_index

    def process_tokenized_sequence(self, tokens, token_boxes, label_ids):

        # Add end of sentence token
        tokens.append(self.sep_token)
        token_boxes.append(self.sep_token_box)
        label_ids.append(self.pad_token_label_id)

        if self.sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens.append(self.sep_token)
            token_boxes.append(self.sep_token_box)
            label_ids.append(self.pad_token_label_id)

        segment_ids = [self.sequence_a_segment_id] * len(tokens)

        # Add cls token
        if self.cls_token_at_end:
            tokens.append(self.cls_token)
            token_boxes.append(self.cls_token_box)
            label_ids.append(self.pad_token_label_id)
            segment_ids.append(self.cls_token_segment_id)
        else:
            tokens = [self.cls_token] + tokens
            token_boxes = [self.cls_token_box] + token_boxes
            label_ids = [self.pad_token_label_id] + label_ids
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
            else:
                input_ids += [self.pad_token] * padding_length
                input_mask += [0 if self.mask_padding_with_zero else 1] * padding_length
                segment_ids += [self.pad_token_segment_id] * padding_length
                label_ids += [self.pad_token_label_id] * padding_length
                token_boxes += [self.pad_token_box] * padding_length

        assert len(input_ids) == self.max_seq_length, f"{len(input_ids)}"
        assert len(input_mask) == self.max_seq_length, f"{len(input_mask)}"
        assert len(segment_ids) == self.max_seq_length, f"{len(segment_ids)}"
        assert len(label_ids) == self.max_seq_length, f"{len(label_ids)}"
        assert len(token_boxes) == self.max_seq_length, f"{len(token_boxes)}"

        return input_ids, token_boxes, label_ids, input_mask, segment_ids

    def process_predictions(self, predictions, token_to_text_map):

        start_idx, end_idx = 1, -1
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


class LayoutLMEnhancedTokenier:
    """A tokenizer for LayoutLM for both encoding tokens and coordinates.
    The implementation is based on newer version of transformers (>=3.5)
    and is about 5 times faster.

    It also supports flexible tokenization without using layout information.
    """

    def __init__(
        self,
        tokenizer_name,
        use_layout=True,
        use_basic_lm=False,
        special_token_boxes=DEFAULT_SPECIAL_TOKEN_BOXES,
        max_length=512,
    ):

        if use_basic_lm:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            use_layout = False
        else:
            self.tokenizer = LayoutLMTokenizerFast.from_pretrained(tokenizer_name)

        self.use_layout = use_layout
        self.max_length = max_length

        token_to_id_map = {
            token: idx
            for idx, token in zip(
                self.tokenizer.all_special_ids, self.tokenizer.all_special_tokens
            )
        }

        self.special_token_boxes = {
            token_to_id_map[token]: box for token, box in special_token_boxes.items()
        }

    def encode_plus(self, features, encode_label=True):

        first = features[0]

        all_words = [f["words"] for f in features]
        encoded_inputs = self.tokenizer.batch_encode_plus(
            all_words,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_offsets_mapping=self.use_layout,
            return_special_tokens_mask=self.use_layout,
            return_tensors="pt",
        )

        if encode_label:
            dtype = torch.long if isinstance(first["label"], int) else torch.float
            encoded_inputs["labels"] = torch.tensor(
                [f["label"] for f in features], dtype=dtype
            )

        if self.use_layout:
            bz, sequence_length = encoded_inputs["input_ids"].shape
            encoded_bbox = torch.zeros(bz, sequence_length, 4, dtype=torch.long)

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

                special_token = 102
                # hard coded for current mode - only SEP token has a special bbox

                special_token_idx = torch.where(input_ids == special_token)[0]
                encoded_bbox[idx, special_token_idx, :] = self.special_token_boxes[
                    special_token
                ]

            encoded_inputs["bbox"] = encoded_bbox

        return encoded_inputs


# class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
#     """Auxiliary model for sequence classification based on LayoutLM.
#     It does not have any LayoutLM models for sequence classification on
#     the current transformers module (4.0).
#     """

#     config_class = LayoutLMConfig
#     pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST
#     base_model_prefix = "layoutlm"

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.layoutlm = LayoutLMModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         self.init_weights()

#     def get_input_embeddings(self):
#         return self.layoutlm.embeddings.word_embeddings

#     def forward(
#         self,
#         input_ids=None,
#         bbox=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         outputs = self.layoutlm(
#             input_ids=input_ids,
#             bbox=bbox,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class LayoutLMForTokenClassificationWithImageFeature(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size + IMAGE_FEATURE_SIZE, config.num_labels
        )

        self.image_encoder = None

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    @property
    def image_encoder(self):
        return self.__image_encoder

    @image_encoder.setter
    def image_encoder(self, image_encoder):
        self.__image_encoder = image_encoder

    def forward(
        self,
        input_ids=None,
        bbox=None,
        images=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        image_features = torch.stack(
            [
                self.image_encoder(image, bbox[idx].detach().clone())
                for idx, image in enumerate(images)
            ],
            dim=0,
        ).to(self.device)

        logits = self.classifier(
            torch.cat([sequence_output, image_features], dim=-1)
        )

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        del images
        del image_features

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMTokenPredictor:
    """Predict token categories for a page based on LayoutLM."""

    # TODO: Rethink the class names?, maybe add `ForPage`

    def __init__(
        self,
        model_path,
        tokenizer_path="microsoft/layoutlm-large-uncased",
        max_seq_length=512,
        label_map=None,
    ):
        """LayoutLM Token Predictor:
            Predict the layout category for each token+bbox within a document.

        Args:
            model_path (str):
                A folder path which contains the layoutlm model weights and
                tokenizer vocabs.
            tokenizer_path (str):
                The path to vocab folder/files.
                Defaults to "microsoft/layoutlm-large-uncased".
            max_seq_length (int, optional):
                Max sequence length for a input to layoutlm model.
                Defaults to 512.
            label_map (dict, optional):
                The map from the model prediction (ids) to real word labels.
                When not set, it will use the label info stored in
                `model.config.id2label`.
        """
        self.tokenizer = LayoutLMPageTokenizer(
            tokenizer_path, max_seq_length=max_seq_length
        )
        self.model = LayoutLMForTokenClassification.from_pretrained(model_path)
        if label_map != None:
            self.label_map = (
                label_map
                if isinstance(label_map, dict)
                else {idx: label for idx, label in enumerate(label_map)}
            )
        else:
            self.label_map = {
                int(idx): name for idx, name in self.model.config.id2label.items()
            }

    def detect(self, layout, width, height):
        """Predict the layout category for all tokens in layout.

        Args:
            layout (layoutparser.Layout):
                A Layout object contains word texts and coordinates.

            width (int):
                The width of the page.

            height (int):
                The height of the page.

        Returns:
            layoutparser.Layout:
                The same layout object with predicted types
                for each element.
        """

        assert not any(
            [not (b.text or b.text.isspace()) for b in layout]
        ), "Please ensure all layout elements have non-empty texts"

        texts = [str(b.text) for b in layout]
        bboxes = [b.scale((1000 / width, 1000 / height)).coordinates for b in layout]

        model_inputs, token_to_text_map = self.tokenizer.encode(texts, bboxes)
        predictions = self.model(**model_inputs)[0].max(-1)
        # Model returns a tuple, the first of which is the category prob.

        prediction_groups = self.tokenizer.process_predictions(
            predictions, token_to_text_map
        )

        texts_predictions = [ele[0] for ele in prediction_groups]
        # Select the first element as the prediction category.

        final_layout = lp.Layout()
        for block, (score, label) in zip(layout, texts_predictions):
            label = self.label_map.get(label, label)
            final_layout.append(block.set(type=label, score=score))

        return final_layout


class LayoutLMBlockPredictor:
    """Predict token categories for a block based on LayoutLM."""

    def __init__(
        self,
        model_path,
        tokenizer_path,
        max_seq_length=512,
        label_map=None,
        use_basic_lm=False,
    ):
        """LayoutLM Token Predictor:
            Predict the layout category for each token+bbox within a document.

        Args:
            model_path (str):
                A folder path which contains the layoutlm model weights and
                tokenizer vocabs.
            tokenizer_path (str):
                The path to vocab folder/files.
                Defaults to "microsoft/layoutlm-large-uncased".
            max_seq_length (int, optional):
                Max sequence length for a input to layoutlm model.
                Defaults to 512.
            label_map (dict, optional):
                The map from the model prediction (ids) to real word labels.
                When not set, it will use the label info stored in
                `model.config.id2label`.
        """
        self.tokenizer = LayoutLMEnhancedTokenier(
            tokenizer_path,
            max_length=max_seq_length,
            use_layout=True,
            use_basic_lm=use_basic_lm,
        )
        if use_basic_lm:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = LayoutLMForSequenceClassification.from_pretrained(model_path)

        if label_map != None:
            self.label_map = (
                label_map
                if isinstance(label_map, dict)
                else {idx: label for idx, label in enumerate(label_map)}
            )
        else:
            self.label_map = {
                int(idx): name for idx, name in self.model.config.id2label.items()
            }

    def encode_input(self, tokens, width, height):
        """Encode the inputs as model inputs

        Args:
            tokens (layoutparser.Layout):
                A Layout object contains word texts and coordinates.

            width (int):
                The width of the page.

            height (int):
                The height of the page.

        Returns:
            dict:
                A dictionary with words, bboxes, and label.
        """

        assert not any(
            [not (b.text or b.text.isspace()) for b in tokens]
        ), "Please ensure all layout elements have non-empty texts"

        words = [str(b.text) for b in tokens]
        bbox = [b.scale((1000 / width, 1000 / height)).coordinates for b in tokens]

        encoded_inputs = dict(words=words, bbox=bbox)
        return encoded_inputs

    def detect(self, tokens, width, height):
        """Predict the block category given all tokens in the block.

        Args:
            tokens (layoutparser.Layout):
                A Layout object contains word texts and coordinates.

            width (int):
                The width of the page.

            height (int):
                The height of the page.

        Returns:
            layoutparser.Layout:
                The same layout object with predicted types
                for each element. With the pred_type stored in `type`
        """

        encoded_inputs = self.encode_input(tokens, width, height)

        model_inputs = self.tokenizer.encode_plus([encoded_inputs], encode_label=False)

        model_outputs = self.model(**model_inputs).logits.max(dim=1)
        confidence = model_outputs.values.item()
        prediction = model_outputs.indices.item()
        label = self.label_map.get(prediction, prediction)

        final_layout = lp.Layout(
            [block.set(type=label, score=confidence) for block in tokens]
        )

        return final_layout