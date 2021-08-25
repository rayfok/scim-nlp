# coding=utf-8
# Modified based on http://github.com/huggingface/transformers/blob/master/src/transformers/models/layoutlm/modeling_layoutlm.py
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LayoutLM model. """

from typing import List, Union, Dict, Any, Tuple
import math
from itertools import cycle

import layoutparser as lp
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.layoutlm.configuration_layoutlm import LayoutLMConfig

from transformers import LayoutLMTokenizer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LayoutLMConfig"
_TOKENIZER_FOR_DOC = "LayoutLMTokenizer"

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlm-base-uncased",
    "layoutlm-large-uncased",
]

from ..constants import MAX_FONT_NUMBER

LayoutLMLayerNorm = torch.nn.LayerNorm


class LayoutLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(LayoutLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.font_embeddings = nn.Embedding(MAX_FONT_NUMBER, config.hidden_size)

        self.LayerNorm = LayoutLMLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        font_ids=None,
        inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if font_ids is None:
            font_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        words_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)

        try:
            font_embeddings = self.font_embeddings(font_ids)
        except IndexError as e:
            raise IndexError(
                f"The :obj:`font_ids` values should be within 0-{MAX_FONT_NUMBER} range."
            ) from e

        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError(
                "The :obj:`bbox`coordinate values should be within 0-1000 range."
            ) from e

        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            words_embeddings
            # + position_embeddings
            + font_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->LayoutLM
class LayoutLMSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in LayoutLMModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->LayoutLM
class LayoutLMSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->LayoutLM
class LayoutLMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMSelfAttention(config)
        self.output = LayoutLMSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class LayoutLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->LayoutLM
class LayoutLMOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->LayoutLM
class LayoutLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = LayoutLMAttention(config)
        self.intermediate = LayoutLMIntermediate(config)
        self.output = LayoutLMOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->LayoutLM
class LayoutLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [LayoutLMLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class LayoutLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->LayoutLM
class LayoutLMPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->LayoutLM
class LayoutLMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LayoutLMPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->LayoutLM
class LayoutLMOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LayoutLMLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LayoutLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlm"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayoutLMLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


LAYOUTLM_START_DOCSTRING = r"""
    The LayoutLM model was proposed in `LayoutLM: Pre-training of Text and Layout for Document Image Understanding
    <https://arxiv.org/abs/1912.13318>`__ by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and Ming Zhou.

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.LayoutLMConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

LAYOUTLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.LayoutLMTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        bbox (:obj:`torch.LongTensor` of shape :obj:`({0}, 4)`, `optional`):
            Bounding boxes of each input sequence tokens. Selected in the range ``[0,
            config.max_2d_position_embeddings-1]``. Each bounding box should be a normalized version in (x0, y0, x1,
            y1) format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and
            (x1, y1) represents the position of the lower right corner. See :ref:`Overview` for normalization.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``: ``0`` corresponds to a `sentence A` token, ``1`` corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``: :obj:`1`
            indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under
            returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned
            tensors for more detail.
        return_dict (:obj:`bool`, `optional`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
            plain tuple.
"""


@add_start_docstrings(
    "The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMModel(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super(LayoutLMModel, self).__init__(config)
        self.config = config

        self.embeddings = LayoutLMEmbeddings(config)
        self.encoder = LayoutLMEncoder(config)
        self.pooler = LayoutLMPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        bbox=None,
        font_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMModel
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMModel.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if bbox is None:
            bbox = torch.zeros(
                tuple(list(input_shape) + [4]), dtype=torch.long, device=device
            )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            font_ids=font_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """LayoutLM Model with a `language modeling` head on top. """,
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMForMaskedLM(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.layoutlm = LayoutLMModel(config)
        self.cls = LayoutLMOnlyMLMHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        bbox=None,
        font_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMForMaskedLM
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForMaskedLM.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "[MASK]"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])

            >>> labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=labels)

            >>> loss = outputs.loss
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.layoutlm(
            input_ids,
            bbox,
            font_ids=font_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
    document image classification tasks such as the `RVL-CDIP <https://www.cs.cmu.edu/~aharley/rvl-cdip/>`__ dataset.
    """,
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids=None,
        bbox=None,
        font_ids=None,
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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForSequenceClassification.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> sequence_label = torch.tensor([1])

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=sequence_label)

            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            font_ids=font_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    sequence labeling (information extraction) tasks such as the `FUNSD <https://guillaumejaume.github.io/FUNSD/>`__
    dataset and the `SROIE <https://rrc.cvc.uab.es/?ch=13>`__ dataset.
    """,
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMForTokenClassification(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids=None,
        bbox=None,
        font_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_weight=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> token_labels = torch.tensor([1,1,0,0]).unsqueeze(0) # batch size of 1

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=token_labels)

            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            font_ids=font_ids,
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
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weight)

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

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
        cls_token_font_id=2,
        sep_token_font_id=4,
        pad_token_font_id=0,
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

        self.cls_token_font_id = cls_token_font_id
        self.sep_token_font_id = sep_token_font_id
        self.pad_token_font_id = pad_token_font_id

        self.pad_token_segment_id = pad_token_segment_id
        self.pad_token_label_id = pad_token_label_id
        self.cls_token_label_id = pad_token_label_id
        self.sequence_a_segment_id = sequence_a_segment_id
        self.mask_padding_with_zero = mask_padding_with_zero

    def encode(self, texts, bboxes, blocks, labels=None):
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
        label_ids = []
        labels_iter = labels if labels is not None else cycle([labels])

        tok_to_orig_index = []
        orig_to_tok_index = []

        for idx, (word, bbox, font_id, label_id) in enumerate(
            zip(texts, bboxes, blocks, labels_iter)
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
            font_ids.extend([font_id] * num_tokens)
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
            len(tokens) == len(token_boxes) == len(label_ids) == len(font_ids)
        ), "Incompatible input sizes"

        special_tokens_count = 3 if self.sep_token_extra else 2
        valid_sequence_length = self.max_seq_length - special_tokens_count

        batched_input_ids = []
        batched_token_boxes = []
        batched_font_ids = []
        batched_label_ids = []
        batched_input_mask = []
        batched_segment_ids = []

        ptr = 0
        while len(tokens) - ptr > 0:
            cur_tokens = tokens[ptr : ptr + valid_sequence_length]
            curtoken_boxes = token_boxes[ptr : ptr + valid_sequence_length]
            curfont_ids = font_ids[ptr : ptr + valid_sequence_length]
            curlabel_ids = label_ids[ptr : ptr + valid_sequence_length]

            (
                encoded_input_ids,
                encoded_token_boxes,
                encoded_font_ids,
                encoded_label_ids,
                input_mask,
                segment_ids,
            ) = self.process_tokenized_sequence(
                cur_tokens,
                curtoken_boxes,
                curfont_ids,
                curlabel_ids,
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

    def process_tokenized_sequence(self, tokens, token_boxes, font_ids, label_ids):

        # Add end of sentence token
        tokens.append(self.sep_token)
        token_boxes.append(self.sep_token_box)
        font_ids.append(self.pad_token_font_id)
        label_ids.append(self.pad_token_label_id)

        if self.sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens.append(self.sep_token)
            token_boxes.append(self.sep_token_box)
            font_ids.append(self.sep_token_font_id)
            label_ids.append(self.pad_token_label_id)

        segment_ids = [self.sequence_a_segment_id] * len(tokens)

        # Add cls token
        if self.cls_token_at_end:
            tokens.append(self.cls_token)
            token_boxes.append(self.cls_token_box)
            font_ids.append(self.cls_token_font_id)
            label_ids.append(self.cls_token_label_id)
            segment_ids.append(self.cls_token_segment_id)
        else:
            tokens = [self.cls_token] + tokens
            token_boxes = [self.cls_token_box] + token_boxes
            font_ids = [self.cls_token_font_id] + font_ids
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
            else:
                input_ids += [self.pad_token] * padding_length
                input_mask += [0 if self.mask_padding_with_zero else 1] * padding_length
                segment_ids += [self.pad_token_segment_id] * padding_length
                label_ids += [self.pad_token_label_id] * padding_length
                token_boxes += [self.pad_token_box] * padding_length
                font_ids += [self.pad_token_font_id] * padding_length

        assert len(input_ids) == self.max_seq_length, f"{len(input_ids)}"
        assert len(input_mask) == self.max_seq_length, f"{len(input_mask)}"
        assert len(segment_ids) == self.max_seq_length, f"{len(segment_ids)}"
        assert len(label_ids) == self.max_seq_length, f"{len(label_ids)}"
        assert len(token_boxes) == self.max_seq_length, f"{len(token_boxes)}"
        assert len(font_ids) == self.max_seq_length, f"{len(font_ids)}"

        return input_ids, token_boxes, font_ids, label_ids, input_mask, segment_ids

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
        font_ids = [
            min(b.parent + 1, 32) if b.parent is not None else 0 for b in layout
        ]

        model_inputs, token_to_text_map = self.tokenizer.encode(
            texts, bboxes, font_ids
        )
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