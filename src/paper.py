import difflib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cmp_to_key
from string import punctuation
from typing import Dict, List

import pysbd

from bbox import (
    Block,
    BoundingBox,
    SentenceWithBBoxes,
    _are_same_row,
    _bbox_to_json,
    _union_bboxes,
)

SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=True)


@dataclass
class RhetoricUnit:
    text: str
    label: str
    bboxes: List[BoundingBox]
    section: str
    prob: float
    is_author_statement: bool
    is_in_expected_section: bool

    def to_json(self) -> Dict:
        out = self.__dict__
        out["bboxes"] = [_bbox_to_json(b) for b in self.bboxes]
        return out

    def __repr__(self):
        self.to_json()


@dataclass
class MediaUnit:
    type: str
    text: str
    bbox: BoundingBox

    def to_json(self) -> Dict:
        out = self.__dict__
        out["bbox"] = _bbox_to_json(self.bbox)
        return out

    def __repr__(self):
        self.to_json()


@dataclass
class SectionUnit:
    text: str
    bbox: BoundingBox

    def to_json(self) -> Dict:
        out = self.__dict__
        out["bbox"] = _bbox_to_json(self.bbox)
        return out

    def __repr__(self):
        self.to_json()


class S2OrcPaper:
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        self.id = data["metadata"]["paper_id"]
        self.authors = data["metadata"]["authors"]
        self.title = data["metadata"]["title"]
        self.mag_field_of_study = data["metadata"]["mag_field_of_study"]
        self.body = data["pdf_parse"]["body_text"]
        self.abstract = data["metadata"]["abstract"]
        self.sent_sect_map = self._make_sent_sect_map()
        self.sentences = self.sent_sect_map.keys()
        self.sections = list(set(self.sent_sect.values()))

    def _make_sent_sect_map(self):
        sent_sect_map = {}
        for chunk in self.body:
            section = chunk["section"]
            chunk_sents = [s.sent for s in SEGMENTER.segment(chunk["text"])]
            for sent in chunk_sents:
                sent_sect_map[sent] = section
        return sent_sect_map

    def get_section_for_sentence(self, sentence):
        return self.sent_sect_map[sentence]


class SPPPaper:
    def __init__(self, json_file_path: str, engine: str = "detect"):
        spp_engines = ["cermine", "detect"]
        if engine not in spp_engines:
            print(f"{engine} is not a valid spp engine. Options: {spp_engines}")
            return
        with open(json_file_path, "r") as f:
            self.data = json.load(f)
        self.engine = engine
        self.json_file_path = json_file_path
        self.id = os.path.splitext(os.path.basename(json_file_path))[0]
        self.title = self._get_title()

        self.blocks = self._get_blocks()
        self.sentences = self._get_sentences()
        self.sent_block_map = {s.text: s.block_idx for s in self.sentences}
        self.sent_bbox_map = {s.text: s.bboxes for s in self.sentences}
        self.sent_tokens_map = {s.text: s.tokens for s in self.sentences}

        self.sent_sect_map = self._make_sent_sect_map()
        self.sections = list(set(self.sent_sect_map.values()))

    def _get_blocks(self):
        paragraph_blocks = Block.build_blocks_from_spp_json(
            self.json_file_path, type="paragraph"
        )
        list_blocks = Block.build_blocks_from_spp_json(self.json_file_path, type="list")
        return paragraph_blocks + list_blocks

    def _get_sentences(self):
        sentences = []
        for i, block in enumerate(self.blocks):
            for sent in block.sents:
                sent.block_idx = i
                sentences.append(sent)
        for sent in sentences:
            sent.text = self._clean_sentence(sent.text)
        return sentences

    def _get_tokens_for_sentence(self):
        sent_token_map = {}
        for block in self.blocks:
            for token_sent_cluster in block.token_sent_clusters:
                sent_tokens = [block.tokens[i] for i in token_sent_cluster]
                sent = self._clean_sentence(" ".join([t.text for t in sent_tokens]))
                sent_token_map[sent] = sent_tokens
        return sent_token_map

    def _clean_sentence(self, sentence):
        cleaned = sentence.encode("ascii", "ignore")
        cleaned = cleaned.decode()
        cleaned = cleaned.replace("- ", "")
        cleaned = cleaned.strip()
        return cleaned

    def _get_title(self):
        if self.engine == "cermine":
            title_chunks = []
            for page in self.data:
                for b in page["layout"]["blocks"]:
                    if b["type"] == "TITLE":
                        title_chunks.append(b["text"])
            title = " ".join(title_chunks)
        elif self.engine == "detect":
            title_chunks = []
            for page in self.data:
                for bundle in page["layout"]["bundles"]:
                    if bundle["type"] == "title":
                        for toekn in bundle["tokens"]:
                            title_chunks.append(toekn["text"])
            title = " ".join(title_chunks)
        return title

    def _sort_bbox_objs(self, b1, b2):
        # { type: str, bbox: BoundingBox, entity: SectionUnit | SentenceWithBBoxes }
        horizontal_delta = 0.05
        b1_bbox = b1["bbox"]
        b2_bbox = b2["bbox"]
        if b1_bbox.page < b2_bbox.page:
            return -1
        elif b1_bbox.page > b2_bbox.page:
            return 1
        else:
            if abs(b1_bbox.left - b2_bbox.left) < horizontal_delta:
                return -1 if b1_bbox.top < b2_bbox.top else 1
            else:
                return -1 if b1_bbox.left < b2_bbox.left else 1

    def _make_sent_sect_map(self):
        sent_sect_map = {}

        sents_and_sects = [
            {
                "type": "section",
                "bbox": b.bbox,
                "entity": SectionUnit(text=b.text, bbox=b.bbox),
            }
            for b in self.get_section_bboxes()
        ]
        for sent in self.sentences:
            if not sent.bboxes:
                continue
            repr_bbox = sorted(
                [{"bbox": bbox} for bbox in sent.bboxes],
                key=cmp_to_key(self._sort_bbox_objs),
            )[0]["bbox"]
            sents_and_sects.append(
                {"type": "sentence", "bbox": repr_bbox, "entity": sent}
            )

        sents_and_sects = sorted(sents_and_sects, key=cmp_to_key(self._sort_bbox_objs))

        all_sections = {}  # { <section number>: <section text> }
        current_section = ""
        for x in sents_and_sects:
            if x["type"] == "sentence":
                sent_sect_map[x["entity"].text] = current_section
            else:
                current_section = x["entity"].text
                match = re.search(r"[-+]?\d*\.\d+|\d+", current_section)
                if match:
                    sect_number = match.group()
                    all_sections[sect_number] = current_section

        for sent, sect in sent_sect_map.items():
            match = re.search(r"[-+]?\d*\.\d+|\d+", sect)
            if match:
                sect_number = match.group()
            sect_number_parts = sect_number.split(".")

            if len(sect_number_parts) == 3:
                first_level_header = all_sections.get(sect_number_parts[0], "")
                second_level_header = all_sections.get(
                    ".".join(sect_number_parts[:2]), ""
                )
                sent_sect_map[
                    sent
                ] = f"{first_level_header} @@ {second_level_header} @@ {sect}"
            elif len(sect_number_parts) == 2:
                first_level_header = all_sections.get(sect_number_parts[0], "")
                sent_sect_map[sent] = f"{first_level_header} @@ {sect}"

        return sent_sect_map

    def get_section_for_sentence(self, sentence: str):
        return self.sent_sect_map[sentence]

    def get_bboxes_for_span(self, span, sentence) -> List[BoundingBox]:
        matched_tokens = []
        matched_token_indices = []
        prev_find_ind = None

        # Preprocess span to match tokenization schema of sentence tokens
        span = span.replace(" - ", "-")

        for i, token in enumerate(self.sent_tokens_map[sentence]):
            # Preprocess sentence tokens to match tokenization schema of span
            cleaned_token_text = token.text.lower().strip(punctuation)
            if prev_find_ind:
                find_ind = span.find(cleaned_token_text, start=prev_find_ind)
                prev_find_ind = find_ind
            else:
                find_ind = span.find(cleaned_token_text)
            if find_ind != -1:
                matched_tokens.append(token)
                matched_token_indices.append(i)
        longest_matched_token_indices = self._get_longest_consecutive_seq(
            matched_token_indices
        )
        final_matched_tokens = [
            t
            for i, t in enumerate(self.sent_tokens_map[sentence])
            if i in longest_matched_token_indices
        ]
        tokens_by_rows = self._cluster_tokens_by_row(final_matched_tokens)

        final_bboxes = []
        for row in tokens_by_rows:
            final_bboxes.append(_union_bboxes([token.bbox for token in row]))

        return final_bboxes

    def _get_longest_consecutive_seq(self, X: List[int]):
        longest = [X[0]]
        cur = [X[0]]
        for i in range(1, len(X)):
            if X[i] == cur[-1] + 1:
                cur.append(X[i])
            else:
                if len(cur) > len(longest):
                    longest = cur
                cur = [X[i]]
        if len(cur) > len(longest):
            longest = cur
        return longest

    def _cluster_tokens_by_row(self, tokens) -> List[List[int]]:
        if not tokens:
            return []
        clusters_ind = [[0]]
        for token_idx in range(1, len(tokens)):
            current_token = tokens[token_idx]
            prev_token = tokens[token_idx - 1]
            if _are_same_row(bbox1=prev_token.bbox, bbox2=current_token.bbox):
                clusters_ind[-1].append(token_idx)
            else:
                clusters_ind.append([token_idx])

        # map clusters of indices to clusters of tokens
        clusters = []
        for row in clusters_ind:
            clusters.append([tokens[i] for i in row])

        return clusters

    def get_caption_bboxes(self):
        return Block.build_blocks_from_spp_json(
            infile=self.json_file_path, type="caption"
        )

    def get_section_bboxes(self):
        return Block.build_blocks_from_spp_json(
            infile=self.json_file_path, type="section"
        )

    def get_first_sentences(self) -> List[SentenceWithBBoxes]:
        if not self.blocks:
            return []
        first_sents = []
        for block in self.blocks:
            if not block.sents:
                continue
            if block.text[0].isupper():
                first_sent = block.sents[0]
                if len(first_sent.text) >= 15:
                    first_sents.append(first_sent)
            elif block.text[0].isdigit():
                if len(block.sents) > 1:
                    if block.sents[1].text[0].isupper():
                        first_sent = block.sents[1]
                        if len(first_sent.text) >= 15:
                            first_sents.append(first_sent)
        return first_sents


class SciSummPaper:
    def __init__(self, xml_file_path: str):
        self.xml_file_path = xml_file_path
        self.id = os.path.splitext(os.path.basename(xml_file_path))[0]
        self.tree = ET.parse(xml_file_path)
        root = self.tree.getroot()

        self.body = {}
        for child in root:
            if child.tag == "S":
                self.title = child.text
            elif child.tag == "ABSTRACT":
                self.abstract = [sent.text for sent in child]
            elif child.tag == "SECTION":
                self.body[child.attrib["title"]] = [sent.text for sent in child]

    def convert_to_json(self, json_output_dir):
        data = {}
        data["id"] = self.id
        data["title"] = self.title
        data["body"] = self.body
        json_output_path = f"{json_output_dir}/{self.id}.json"
        with open(json_output_path, "w") as out:
            json.dump(data, out, indent=2)

    def get_full_text(self):
        sentences = [sentence for section in self.body.values() for sentence in section]
        return sentences
