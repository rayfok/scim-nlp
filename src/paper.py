import difflib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from string import punctuation
from typing import Dict, List

import pysbd

from bbox import Block, BoundingBox, _are_same_row, _bbox_to_json, _union_bboxes

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
        self.sent_bbox_map = {s.text: s.bboxes for s in self.sentences}
        self.sent_tokens_map = {s.text: s.tokens for s in self.sentences}

        self.sent_sect_map = self._make_sent_sect_map()
        self.sections = list(set(self.sent_sect_map.values()))

    def _get_blocks(self):
        return Block.build_blocks_from_spp_json(self.json_file_path)

    def _get_sentences(self):
        sentences = []
        for block in self.blocks:
            sentences.extend([sent for sent in block.sents])
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

    def _normalize_section_headers(self, section_data):
        top_level_sections = {}
        for section in section_data["sections"]:
            header = section.get("header", None)
            if header:
                section_number_list = re.findall(r"[-+]?\d*\.\d+|\d+", header)
                if (
                    len(section_number_list) == 1
                ):  # Make sure there's only one section number per header
                    section_number = section_number_list[0]
                    if (
                        "." not in section_number
                    ):  # no decimal point means top level section
                        top_level_sections[section_number] = header
        # For all non top level section headers, append top level section header
        for section in section_data["sections"]:
            header = section.get("header", None)
            if header:
                section_number_list = re.findall(r"[-+]?\d*\.\d+|\d+", header)
                if len(section_number_list) == 1:
                    section_number = section_number_list[0]
                    if "." in section_number:
                        top_level_section_number = section_number.split(".")[0]
                        top_level_section_header = top_level_sections.get(
                            top_level_section_number, None
                        )
                        if top_level_section_header:
                            section[
                                "header"
                            ] = f"{top_level_section_header} {section['header']}"

    def _make_sent_sect_map(self):
        SECTIONS_DIR = "data/sections"
        with open(f"{SECTIONS_DIR}/{self.id}.json", "r") as f:
            section_data = json.load(f)
        self._normalize_section_headers(section_data)

        sent_to_sect = {}
        for section in section_data["sections"]:
            header = section.get("header", None)
            if not header:
                continue
            body = section.get("body", None)
            sents = [span.sent.strip() for span in SEGMENTER.segment(body)]
            for sent in sents:
                sent_to_sect[sent] = header

        for sent in self.sentences:
            cands = difflib.get_close_matches(sent.text, sent_to_sect.keys())
            num_close_matches = len(cands)

            # If there's exactly one close match, use it for the section label
            if num_close_matches == 1:
                section = sent_to_sect[cands[0]]

            # If there's no close match, give the sentence a None section label
            elif num_close_matches == 0:
                section = ""

            # If there's more than one close match:
            #   - If all close matches have the same section label, use that label
            #   - If there are multiple, use the label of the closest match
            else:
                cand_sections = list(set([sent_to_sect[c] for c in cands]))
                if len(cand_sections) == 1:
                    section = cand_sections[0]
                else:
                    section = ""
            sent_to_sect[sent.text] = section

        # For unlabeled sentences, if the previous and following labeled sentences
        # have the same label, use that label for the unlabeled sentence as well.
        for i, sent in enumerate(self.sentences):
            if sent_to_sect[sent.text] == "":
                prev_label = ""
                j = i - 1
                while j >= 0:
                    cand_prev_label = sent_to_sect[self.sentences[j].text]
                    if cand_prev_label != "":
                        prev_label = cand_prev_label
                        break
                    j = j - 1

                next_label = ""
                j = i + 1
                while j < len(self.sentences):
                    cand_next_label = sent_to_sect[self.sentences[j].text]
                    if cand_next_label != "":
                        next_label = cand_next_label
                        break
                    j = j + 1
                if prev_label == next_label:
                    sent_to_sect[sent] = next_label

        return sent_to_sect

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
