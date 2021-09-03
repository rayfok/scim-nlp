import difflib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List

import pysbd

from bbox import Block, BoundingBox, _bbox_to_json

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
        self.sentences_bbox = self._get_sentences_with_bbox()

        self.sent_sect_map = self._make_sent_sect_map()
        self.sections = list(set(self.sent_sect_map.values()))

    def _get_blocks(self):
        return Block.build_blocks_from_spp_json(self.json_file_path)

    def _get_sentences(self):
        sents = []
        for block in self.blocks:
            sents.extend([self._clean_sentence(sent) for sent in block.sents])
        return sents

    def _get_sentences_with_bbox(self):
        sents_bbox = []
        for block in self.blocks:
            sents_bbox.extend([sent for sent in block.sents])
        for sent in sents_bbox:
            sent.text = self._clean_sentence(sent)
        return sents_bbox

    def _clean_sentence(self, sentence):
        sentence_text = sentence.text.encode("ascii", "ignore")
        sentence_text = sentence_text.decode()
        sentence_text = sentence_text.replace("- ", "")
        return sentence_text

    # def _get_sentences(self):
    #     if self.engine == "cermine":
    #         body_blocks = []
    #         for page in self.data:
    #             for b in page["layout"]["blocks"]:
    #                 if b["type"] == "GEN_BODY":
    #                     body_blocks.append(b["text"])
    #         body_text = " ".join(body_blocks)
    #         body_text = body_text.replace("- ", "")
    #         sentences = [s.sent for s in SEGMENTER.segment(body_text)]
    #         return sentences
    #     elif self.engine == "detect":
    #         full_chunks = []
    #         for page in self.data:
    #             page_chunks = []
    #             for bundle in page["layout"]["bundles"]:
    #                 if bundle["type"] != "paragraph":
    #                     continue
    #                 bundle_chunks = []
    #                 for token in bundle["tokens"]:
    #                     bundle_chunks.append(token["text"])
    #                 bundle_text = " ".join(bundle_chunks)
    #                 page_chunks.append(bundle_text)
    #             page_text = " ".join(page_chunks)
    #             full_chunks.append(page_text)
    #         full_text = " ".join(full_chunks)
    #         full_text = full_text.replace("- ", "")
    #         sentences = [s.sent.strip() for s in SEGMENTER.segment(full_text)]
    #         return sentences

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

    def _make_sent_sect_map(self):
        SECTIONS_DIR = "data/sections"
        with open(f"{SECTIONS_DIR}/{self.id}.json", "r") as f:
            section_data = json.load(f)
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
            cands = difflib.get_close_matches(sent, sent_to_sect.keys())
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
            sent_to_sect[sent] = section

        # For unlabeled sentences, if the previous and following labeled sentences
        # have the same label, use that label for the unlabeled sentence as well.
        for i, sent in enumerate(self.sentences):
            if sent_to_sect[sent] == "":
                prev_label = ""
                j = i - 1
                while j >= 0:
                    cand_prev_label = sent_to_sect[self.sentences[j]]
                    if cand_prev_label != "":
                        prev_label = cand_prev_label
                        break
                    j = j - 1

                next_label = ""
                j = i + 1
                while j < len(self.sentences):
                    cand_next_label = sent_to_sect[self.sentences[j]]
                    if cand_next_label != "":
                        next_label = cand_next_label
                        break
                    j = j + 1
                if prev_label == next_label:
                    sent_to_sect[sent] = next_label

        return sent_to_sect

    def get_section_for_sentence(self, sentence):
        return self.sent_sect_map[sentence]


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
