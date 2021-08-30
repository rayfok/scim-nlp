import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import itertools

import pysbd

SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=True)


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
        self.sent_sect_map, self.sect_sent_map = self._make_sent_sect_map()
        self.sentences = list(itertools.chain.from_iterable(self.sect_sent_map.values()))

    def _make_sent_sect_map(self):
        sent_sect_map = {}
        sect_sent_map = defaultdict(list)
        for chunk in self.body:
            section = chunk["section"]
            chunk_sents = [s.sent for s in SEGMENTER.segment(chunk["text"])]
            for sent in chunk_sents:
                sent_sect_map[sent] = section
            sect_sent_map[section].extend(chunk_sents)
        return sent_sect_map, sect_sent_map

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
        self.id = os.path.basename(json_file_path)[0]
        self.sentences = self._get_sentences()
        self.title = self._get_title()

    def _get_sentences(self):
        if self.engine == "cermine":
            body_blocks = []
            for page in self.data:
                for b in page["layout"]["blocks"]:
                    if b["type"] == "GEN_BODY":
                        body_blocks.append(b["text"])
            body_text = " ".join(body_blocks)
            body_text = body_text.replace("- ", "")
            sentences = [s.sent for s in SEGMENTER.segment(body_text)]
            return sentences
        elif self.engine == "detect":
            full_chunks = []
            for page in self.data:
                page_chunks = []
                for bundle in page["layout"]["bundles"]:
                    if bundle["type"] != "paragraph":
                        continue
                    bundle_chunks = []
                    for token in bundle["tokens"]:
                        bundle_chunks.append(token["text"])
                    bundle_text = " ".join(bundle_chunks)
                    page_chunks.append(bundle_text)
                page_text = " ".join(page_chunks)
                full_chunks.append(page_text)
            full_text = " ".join(full_chunks)
            full_text = full_text.replace("- ", "")
            sentences = [s.sent.strip() for s in SEGMENTER.segment(full_text)]
            return sentences

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
