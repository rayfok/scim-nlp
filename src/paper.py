import json
import os

import pysbd

SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=True)


class S2OrcPaper:
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        self.id = data["paper_id"]
        self.authors = data["authors"]
        self.title = data["title"]
        self.mag_field_of_study = data["mag_field_of_study"]
        self.body = data["pdf_parse"]["body_text"]
        self.abstract = data["abstract"]
        self.sentences = [s.sent for s in SEGMENTER.segment(self._get_full_text())]

    def _get_full_text(self):
        return " ".join([chunk["text"] for chunk in self.body])


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
