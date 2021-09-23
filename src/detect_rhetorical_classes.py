import json
import os
import re
from collections import defaultdict
from typing import List, Union

import spacy
from spacy.symbols import PRON, VERB

from paper import RhetoricUnit, S2OrcPaper, SPPPaper

nlp = spacy.load("en_core_web_md")


class LexicalSet:
    def __init__(self):
        self.lexical_set = {
            "we": ["our", "present study", "this paper", "this work", "this research"],
            "previous": ["previously", "recent", "recently"],
            "thus": ["therefore"],
            "aim": ["objective", "goal", "purpose", "objectives", "goals", "purposes"],
            "question": ["questions", "hypothesis", "hypotheses"],
            "investigate": [
                "explore",
                "study",
                "test",
                "examine",
                "evaluate",
                "assess",
                "determine",
                "characterize",
                "analyze",
                "report",
                "seek",
                # "present",
            ],
            "use": ["employ"],
            "method": ["algorithm", "assay"],
            "observe": ["we see", "find", "show"],
            "conclude": ["conclusion", "summarize", "summary"],
            "suggest": [
                "illustrate",
                "demonstrate",
                "imply",
                "indicate",
                "confirm",
                "reflect",
                "support",
                "prove",
                "reveal",
            ],
            "because": ["result from", "attribute to"],
            "likely": ["probable", "probably", "possible", "possibly", "may", "could"],
            "need": ["remain"],
            "future": ["further"],
            "consistent": [
                "match",
                "agree",
                "support",
                "in line",
                "in agreement",
                "similar",
                "same",
                "analogous",
            ],
            "inconsistent": [
                "conflicting",
                "conflict",
                "contrast",
                "contrary",
                "differ",
                "different",
                "difference",
                "unlike",
            ],
            "than": ["compare"],
            "however": ["other hand", "although", "though", "but", "while", "despite"],
            "extend": ["extension", "extends"],
            "contribution": ["contributions", "contribute", "contributes"],
        }

    def get_roots(self):
        return self.lexical_set.keys()

    def get_aliases(self, root):
        return self.lexical_set.get(root, [])


class AZClassifier:
    def __init__(self, json_file_path: str, dataset: str):
        self.json_file_path = json_file_path
        self.dataset = dataset

        self.id = os.path.splitext(os.path.basename(json_file_path))[0]

        if dataset == "s2orc":
            self.paper = S2OrcPaper(json_file_path)
        elif dataset == "spp":
            self.paper = SPPPaper(json_file_path, "detect")
        elif dataset == "spp-cermine":
            self.paper = SPPPaper(json_file_path, "cermine")

        self.lexical_set = LexicalSet()

    def _sentence_contains_root_or_aliases(
        self, sentence: str, roots: Union[str, List[str]]
    ):
        if isinstance(roots, str):
            roots = [roots]
        sentence = sentence.lower().strip()
        found = []
        for root in roots:
            aliases = self.lexical_set.get_aliases(root)
            for token in [root, *aliases]:
                if re.search(r"\b{}\b".format(token), sentence):
                    found.append(token)
        return found

    def _has_section(self, aliases: List[str]):
        for section in self.paper.sections:
            for a in aliases:
                if a in section.lower():
                    return True
        return False

    def _is_in_introduction(self, sentence: str):
        section_found = self.paper.get_section_for_sentence(sentence)
        section_found = section_found.lower()
        return "introduction" in section_found

    def _is_in_conclusion(self, sentence: str):
        section_found = self.paper.get_section_for_sentence(sentence)
        section_found = section_found.lower()
        aliases = ["conclusion", "concluding"]
        return any(a in section_found for a in aliases)

    def _is_in_discussion(self, sentence: str):
        section_found = self.paper.get_section_for_sentence(sentence)
        section_found = section_found.lower()
        return "discussion" in section_found

    def _is_in_related_work(self, sentence: str):
        section_found = self.paper.get_section_for_sentence(sentence)
        section_found = section_found.lower()
        aliases = ["related work", "background", "relatedwork"]
        return any(a in section_found for a in aliases)

    def _is_in_future_work(self, sentence: str):
        section_found = self.paper.get_section_for_sentence(sentence)
        section_found = section_found.lower()
        aliases = ["future work"]
        return any(a in section_found for a in aliases)

    def _is_in_result(self, sentence: str):
        section_found = self.paper.get_section_for_sentence(sentence)
        section_found = section_found.lower()
        aliases = ["result", "evaluation", "finding", "study", "experiment"]
        return any(a in section_found for a in aliases)

    def _is_in_method(self, sentence: str):
        section_found = self.paper.get_section_for_sentence(sentence)
        section_found = section_found.lower()
        aliases = ["method", "system"]
        return any(a in section_found for a in aliases)

    def _is_in_expected_section(self, sentence: str, label: str):
        if label == "contribution":
            return self._is_in_introduction(sentence)
        elif label == "objective":
            return self._is_in_introduction(sentence) or self._is_in_conclusion(
                sentence
            )
        elif label == "novelty":
            return self._is_in_related_work(sentence) or self._is_in_introduction(
                sentence
            )
        elif label == "result":
            return (
                self._is_in_introduction(sentence)
                or self._is_in_result(sentence)
                or self._is_in_discussion(sentence)
                or self._is_in_conclusion(sentence)
            )
        elif label == "conclusion":
            return self._is_in_conclusion(sentence) or self._is_in_discussion(sentence)
        elif label == "future work":
            return (
                self._is_in_introduction(sentence)
                or self._is_in_conclusion(sentence)
                or self._is_in_discussion(sentence)
                or self._is_in_future_work(sentence)
            )
        elif label == "method":
            return not (
                self._is_in_related_work(sentence)
                or self._is_in_result(sentence)
                or self._is_in_future_work(sentence)
                or self._is_in_discussion(sentence)
            )
        else:
            return False

    def detect_contribution(self):
        print("=== Contribution ===")
        detected = []
        for sent_bbox_obj in self.paper.sentences:
            sentence = sent_bbox_obj.text
            bboxes = sent_bbox_obj.bboxes
            found = self._sentence_contains_root_or_aliases(sentence, ["contribution"])

            if not found:
                continue

            section = self.paper.get_section_for_sentence(sentence)
            is_author_statement = (
                len(self._sentence_contains_root_or_aliases(sentence, "we")) > 0
            )
            is_in_expected_section = self._is_in_expected_section(
                sentence, "contribution"
            )
            rhetoric_unit = RhetoricUnit(
                text=sentence,
                label="Contribution",
                bboxes=bboxes,
                section=section,
                prob=None,
                is_author_statement=is_author_statement,
                is_in_expected_section=is_in_expected_section,
            )
            detected.append(rhetoric_unit)
        return detected

    def detect_objective(self):
        print("=== Objective ===")
        detected = []
        for sent_bbox_obj in self.paper.sentences:
            sentence = sent_bbox_obj.text
            bboxes = sent_bbox_obj.bboxes
            aim_noun_found = self._sentence_contains_root_or_aliases(
                sentence, ["aim", "question"]
            )
            aim_verb_found = self._sentence_contains_root_or_aliases(
                sentence, ["investigate"]
            )

            if not aim_noun_found and not aim_verb_found:
                continue

            # only consider when contribution roots and alises are verbs
            doc = nlp(sentence.lower())
            keep_aim_tokens = []
            for token in doc:
                if token.pos_ != VERB:
                    continue
                if token.text in aim_verb_found:
                    keep_aim_tokens.append(token)
            aim_found = aim_noun_found + keep_aim_tokens

            if not aim_found:
                continue

            section = self.paper.get_section_for_sentence(sentence)
            is_author_statement = (
                len(self._sentence_contains_root_or_aliases(sentence, "we")) > 0
            )
            is_in_expected_section = self._is_in_expected_section(sentence, "objective")
            rhetoric_unit = RhetoricUnit(
                text=sentence,
                label="Objective",
                bboxes=bboxes,
                section=section,
                prob=None,
                is_author_statement=is_author_statement,
                is_in_expected_section=is_in_expected_section,
            )
            detected.append(rhetoric_unit)
        return detected

    def detect_novelty(self):
        print("=== Novelty ===")
        detected = []
        for sent_bbox_obj in self.paper.sentences:
            sentence = sent_bbox_obj.text
            bboxes = sent_bbox_obj.bboxes
            found = self._sentence_contains_root_or_aliases(
                sentence, ["inconsistent", "however", "extend"]
            )

            if not found:
                continue

            section = self.paper.get_section_for_sentence(sentence)
            is_author_statement = (
                len(self._sentence_contains_root_or_aliases(sentence, "we")) > 0
            )
            is_in_expected_section = self._is_in_expected_section(sentence, "novelty")
            rhetoric_unit = RhetoricUnit(
                text=sentence,
                label="Novelty",
                bboxes=bboxes,
                section=section,
                prob=None,
                is_author_statement=is_author_statement,
                is_in_expected_section=is_in_expected_section,
            )
            detected.append(rhetoric_unit)
        return detected

    def detect_result(self):
        print("=== Result ===")
        detected = []
        for sent_bbox_obj in self.paper.sentences:
            sentence = sent_bbox_obj.text
            bboxes = sent_bbox_obj.bboxes
            found = self._sentence_contains_root_or_aliases(sentence, ["observe"])

            if not found:
                continue

            section = self.paper.get_section_for_sentence(sentence)
            is_author_statement = (
                len(self._sentence_contains_root_or_aliases(sentence, "we")) > 0
            )
            is_in_expected_section = self._is_in_expected_section(sentence, "result")
            rhetoric_unit = RhetoricUnit(
                text=sentence,
                label="Result",
                bboxes=bboxes,
                section=section,
                prob=None,
                is_author_statement=is_author_statement,
                is_in_expected_section=is_in_expected_section,
            )
            detected.append(rhetoric_unit)
        return detected

    def detect_conclusion(self):
        print("=== Conclusion ===")
        detected = []
        for sent_bbox_obj in self.paper.sentences:
            sentence = sent_bbox_obj.text
            bboxes = sent_bbox_obj.bboxes
            con_found = self._sentence_contains_root_or_aliases(sentence, ["conclude"])
            con_likely_found = self._sentence_contains_root_or_aliases(
                sentence, ["suggest", "thus", "likely", "because"]
            )

            if not con_found and not con_likely_found:
                continue

            section = self.paper.get_section_for_sentence(sentence)
            is_author_statement = (
                len(self._sentence_contains_root_or_aliases(sentence, "we")) > 0
            )
            is_in_expected_section = self._is_in_expected_section(
                sentence, "conclusion"
            )
            if con_found:
                prob = 1
            elif con_likely_found:
                prob = 0.5
            rhetoric_unit = RhetoricUnit(
                text=sentence,
                label="Conclusion",
                bboxes=bboxes,
                section=section,
                prob=prob,
                is_author_statement=is_author_statement,
                is_in_expected_section=is_in_expected_section,
            )
            detected.append(rhetoric_unit)
        return detected

    def detect_future_work(self):
        print("=== Future Work ===")
        detected = []
        for sent_bbox_obj in self.paper.sentences:
            sentence = sent_bbox_obj.text
            bboxes = sent_bbox_obj.bboxes
            found = self._sentence_contains_root_or_aliases(
                sentence, ["will", "need", "future"]
            )

            if not found:
                continue

            section = self.paper.get_section_for_sentence(sentence)
            is_author_statement = (
                len(self._sentence_contains_root_or_aliases(sentence, "we")) > 0
            )
            is_in_expected_section = self._is_in_expected_section(
                sentence, "future work"
            )
            rhetoric_unit = RhetoricUnit(
                text=sentence,
                label="Future Work",
                bboxes=bboxes,
                section=section,
                prob=None,
                is_author_statement=is_author_statement,
                is_in_expected_section=is_in_expected_section,
            )
            detected.append(rhetoric_unit)
        return detected

    def detect_method(self):
        print("=== Method ===")
        detected = []
        seen_blocks = set()
        for sent_bbox_obj in self.paper.sentences:
            sentence = sent_bbox_obj.text
            bboxes = sent_bbox_obj.bboxes
            is_author_statement = (
                len(self._sentence_contains_root_or_aliases(sentence, "we")) > 0
            )
            if not is_author_statement:
                continue

            block_idx = sent_bbox_obj.block_idx
            if block_idx in seen_blocks:
                continue

            seen_blocks.add(sent_bbox_obj.block_idx)
            section = self.paper.get_section_for_sentence(sentence)
            is_in_expected_section = self._is_in_expected_section(
                sentence, "method"
            )
            rhetoric_unit = RhetoricUnit(
                text=sentence,
                label="Method",
                bboxes=bboxes,
                section=section,
                prob=None,
                is_author_statement=is_author_statement,
                is_in_expected_section=is_in_expected_section,
            )
            detected.append(rhetoric_unit)
        return detected

    def make_ssc_rhetoric_unit(self, sentence: str, label: str, prob: float = None):
        is_author_statement = (
            len(self._sentence_contains_root_or_aliases(sentence, "we")) > 0
        )
        ssc_rhetoric_unit_label_map = {
            "Method": "method",
            "Objective": "objective",
            "Result": "result",
        }
        is_in_expected_section = self._is_in_expected_section(
            sentence, ssc_rhetoric_unit_label_map[label]
        )
        return RhetoricUnit(
            text=sentence,
            label=label,
            bboxes=self.paper.sent_bbox_map[sentence],
            section=self.paper.sent_sect_map[sentence],
            prob=prob,
            is_author_statement=is_author_statement,
            is_in_expected_section=is_in_expected_section,
        )

    def _token_to_noun_chunk(self, doc):
        noun_chunks = [nc for nc in doc.noun_chunks]
        token_chunks = {}
        for tok in doc:
            for nc in noun_chunks:
                if tok.i >= nc.start and tok.i < nc.end:
                    token_chunks[tok] = nc
                    break
        return token_chunks

    def _get_noun_chunk_after(self, noun_chunks, i):
        for nc in noun_chunks:
            if i >= nc.start and i < nc.end:
                return nc

    def get_author_statements(self):
        author_statements = []
        for sentence in self.paper.sentences:
            if self._sentence_contains_root_or_aliases(sentence.text, "we"):
                author_statements.append(sentence)
        return author_statements

    def get_short_author_statements(self):
        short_statements = []
        author_statements = self.get_author_statements()
        for sent in author_statements:
            doc = nlp(sent.text.lower())
            clause_tokens = []
            pron_found = False
            verb_found = False
            noun_chunk_found = False
            clause_completed = False
            sconj_found = False
            for token in doc:
                if token.lemma_ == "-PRON-" or token.pos == PRON:
                    pron_found = True
                    clause_completed = False
                if pron_found and not clause_completed:
                    if token.pos == VERB:
                        clause_tokens.append(token.text)
                        verb_found = True
                    elif token.lemma_ == "that":
                        clause_tokens.append(token.text)
                        sconj_found = True
                    elif verb_found:
                        if not noun_chunk_found:
                            clause_tokens.append(token.text)
                            if self._get_noun_chunk_after(doc.noun_chunks, token.i):
                                noun_chunk_found = True
                        else:
                            if not self._get_noun_chunk_after(doc.noun_chunks, token.i):
                                if sconj_found:
                                    clause_tokens.append(token.text)
                                    noun_chunk_found = False
                                    sconj_found = False
                                else:
                                    short_statement = " ".join(clause_tokens)
                                    short_statements.append((short_statement, sent))
                                    break
                            else:
                                clause_tokens.append(token.text)
                    else:
                        clause_tokens.append(token.text)
        return short_statements


if __name__ == "__main__":
    # Run on s2orc_acl_20200705v1 data
    ssc_output_file = "data/ssc-input/s2orc_acl_2016"
    DATA_DIR = "data/s2orc_acl_20200705v1"
    by_year = defaultdict(list)
    for file_e in os.scandir(DATA_DIR):
        if not os.path.isfile(file_e.path):
            continue
        with open(file_e.path, "r") as f:
            paper = json.load(f)
            metadata = paper["metadata"]
        if "year" not in metadata:
            continue
        if "arxiv_id" not in metadata:
            continue
        year = metadata["year"]
        arxiv_id = metadata["arxiv_id"]
        if year and arxiv_id:
            by_year[year].append((file_e.path, arxiv_id))
    for paper in by_year[2016]:
        paper_path, arxiv_id = paper
        print(paper_path)
        detected = []
        azc = AZClassifier(paper_path, "s2orc")
        detected += azc.detect_contribution()
        detected += azc.detect_novelty()
        detected += azc.detect_objective()
        detected += azc.detect_conclusion()
        detected += azc.detect_future_work()
