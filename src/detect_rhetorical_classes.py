from paper import S2OrcPaper, SPPPaper

import spacy
from spacy.symbols import VERB
import os
import re

nlp = spacy.load("en_core_web_md")


class LexicalSet:
    def __init__(self):
        self.lexical_set = {
            "we": ["our", "present study", "this paper", "this work"],
            "previous": ["previously", "recent", "recently"],
            "thus": ["therefore"],
            "aim": ["objective", "goal", "purpose"],
            "question": ["hypothesis"],
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
                # "present",
            ],
            "use": ["employ"],
            "method": ["algorithm", "assay"],
            "observe": ["see", "find", "show"],
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
            ],
            "than": ["compare"],
            "however": ["other hand", "although", "though", "but"],
            "contribution": ["contribute"],
        }

    def get_roots(self):
        return self.lexical_set.keys()

    def get_aliases(self, root):
        return self.lexical_set.get(root, [])


class AZClassifier:
    def __init__(self, json_file_path: str, dataset: str):
        self.json_file_path = json_file_path
        self.dataset = dataset

        if dataset == "s2orc":
            self.paper = S2OrcPaper(json_file_path)
        elif dataset == "spp":
            self.paper = SPPPaper(json_file_path, "detect")
        elif dataset == "spp-cermine":
            self.paper = SPPPaper(json_file_path, "cermine")

        self.sentences = self.paper.sentences
        self.lexical_set = LexicalSet()

    def _sentence_contains_root_or_aliases(self, sentence, roots):
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

    def detect_novelty(self):
        for i, sentence in enumerate(self.sentences):
            # doc = nlp(sentence.lower())
            # for token in doc:
            #     print(token.text, token.pos_, token.tag_, token.dep_)
            diff_found = self._sentence_contains_root_or_aliases(
                sentence, ["inconsistent", "than", "however"]
            )
            we_found = self._sentence_contains_root_or_aliases(sentence, "we")
            if len(diff_found) > 0 and len(we_found) > 0:
                print(sentence, "\n")

    def detect_contribution(self):
        for sentence in self.sentences:
            cont_found = self._sentence_contains_root_or_aliases(
                sentence, ["contribution"]
            )
            we_found = self._sentence_contains_root_or_aliases(sentence, "we")
            if len(cont_found) > 0 and len(we_found) > 0:
                section_found = self.paper.get_section_for_sentence(sentence)
                if "introduction" in section_found.lower():
                    print(
                        f"tokens: {cont_found}\nsection: {self.paper.get_section_for_sentence(sentence)}\n{sentence}\n"
                    )

    def detect_objective(self):
        for sentence in self.sentences:
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
                if token.text in aim_verb_found and token.pos_ == VERB:
                    keep_aim_tokens.append(token)

            aim_found = aim_noun_found + keep_aim_tokens
            we_found = self._sentence_contains_root_or_aliases(sentence, "we")
            if aim_found and we_found:
                section_found = self.paper.get_section_for_sentence(sentence)
                if (
                    "introduction" in section_found.lower()
                    or "conclusion" in section_found.lower()
                ):
                    print(
                        f"tokens: {aim_found}\nsection: {self.paper.get_section_for_sentence(sentence)}\n{sentence}\n"
                    )


if __name__ == "__main__":
    azc = AZClassifier("data/s2orc_acl_20200705v1/210987719.json", "s2orc")
    for paper in os.scandir("data/s2orc_acl_20200705v1/"):
        azc = AZClassifier(paper.path, "s2orc")
        print(azc.json_file_path)
        # azc.detect_contribution()
        # azc.detect_novelty()
        azc.detect_objective()
