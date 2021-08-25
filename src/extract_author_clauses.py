import json
import os

import spacy
from spacy.symbols import VERB

from paper import S2OrcPaper, SPPPaper

nlp = spacy.load("en_core_web_trf")


def token_to_noun_chunk(doc):
    noun_chunks = [nc for nc in doc.noun_chunks]
    token_chunks = {}
    for tok in doc:
        for nc in noun_chunks:
            if tok.i >= nc.start and tok.i < nc.end:
                token_chunks[tok] = nc
                break
    return token_chunks


def get_noun_chunk_after(noun_chunks, i):
    for nc in noun_chunks:
        if i >= nc.start and i < nc.end:
            return nc


def extract_author_clauses(json_file_path: str, dataset):
    print(json_file_path)

    if dataset == "s2orc":
        paper = S2OrcPaper(json_file_path)
    elif dataset == "spp":
        paper = SPPPaper(json_file_path, "detect")
    elif dataset == "spp-cermine":
        paper = SPPPaper(json_file_path, "cermine")
    sents = paper.sentences
    clauses = []
    for sent in sents:
        sent_split = sent.lower().split()
        if any(pron in sent_split for pron in ["we", "our"]):
            doc = nlp(sent)
            # for token in doc:
            #     print(token.text, token.pos_, token.tag_, token.dep_)
            clause = []
            pron_found = False
            verb_found = False
            noun_chunk_found = False
            clause_completed = False
            sconj_found = False
            for token in doc:
                if token.lemma_ == "we":
                    pron_found = True
                    clause_completed = False
                if pron_found and not clause_completed:
                    if token.pos == VERB:
                        clause.append(token.text)
                        verb_found = True
                    elif token.lemma_ == "that":
                        clause.append(token.text)
                        sconj_found = True
                    elif verb_found:
                        if not noun_chunk_found:
                            clause.append(token.text)
                            if get_noun_chunk_after(doc.noun_chunks, token.i):
                                noun_chunk_found = True
                        else:
                            if not get_noun_chunk_after(doc.noun_chunks, token.i):
                                if sconj_found:
                                    clause.append(token.text)
                                    noun_chunk_found = False
                                    sconj_found = False
                                else:
                                    clause_completed = True
                                    clauses.append(" ".join(clause))
                                    clause = []
                            else:
                                clause.append(token.text)
                    else:
                        clause.append(token.text)

    return clauses


def run_example():
    json_file_path = "data/spp/hsu_smell-pittsburgh.json"
    clauses = extract_author_clauses(json_file_path, "spp")
    for c in clauses:
        print(c, "\n")


if __name__ == "__main__":
    run_example()
