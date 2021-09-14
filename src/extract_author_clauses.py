import argparse
import os
from typing import List

import spacy
from spacy.symbols import VERB

from paper import S2OrcPaper, SPPPaper
from utils import make_ssc_input

# Requires spacy>=3.0 and en_core_web_trf downloaded
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


def extract_author_sentences(json_file_path: str, dataset: str):
    if dataset == "s2orc":
        paper = S2OrcPaper(json_file_path)
    elif dataset == "spp":
        paper = SPPPaper(json_file_path, "detect")
    elif dataset == "spp-cermine":
        paper = SPPPaper(json_file_path, "cermine")
    sents = paper.sentences
    sentences = []
    for sent in sents:
        sent_split = sent.lower().split()
        if any(pron in sent_split for pron in ["we", "our"]):
            sentences.append(sent)
            continue
        if any(p in sent.lower() for p in ["this paper"]):
            sentences.append(sent)
    return sentences


def extract_author_clauses(json_file_path: str, dataset: str):
    sentences = extract_author_sentences(json_file_path, dataset)
    clauses = [(condense_author_clause(s), s) for s in sentences]
    clauses = [c for c in clauses if c[0] is not None]
    return clauses


def condense_author_clause(sent: str):
    doc = nlp(sent)
    clause_tokens = []
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
                clause_tokens.append(token.text)
                verb_found = True
            elif token.lemma_ == "that":
                clause_tokens.append(token.text)
                sconj_found = True
            elif verb_found:
                if not noun_chunk_found:
                    clause_tokens.append(token.text)
                    if get_noun_chunk_after(doc.noun_chunks, token.i):
                        noun_chunk_found = True
                else:
                    if not get_noun_chunk_after(doc.noun_chunks, token.i):
                        if sconj_found:
                            clause_tokens.append(token.text)
                            noun_chunk_found = False
                            sconj_found = False
                        else:
                            final_clause = " ".join(clause_tokens)
                            return final_clause
                    else:
                        clause_tokens.append(token.text)
            else:
                clause_tokens.append(token.text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, help="Path to input json file", required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset which contains the specified input file",
        choices=["s2orc", "spp", "spp-cermine"],
        required=True,
    )
    parser.add_argument(
        "--export_in_ssc_format",
        "-e",
        action="store_true",
        help="Export author clauses in JSONL format for input to ssc module",
    )
    args = parser.parse_args()
    sentences = extract_author_sentences(args.input_file, args.dataset)
    if args.export_in_ssc_format:
        filename = os.path.splitext(os.path.basename(args.input_file))[0]
        make_ssc_input(id=filename, sentences=sentences)
    clauses = extract_author_clauses(args.input_file, args.dataset)
    for c in clauses:
        print(c, "\n")


if __name__ == "__main__":
    main()
