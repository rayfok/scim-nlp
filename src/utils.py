import os
from collections import defaultdict
from typing import Any, List

import jsonlines

from paper import SPPPaper

DATA_ROOT = "data"
SPP_OUTPUT_DIR = f"{DATA_ROOT}/spp-output"
SSC_INPUT_DIR = f"{DATA_ROOT}/ssc-input"
SSC_OUTPUT_DIR = f"{DATA_ROOT}/ssc-output"


def make_ssc_input(id: str, sentences: List[str]) -> None:
    os.makedirs(SSC_INPUT_DIR, exist_ok=True)
    with jsonlines.open(f"{SSC_INPUT_DIR}/{id}.jsonl", "w") as out:
        out.write({"paper_id": id, "sentences": sentences})


def make_spp_output_to_ssc_input(arxiv_id: str):
    p = SPPPaper(f"{SPP_OUTPUT_DIR}/{arxiv_id}.json")
    make_ssc_input(arxiv_id, p.sentences)


def get_top_k_ssc_pred(data: Any, k: int = 5):
    output = {}  # { paper_id: { <label>: [<pred>] }
    for paper_id, preds in data.items():
        output[paper_id] = {}
        by_cat = defaultdict(list)
        for pred in preds:
            by_cat[pred["label"]].append(pred)
        for cat, cat_preds in by_cat.items():
            cat_preds_sorted = sorted(cat_preds, key=lambda x: x["prob"], reverse=True)
            output[paper_id][cat] = cat_preds_sorted[:k]
    return output
