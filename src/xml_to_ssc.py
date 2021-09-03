import json
import os

import jsonlines

from paper import SciSummPaper

INPUT_DIR = "data/scisummnet/top1000_complete"
OUTPUT_DIR = "data/ssc-input"
OUTPUT_FILE = "scisummnet.jsonl"
os.makedirs(OUTPUT_DIR, exist_ok=True)


processed = []
for paper in os.scandir(INPUT_DIR):
    if not os.path.isdir(paper.path):
        continue
    for xml_file in os.scandir(os.path.join(paper.path, "Documents_xml")):
        if not os.path.isfile(xml_file.path):
            continue
        p = SciSummPaper(xml_file.path)
        sentences = p.get_full_text()
        processed.append(
            {
                "paper_id": p.id,
                "sentences": sentences
            }
        )
with jsonlines.open(f"{OUTPUT_DIR}/{OUTPUT_FILE}", "w") as out:
    out.write_all(processed)
