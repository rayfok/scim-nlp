import argparse
import json
import os
from pprint import pprint

from detect_rhetorical_classes import AZClassifier
from paper import *
from paper import RhetoricUnit
from run_spp import get_parsed_arxiv_pdf
from run_ssc import run_ssc
from utils import (
    SPP_OUTPUT_DIR,
    SSC_INPUT_DIR,
    SSC_OUTPUT_DIR,
    SSC_ABSTRACT_INPUT_DIR,
    SSC_ABSTRACT_OUTPUT_DIR,
    get_top_k_ssc_pred,
    get_top_pred_by_block,
    make_spp_output_to_ssc_input,
)

OUTPUT_ROOT_DIR = "output"


def main(args):
    output_path = f"{OUTPUT_ROOT_DIR}/facets"
    sections_output_path = f"{OUTPUT_ROOT_DIR}/sections"
    captions_output_path = f"{OUTPUT_ROOT_DIR}/captions"
    first_sentence_output_path = f"{OUTPUT_ROOT_DIR}/first_sentences"
    all_sentences_output_path = f"{OUTPUT_ROOT_DIR}/all_sentences"
    abstract_output_path = f"{OUTPUT_ROOT_DIR}/abstract"

    paths = [
        output_path,
        sections_output_path,
        captions_output_path,
        first_sentence_output_path,
        all_sentences_output_path,
        abstract_output_path,
    ]
    if all(os.path.exists(f"{p}/{args.arxiv_id}.json") for p in paths):
        return

    for p in paths:
        os.makedirs(p, exist_ok=True)

    rhetorical_units = []  # list to hold outputs

    # Run scienceparseplus to detect all tokens and bounding boxes
    spp_output_file = f"{SPP_OUTPUT_DIR}/{args.arxiv_id}.json"
    if not os.path.exists(spp_output_file):
        layout = get_parsed_arxiv_pdf(args.arxiv_id)
        with open(spp_output_file, "w") as out:
            json.dump(layout, out, indent=2)

    azc = AZClassifier(spp_output_file, dataset="spp")

    # Convert scienceparseplus output into a data format for sequential-sentence-classification
    ssc_input_file = f"{SSC_INPUT_DIR}/{args.arxiv_id}.jsonl"
    ssc_abstract_input_file = f"{SSC_ABSTRACT_INPUT_DIR}/{args.arxiv_id}.jsonl"
    if not os.path.exists(ssc_input_file) or not os.path.exists(
        ssc_abstract_input_file
    ):
        make_spp_output_to_ssc_input(args.arxiv_id)

    # Run sequential-sentence-classification to classify each sentence into one of
    # five categories: Background, Objective, Method, Result, Other.
    os.makedirs(SSC_OUTPUT_DIR, exist_ok=True)
    ssc_output_file = f"{SSC_OUTPUT_DIR}/{args.arxiv_id}.json"
    if not os.path.exists(ssc_output_file):
        model_path = "sequential_sentence_classification/model.tar.gz"
        run_ssc(
            model_path=model_path,
            test_jsonl_file=ssc_input_file,
            output_json_file=ssc_output_file,
        )

    os.makedirs(SSC_ABSTRACT_OUTPUT_DIR, exist_ok=True)
    ssc_abstract_output_file = f"{SSC_ABSTRACT_OUTPUT_DIR}/{args.arxiv_id}.json"
    if not os.path.exists(ssc_abstract_output_file):
        model_path = "sequential_sentence_classification/model.tar.gz"
        run_ssc(
            model_path=model_path,
            test_jsonl_file=ssc_abstract_input_file,
            output_json_file=ssc_abstract_output_file,
        )
    with open(ssc_abstract_output_file, "r") as f:
        ssc_abstract_preds = json.load(f)
    abstract_units = []
    for pred_obj in ssc_abstract_preds[args.arxiv_id]:
        abstract_units.append(
            azc.make_ssc_rhetoric_unit(
                pred_obj["sentence"], pred_obj["label"], pred_obj["prob"]
            )
        )
    with open(f"{abstract_output_path}/{args.arxiv_id}.json", "w") as out:
        serialized = [r.to_json() for r in abstract_units]
        json.dump(serialized, out, indent=2)

    # Extract rhetorical classes with heuristics
    azc = AZClassifier(spp_output_file, dataset="spp")

    # First, we get approximate figure and table locations (using their caption bboxes)
    media_units = []
    caption_bboxes = azc.paper.get_caption_bboxes()
    for caption_bbox in caption_bboxes:
        if any(x in caption_bbox.text.lower() for x in ["fig.", "figure"]):
            media_type = "figure"
        elif any(x in caption_bbox.text.lower() for x in ["tab.", "table"]):
            media_type = "table"
        else:
            media_type = ""
        media_units.append(
            MediaUnit(type=media_type, text=caption_bbox.text, bbox=caption_bbox.bbox)
        )
    with open(f"{captions_output_path}/{args.arxiv_id}.json", "w") as out:
        serialized = [m.to_json() for m in media_units]
        json.dump(serialized, out, indent=2)

    # Get all sections and their bboxes.
    section_units = []
    for section_block in azc.paper.get_section_bboxes():
        section_units.append(
            SectionUnit(text=section_block.text, bbox=section_block.bbox)
        )
    with open(f"{sections_output_path}/{args.arxiv_id}.json", "w") as out:
        serialized = [s.to_json() for s in section_units]
        json.dump(serialized, out, indent=2)

    # Get first sentences and their bboxes.
    with open(f"{first_sentence_output_path}/{args.arxiv_id}.json", "w") as out:
        first_sents = [s.to_json() for s in azc.paper.get_first_sentences()]
        for first_sent in first_sents:
            if first_sent["text"] in azc.paper.sent_sect_map:
                first_sent["section"] = azc.paper.sent_sect_map[first_sent["text"]]
            else:
                print(f"No section found for sentence: {first_sent['text']}")
        json.dump(first_sents, out, indent=2)

    # Get all sentences and their bounding boxes
    with open(f"{all_sentences_output_path}/{args.arxiv_id}.json", "w") as out:
        all_sents = [s.to_json() for s in azc.paper.sentences]
        for sent in all_sents:
            if sent["text"] in azc.paper.sent_sect_map:
                sent["section"] = azc.paper.sent_sect_map[sent["text"]]
            else:
                print(f"No section found for sentence: {sent['text']}")
        json.dump(all_sents, out, indent=2)

    ## Next, we get author statements (i.e., clauses including "we", "our", "this paper")
    # author_statements = azc.get_short_author_statements()
    # for author_statement in author_statements:
    #     short, full = author_statement[0], author_statement[1]
    #     bboxes = azc.paper.get_bboxes_for_span(short, full.text)
    #     rhetorical_units.append(
    #         RhetoricUnit(
    #             text=short,
    #             label="Author",
    #             bboxes=bboxes,
    #             section=None,
    #             prob=None,
    #             is_author_statement=True,
    #             is_in_expected_section=True,
    #         )
    #     )
    author_statements = azc.get_author_statements()
    for author_statement in author_statements:
        rhetorical_units.append(
            RhetoricUnit(
                text=author_statement.text,
                label="Author",
                bboxes=author_statement.bboxes,
                section=None,
                prob=None,
                is_author_statement=True,
                is_in_expected_section=True,
            )
        )

    ## Next, we detect each of the predefined facets
    rhetorical_units += azc.detect_contribution()
    rhetorical_units += azc.detect_novelty()
    rhetorical_units += azc.detect_objective()
    rhetorical_units += azc.detect_method()
    rhetorical_units += azc.detect_result()
    rhetorical_units += azc.detect_conclusion()
    rhetorical_units += azc.detect_future_work()

    ## Next, we filter the output of the facet classifier with a facet-sensitive threhsold
    with open(ssc_output_file, "r") as f:
        ssc_preds = json.load(f)

    top_preds = {}
    top_preds["Objective"] = get_top_pred_by_block(
        ssc_preds, azc.paper.sent_block_map, "Objective", max_overall=3
    )
    top_preds["Method"] = get_top_pred_by_block(
        ssc_preds, azc.paper.sent_block_map, "Method", max_per_block=2
    )
    top_preds["Result"] = get_top_pred_by_block(
        ssc_preds, azc.paper.sent_block_map, "Result"
    )

    for label, preds_for_label in top_preds.items():
        for pred_obj in preds_for_label:
            sentence = pred_obj["sentence"]
            label = pred_obj["label"]
            prob = pred_obj["prob"]
            if label in ["Objective", "Method", "Result"]:
                rhetorical_units.append(
                    azc.make_ssc_rhetoric_unit(sentence, label, prob)
                )

    ## Finally, we output all detected rhetorical units to a JSON file
    with open(f"{output_path}/{args.arxiv_id}.json", "w") as out:
        serialized = [r.to_json() for r in rhetorical_units]
        json.dump(serialized, out, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arxiv_id", type=str, required=True,
    )
    args = parser.parse_args()

    main(args)
