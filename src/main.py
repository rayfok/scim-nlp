import argparse
import json
import os
from pprint import pprint

from detect_rhetorical_classes import AZClassifier
from paper import RhetoricUnit
from run_spp import get_parsed_arxiv_pdf
from run_ssc import run_ssc
from utils import (SPP_OUTPUT_DIR, SSC_INPUT_DIR, SSC_OUTPUT_DIR,
                   get_top_k_ssc_pred, make_spp_output_to_ssc_input)


def main(args):
    output_dir = "output/detected"
    if os.path.exists(f"{output_dir}/{args.arxiv_id}.json"):
        return

    rhetorical_units = []  # list to hold outputs

    # Run scienceparseplus to detect all tokens and bounding boxes
    spp_output_file = f"{SPP_OUTPUT_DIR}/{args.arxiv_id}.json"
    if not os.path.exists(spp_output_file):
        layout = get_parsed_arxiv_pdf(args.arxiv_id)
        with open(spp_output_file, "w") as out:
            json.dump(layout, out, indent=2)

    # Convert scienceparseplus output into a data format for sequential-sentence-classification
    ssc_input_file = f"{SSC_INPUT_DIR}/{args.arxiv_id}.jsonl"
    if not os.path.exists(ssc_input_file):
        make_spp_output_to_ssc_input(args.arxiv_id)

    # Run sequential-sentence-classification to classify each sentence into one of
    # five categories: Background, Objective, Method, Result, Other.
    ssc_output_file = f"{SSC_OUTPUT_DIR}/{args.arxiv_id}.json"
    if not os.path.exists(ssc_output_file):
        model_path = "sequential_sentence_classification/model.tar.gz"
        run_ssc(
            model_path=model_path,
            test_jsonl_file=ssc_input_file,
            output_json_file=ssc_output_file,
        )

    # Extract rhetorical classes with heuristics
    azc = AZClassifier(spp_output_file, dataset="spp")

    author_statements = azc.get_short_author_statements()
    for author_statement in author_statements:
        short, full = author_statement[0], author_statement[1]
        bboxes = azc.paper.get_bboxes_for_span(short, full)
        rhetorical_units.append(
            RhetoricUnit(
                text=short,
                label="Author",
                bboxes=bboxes,
                section=None,
                prob=None,
                is_author_statement=True,
                is_in_expected_section=True,
            )
        )

    rhetorical_units += azc.detect_contribution()
    rhetorical_units += azc.detect_novelty()
    rhetorical_units += azc.detect_objective()
    rhetorical_units += azc.detect_result()
    rhetorical_units += azc.detect_conclusion()
    rhetorical_units += azc.detect_future_work()

    # Print the top k sentences (based on model probs) for each category
    with open(ssc_output_file, "r") as f:
        ssc_preds = json.load(f)
    top_k_pred = get_top_k_ssc_pred(ssc_preds, k=5)
    # pprint(top_k_pred)
    for label, preds_for_label in top_k_pred[args.arxiv_id].items():
        for pred_obj in preds_for_label:
            sentence = pred_obj["sentence"]
            label = pred_obj["label"]
            prob = pred_obj["prob"]
            if label in ["Method", "Objective", "Result"]:
                rhetorical_units.append(
                    azc.make_ssc_rhetoric_unit(sentence, label, prob)
                )

    # for unit in rhetorical_units:
    #     print(json.dumps(unit.to_json(), indent=2))

    # Dump all detected rhetorical units to JSON file
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{args.arxiv_id}.json", "w") as out:
        serialized = [r.to_json() for r in rhetorical_units]
        json.dump(serialized, out, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arxiv_id", type=str, required=True,
    )
    args = parser.parse_args()

    main(args)
