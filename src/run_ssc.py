import argparse
import json
import os
import sys

sys.path.append(".")

import numpy as np
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from sequential_sentence_classification.sequential_sentence_classification import (
    dataset_reader,
    model,
    predictor,
)
from tqdm import tqdm

from utils import SSC_OUTPUT_DIR, SSC_ABSTRACT_OUTPUT_DIR


def run_ssc(model_path: str, test_jsonl_file: str, output_json_file: str):
    text_field_embedder = {
        "token_embedders": {
            "bert": {
                "pretrained_model": "https://ai2-s2-research.s3-us-west-2.amazonaws.com/scibert/allennlp_files/scibert_scivocab_uncased.tar.gz"
            }
        }
    }
    token_indexers = {
        "bert": {
            "pretrained_model": "https://ai2-s2-research.s3-us-west-2.amazonaws.com/scibert/allennlp_files/scivocab_uncased.vocab"
        }
    }
    overrides = {
        "model": {"text_field_embedder": text_field_embedder},
        "dataset_reader": {
            "token_indexers": token_indexers,
            "use_sep": False,
            "predict": True,
        },
    }

    model_archive = load_archive(
        model_path, overrides=json.dumps(overrides), cuda_device=2
    )
    predictor = Predictor.from_archive(model_archive, "SeqClassificationPredictor")
    dataset_reader = predictor._dataset_reader
    model = predictor._model
    idx_to_label_map = model.vocab._index_to_token["labels"]

    test_jsons = []
    with open(test_jsonl_file) as f:
        for line in f:
            test_jsons.append(json.loads(line))
    print("{} test jsons loaded".format(len(test_jsons)))

    result = {}
    with open(output_json_file, "w") as out_f:
        for json_dict in tqdm(test_jsons, desc="Predicting..."):
            sentences = json_dict["sentences"]
            if not sentences:
                continue

            instances = dataset_reader.read_one_example(json_dict)
            if not isinstance(instances, list):
                instances = [instances]

            # with open(f"scripts/sentences/{json_dict['abstract_id']}.json", "r") as f:
            #     id_to_sent = json.load(f)
            # sent_to_id = {s: id for id, s in id_to_sent.items()}

            scores_list = []
            for instance in instances:
                prediction = predictor.predict_instance(instance)
                probs = prediction["action_probs"]
                scores_list.extend(probs)

            if len(sentences) != len(scores_list):
                print(
                    "The following paper had a mismatched number of sentences and sentence predictions."
                )
                print(json_dict.get("abstract_id", "") or json_dict.get("paper_id", ""))
                continue
            sentences_with_scores = list(zip(sentences, scores_list))
            sentences_with_scores = sorted(
                sentences_with_scores, key=lambda x: x[1], reverse=True
            )

            preds = []
            for sentence, scores in sentences_with_scores:
                pred_idx = np.argmax(scores)
                pred_label = idx_to_label_map[pred_idx].split("_")[0].capitalize()
                pred_prob = scores[pred_idx]
                preds.append(
                    {"sentence": sentence, "label": pred_label, "prob": pred_prob,}
                )
            if "abstract_id" in json_dict:
                result[json_dict["abstract_id"]] = preds
            elif "paper_id" in json_dict:
                result[json_dict["paper_id"]] = preds
            else:
                continue
        json.dump(result, out_f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_model", help="Path to the model to use for prediction", required=True
    )
    parser.add_argument(
        "--test_jsonl_file",
        help="Path to the jsonl file containing sentences to predict",
        required=True,
    )
    parser.add_argument(
        "--output_file", "-o", help="Name of file to output json predictions.",
    )
    args = parser.parse_args()

    if args.output_file:
        output_filename = args.output_file
    else:
        output_filename = (
            os.path.splitext(os.path.basename(args.test_jsonl_file))[0] + ".json"
        )
    output_json_file = os.path.join(SSC_OUTPUT_DIR, output_filename)
    run_ssc(args.path_to_model, args.test_jsonl_file, output_json_file)
