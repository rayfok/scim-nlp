from typing import List, Union, Dict, Any, Tuple
import pandas as pd
import numpy as np
import os
import json

from sklearn.metrics import precision_recall_fscore_support

def write_json_record(filepath, data):

    with open(filepath, "w") as fp:
        json.dump(data, fp)


def get_ckpt_folder(task_name, dataset_name, args) -> str:
    return f"{args.logging_folder}/{dataset_name}/{task_name}/{args.model_name.replace('/','-')}"


def get_latest_epoch_folder(training_folder) -> str:
    latest_model_id = sorted(
        [
            int(subfolder.split("-")[1])
            for subfolder in os.listdir(training_folder)
            if subfolder.startswith("checkpoint")
        ]
    )[-1]
    return f"{training_folder}/checkpoint-{latest_model_id}"


def compute_metrics_token(eval_pred, labels_map=None):

    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    labels = labels.reshape(-1, 1)
    predicted_labels = predictions.argmax(axis=-1).reshape(-1, 1)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels, predicted_labels, average="micro"
    )

    keys = np.unique(labels)
    scores = precision_recall_fscore_support(
        labels, predicted_labels, labels=list(keys), zero_division=0
    )
    df = pd.DataFrame(
        scores, columns=keys, index=["precision", "recall", "f-score", "support"]
    )
    if labels_map is not None:
        if isinstance(labels_map, list):
            labels_map = {str(idx): label for idx, label in enumerate(labels_map)}
        df.columns = [labels_map.get(str(ele), ele) for ele in df.columns]

    return {
        "accuracy": (predicted_labels == labels).mean(),
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "detailed": df.to_dict(),
    }