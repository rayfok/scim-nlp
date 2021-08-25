from typing import List, Union, Dict, Any, Tuple
import pandas as pd
import numpy as np
import os
import json
from joblib import dump, load

from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV


def report_scores(labels, predicted_labels, labels_map):

    assert labels.shape == predicted_labels.shape

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


def compute_metrics(eval_pred, labels_map=None):

    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)

    return report_scores(labels, predicted_labels, labels_map)


def compute_metrics_with_token_length(eval_pred, token_length, labels_map=None):

    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels,
        predicted_labels,
        average="micro",
        sample_weight=token_length,
    )
    keys = np.unique(labels)
    scores = precision_recall_fscore_support(
        labels,
        predicted_labels,
        labels=list(keys),
        zero_division=0,
        sample_weight=token_length,
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


def get_latest_epoch_folder(training_folder) -> str:
    latest_model_id = sorted(
        [
            int(subfolder.split("-")[1])
            for subfolder in os.listdir(training_folder)
            if subfolder.startswith("checkpoint")
        ]
    )[-1]
    return f"{training_folder}/checkpoint-{latest_model_id}"


def get_model_folder(model_config, args) -> str:
    return f"{model_config['checkpoint_base_path']}/{args.model_name.replace('/','-')}"


def write_json_record(filepath, data):

    with open(filepath, "w") as fp:
        json.dump(data, fp)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n=None):
        self.n = n
        self.classes_ = np.arange(self.n)

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

    def predict_proba(self, X, y=None):
        return X


def create_calibrator(
    predictions,
    labels,
    model_path=None,
    downsample=True,
    method="sigmoid",
    cv=None,
    n_jobs=None,
):

    base_clf = IdentityTransformer(n=predictions.shape[-1])
    calibrated_clf = CalibratedClassifierCV(
        base_estimator=base_clf, method=method, cv=cv, n_jobs=n_jobs
    )

    if downsample:
        calibrated_clf.fit(predictions[::20], labels[::20])
    else:
        calibrated_clf.fit(predictions, labels)

    if model_path is not None:
        dump(calibrated_clf, os.path.join(model_path, f"calibrated-{method}.joblib"))
    return calibrated_clf


def freeze_model_params_except(model, unfreeze_layer_names:List=None):

    if unfreeze_layer_names is None:
        unfreeze_layer_names = []
    
    for param in model.base_model.parameters():
        param.requires_grad = False

    for name, layer in model.base_model.embeddings.named_children():
        if name in unfreeze_layer_names:
            for param in layer.parameters():
                param.requires_grad = True
    
    return model