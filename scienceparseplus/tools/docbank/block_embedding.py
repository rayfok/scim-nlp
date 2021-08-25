import os
import argparse
import configparser
import torch
import sys

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    LayoutLMTokenizerFast,
    TrainingArguments,
    Trainer,
)

sys.path.append("../../src")
from scienceparseplus.datamodel import *
from scienceparseplus.datasets.docbank import *
from scienceparseplus.modeling.layoutlm_block import *
from scienceparseplus.modeling.bert_block import *

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./block_embedding.ini")
parser.add_argument("--model_name", type=str, default="bert-base-uncased")

# Training Mode
parser.add_argument("--eval", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--use_full", action="store_true")
parser.add_argument("--use_docbank", action="store_true")
parser.add_argument("--add_class_weight", action="store_true")

parser.add_argument("--calibration", action="store_true")


DEFAULT_SPECIAL_TOKEN_BOXES = {
    "[PAD]": torch.tensor([0, 0, 0, 0], dtype=torch.long),
    "[CLS]": torch.tensor([0, 0, 0, 0], dtype=torch.long),
    "[SEP]": torch.tensor([1000, 1000, 1000, 1000], dtype=torch.long),
}


class DocBankBlockEmbeddingTokenier:
    def __init__(
        self,
        tokenizer,
        use_layout=True,
        special_token_boxes=DEFAULT_SPECIAL_TOKEN_BOXES,
    ):

        self.tokenizer = tokenizer
        self.use_layout = use_layout

        token_to_id_map = {
            token: idx
            for idx, token in zip(
                tokenizer.all_special_ids, tokenizer.all_special_tokens
            )
        }

        self.special_token_boxes = {
            token_to_id_map[token]: box for token, box in special_token_boxes.items()
        }

    def encode_plus(self, features):

        first = features[0]

        all_words = [f["words"] for f in features]
        encoded_inputs = self.tokenizer.batch_encode_plus(
            all_words,
            max_length=512,
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        bz, sequence_length = encoded_inputs["input_ids"].shape
        encoded_bbox = torch.zeros(bz, sequence_length, 4, dtype=torch.long)
        encoded_labels = torch.zeros(bz, sequence_length, dtype=torch.long)
        encoded_blocks = torch.zeros(bz, sequence_length, dtype=torch.long)

        offset_mapping = encoded_inputs.pop("offset_mapping")
        special_tokens_mask = encoded_inputs.pop("special_tokens_mask")

        for idx, (offset, token_mask, f, input_ids) in enumerate(
            zip(
                offset_mapping,
                special_tokens_mask,
                features,
                encoded_inputs["input_ids"],
            )
        ):
            token_mask = token_mask.bool()
            if False in token_mask:
                mapping = (offset[~token_mask, 0] == 0).cumsum(dim=0) - 1
                encoded_bbox[idx, ~token_mask, :] = torch.tensor(
                    [f["bbox"][idx] for idx in mapping], dtype=torch.long
                )
                encoded_labels[idx, ~token_mask] = torch.tensor(
                    [f["labels"][idx] for idx in mapping], dtype=torch.long
                )
                encoded_blocks[idx, ~token_mask] = torch.tensor(
                    [f["block_ids"][idx] for idx in mapping], dtype=torch.long
                )

            special_token = 102
            # hard coded for current mode - only SEP token has special bbox

            special_token_idx = torch.where(input_ids == special_token)[0]
            encoded_bbox[idx, special_token_idx, :] = self.special_token_boxes[
                special_token
            ]

        encoded_inputs["labels"] = encoded_labels
        encoded_inputs["block_ids"] = encoded_blocks + 1
        # Because there are some negative blocks
        # 0 -> token not in any blocks
        # >=1 -> regular
        if self.use_layout:
            encoded_inputs["bbox"] = encoded_bbox

        if "class_weight" in first:
            encoded_inputs["class_weight"] = first["class_weight"]

        return encoded_inputs


def load_model_and_tokenizer(args, num_labels, config):

    if "layoutlm" in args.model_name.lower():
        tokenizer = LayoutLMTokenizerFast.from_pretrained(args.model_name)
        tokenizer = DocBankBlockEmbeddingTokenier(tokenizer, use_layout=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer = DocBankBlockEmbeddingTokenier(tokenizer, use_layout=False)

    if not args.eval:
        if "layoutlm" in args.model_name.lower():
            if args.use_docbank:
                assert "large" in args.model_name.lower()
                model_szie = "base" if "base" in args.model_name else "large"
                model = LayoutLMForTokenClassification.from_pretrained(
                    "_".join([model_config["dockbank_model_prefix"], model_szie])
                )
                # Manually fix the cls layer
                last_layer = list(model.children())[-1]
                last_layer.reset_parameters()
                last_layer.out_features = num_labels

            else:
                model = LayoutLMForTokenClassification.from_pretrained(
                    args.model_name, num_labels=num_labels
                )
        else:
            try:
                model = BertForTokenClassification.from_pretrained(
                    args.model_name, num_labels=num_labels
                )
            except:
                print("Model not compatible with BERT, use AutoModel to initialize.")
                model = AutoModelForTokenClassification.from_pretrained(
                    args.model_name, num_labels=num_labels
                )
    else:
        model_path = get_latest_epoch_folder(get_model_folder(model_config, args)+'-long')
        if "layoutlm" in args.model_name.lower():
            model = LayoutLMForTokenClassification.from_pretrained(model_path)
        else:
            try:
                model = BertForTokenClassification.from_pretrained(model_path)
            except:
                print("Model not compatible with BERT, use AutoModel to initialize.")
                model = AutoModelForTokenClassification.from_pretrained(
                    args.model_name, num_labels=num_labels
                )
    return tokenizer, model


if __name__ == "__main__":
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    print(f"Use all data? {args.use_full}")
    print(f"In debug mode? {args.debug}")

    select_n, select_ratio = None, None
    if not args.use_full:
        select_ratio = 0.5
    if args.debug:
        select_n = 5

    if args.eval:
        if args.calibration:
            subsets = ["dev", "test"]
        else:
            subsets = ["test"]
    else:
        subsets = ["dev", "train"]

    datasets = {
        subset: DocBankBlockEmbeddingDataset(
            base_path=config["Data"]["doc_bank_base_path"],
            subset=subset,
            select_n=select_n,
            select_ratio=select_ratio,
            filename=f"{subset}-token.json",
            add_class_weight=args.add_class_weight,
        )
        for subset in subsets
    }

    print("Dataset has been loaded.")

    num_labels = len(datasets[list(datasets.keys())[0]].labels)
    model_config = config["Model Config"]
    model_folder = get_model_folder(model_config, args) + '-long'
    if args.add_class_weight:
        model_folder = model_config + "-class-weighted"

    tokenizer, model = load_model_and_tokenizer(args, num_labels, model_config)

    print("Tokenizer and model has been loaded.")
    print(f"GPU Status: {torch.cuda.is_available()}")

    training_args = TrainingArguments(
        model_folder,
        evaluation_strategy="steps",
        logging_steps=model_config.get("eval_steps", 500),
        eval_steps=model_config.get("eval_steps", 2000),
        learning_rate=model_config.get("learning_rate", 2e-5),
        per_device_train_batch_size=int(model_config.get("batch_size", 20)),
        per_device_eval_batch_size=int(model_config.get("batch_size", 20)),
        num_train_epochs=int(model_config.get("num_train_epochs", 3)),
        weight_decay=model_config.get("weight_decay", 0.01),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    if not args.eval:
        print("Start training.")

        trainer = Trainer(
            model,
            training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["dev"],
            data_collator=tokenizer.encode_plus,
            compute_metrics=lambda x: compute_metrics_token(x, datasets["dev"].labels),
        )

        trainer.train()
    else:
        trainer = Trainer(
            model,
            training_args,
            data_collator=tokenizer.encode_plus,
            compute_metrics=lambda x: compute_metrics_token(x, datasets["test"].labels),
        )
        pred, label, eval_results = trainer.predict(datasets["test"])
        if not args.calibration:
            write_json_record(f"{model_folder}/eval-test.json", eval_results)
        else:
            dev_pred, dev_label, _ = trainer.predict(datasets["dev"])
            dev_label = dev_label.reshape(-1, 1)
            dev_pred = dev_pred.reshape(-1, dev_pred.shape[-1])
            pred = pred.reshape(-1, pred.shape[-1])
            label = label.reshape(-1, 1)

            report = {}
            for method in ["sigmoid", "isotonic"]:
                calibrator = create_calibrator(
                    dev_pred,
                    dev_label,
                    model_folder,
                    downsample=True,
                    method=method,
                    cv=3,
                    n_jobs=-1,
                )
                calibrated_pred = calibrator.predict(pred)
                report[f"calibration-{method}"] = report_scores(
                    label, calibrated_pred.reshape(-1, 1), datasets["test"].labels
                )

            write_json_record(f"{model_folder}/eval-test-calibrated.json", report)