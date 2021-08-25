from abc import ABC, abstractmethod
import configparser
import sys
import argparse
import os

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForTokenClassification,
    LayoutLMForTokenClassification,
)

sys.path.append("../../src")
from scienceparseplus.datasets import JSONTokenClassficationDataset
from scienceparseplus.modeling.tokenizer import JSONDatasetTokenizer

from config import get_default_cfg
from utils import compute_metrics_token, write_json_record


class BaseTrainer:
    """A template Trainer for different tasks"""

    task_name = ""

    def __init__(self):
        pass

    @staticmethod
    def get_latest_epoch_folder(ckpt_folder) -> str:
        latest_model_id = sorted(
            [
                int(subfolder.split("-")[1])
                for subfolder in os.listdir(ckpt_folder)
                if subfolder.startswith("checkpoint")
            ]
        )[-1]
        return f"{ckpt_folder}/checkpoint-{latest_model_id}"

    def create_argparser(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str)
        parser.add_argument("--model_name", type=str)
        parser.add_argument("--logging_folder", type=str, default="./exps")

        # Training Mode
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--debug", action="store_true")

        # Load Model Weights
        parser.add_argument("--load_existing_model", action="store_true")
        parser.add_argument("--existing_model_path", type=str, default="")

        # A technique learned from
        # https://github.com/facebookresearch/detectron2/blob/
        # 82ea854b189747d584640330de0943bec284da71/detectron2/engine/defaults.py#L107
        parser.add_argument(
            "opts",
            help="Modify config options by adding 'KEY VALUE' pairs at the end of the command.",
            default=None,
            nargs=argparse.REMAINDER,
        )
        return parser

    def check_arg_compatibility(self):

        print(f"Start checking args compatibility.")

        # Ensure there are existing weights when do_eval
        if self.args.do_eval:
            assert os.path.exists(self.ckpt_folder) or (
                self.args.existing_model_path != ""
            )

        print(f"Finshed checking args compatibility.")

    def parse_args_and_load_config(self, args):

        self.args = args

        print(f"GPU Status: {torch.cuda.is_available()}")

        print(f"Is in debug mode {self.args.debug}?")
        print(f"Do training? {self.args.do_train}")
        print(f"Do evaluation? {self.args.do_eval}")

        print(f"All arguments:")
        print(self.args)

        print("Loading configurations...")
        cfg = get_default_cfg()
        cfg.merge_from_file(self.args.config)
        cfg.merge_from_list(self.args.opts)
        self.cfg = cfg

        print("All configurations:")
        print(self.cfg)

        self.ckpt_folder = self.default_ckpt_folder

        self.check_arg_compatibility()

        print("Creating training arguments")
        self.create_training_arguments()
        print(self.training_args)


    @property
    def default_ckpt_folder(self) -> str:
        if self.args.debug:
            return f"{self.args.logging_folder}/debug/{self.cfg.DATASET.NAME}/{self.task_name}/{self.args.model_name.replace('/','-')}"
        else:
            return f"{self.args.logging_folder}/{self.cfg.DATASET.NAME}/{self.task_name}/{self.args.model_name.replace('/','-')}"

    def create_training_arguments(self):

        # fmt: off
        self.training_args = TrainingArguments(
            output_dir                  = self.ckpt_folder,
            num_train_epochs            = self.cfg.TRAIN.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size = self.cfg.TRAIN.BATCH_SIZE_PER_DEVICE,
            metric_for_best_model       = self.cfg.TRAIN.METRIC_FOR_BEST_MODEL,
            load_best_model_at_end      = self.cfg.TRAIN.LOAD_BEST_MODEL_AT_END,
            save_total_limit            = self.cfg.TRAIN.SAVE_TOTAL_LIMIT,

            learning_rate               = self.cfg.TRAIN.OPTIMIZER.LEARNING_RATE,
            weight_decay                = self.cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            label_smoothing_factor      = self.cfg.TRAIN.OPTIMIZER.LABEL_SMOOTHING_FACTOR,

            per_device_eval_batch_size  = self.cfg.TRAIN.EVAL.BATCH_SIZE_PER_DEVICE,
            logging_strategy            = self.cfg.TRAIN.LOGGING.STRATEGY,
            logging_steps               = self.cfg.TRAIN.LOGGING.STEPS,
            save_strategy               = self.cfg.TRAIN.SAVE.STRATEGY,
            save_steps                  = self.cfg.TRAIN.SAVE.STEPS,
            evaluation_strategy         = self.cfg.TRAIN.EVAL.STRATEGY,
            eval_steps                  = self.cfg.TRAIN.EVAL.STEPS
        )
        # fmt: on

    def determine_training_model_path(self) -> str:
        """Based on the inputs, determine the appropriate model_path 
        used for initializing the models. The logic: 

        If args.do_eval or args.load_existing_model: 
            you need to load from existing trained models 
            If existing_model_path is specified:
                load that exact path 
            Else:
                load the newest model from the checkpoint
        Else:
            just use the model_name and load the public pre-trained weights 
        """
        if self.args.do_eval or self.args.load_existing_model:
            if self.args.existing_model_path != "":
                model_path = self.args.existing_model_path
            else:
                model_path = self.get_latest_epoch_folder(self.ckpt_folder)
        else:
            model_path = self.args.model_name
        
        return model_path

    @abstractmethod
    def load_dataset(self):
        """It will choose the appropriate dataset loader given the config
        and will load the subsets given the training configs
        """

    @abstractmethod
    def load_model_tokenizer(self):
        """It will load the appropriate model and tokenizer based on the
        inputs. Will preload the model weights when necessary.
        """

    @abstractmethod
    def do_train(self):
        """"""

    @abstractmethod
    def before_train(self):
        """"""

    @abstractmethod
    def do_eval(self):
        """"""

    @abstractmethod
    def before_eval(self):
        """"""

    def run(self):

        self.load_dataset()

        self.load_model_tokenizer()

        if self.args.debug:
            print("Currently running in debug mode.")

        if self.args.do_train:
            self.before_train()
            self.do_train()

        if self.args.do_eval:
            self.before_eval()
            self.do_eval()


class BaseTokenClassificationTrainer(BaseTrainer):

    task_name = "token_classification"
    
    def create_argparser(self):

        parser = super().create_argparser()
        parser.add_argument("--freeze_base_model", action="store_true")

        return parser

    def load_dataset(self):

        subsets = []

        if self.args.do_eval:
            subsets.append("test")

        if self.args.do_train:
            subsets.extend(["train", "dev"])

        base_path = self.cfg.DATASET.BASE_PATH
        if self.args.debug:
            base_path += "-debug"

        self.datasets = {
            subset: JSONTokenClassficationDataset(
                base_path=base_path,
                subset=subset,
                filename=f"{subset}-token.json",
            )
            for subset in subsets
        }

        self.num_labels = len(self.datasets[list(self.datasets.keys())[0]].labels)

        print("Dataset has been loaded.")

    def load_model_tokenizer(self):

        tokenizer = JSONDatasetTokenizer(
            self.args.model_name, use_layout="layout" in self.args.model_name
        )

        model_path = self.determine_training_model_path()

        if "layoutlm" in self.args.model_name.lower():
            model = LayoutLMForTokenClassification.from_pretrained(
                model_path, num_labels=self.num_labels
            )
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                model_path, num_labels=self.num_labels
            )

        self.model = model
        self.tokenizer = tokenizer
    
    def before_train(self):
        
        if self.args.freeze_base_model:
            # This is tricky: 
            # So when setting freeze_base_model, you would like to load 
            # it from somewhere, but save the trained weights inanother 
            # folder. 
            # It should only change the ckpt folder before the training.
            # And when it loads the model weights, it should load from 
            # somewhere else. 

            self.ckpt_folder = self.ckpt_folder + '-freeze_base_model'
            
            print("Freeze base model weights:")
            for name, param in self.model.base_model.named_parameters():
                if self.args.debug:
                    print("Freezing: \t", name)
                param.requires_grad = False

    def do_train(self):

        trainer = Trainer(
            self.model,
            self.training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["dev"],
            data_collator=self.tokenizer.encode_plus,
            compute_metrics=lambda x: compute_metrics_token(
                x, self.datasets["dev"].labels
            ),
        )

        trainer.train()

        pred, label, eval_results = trainer.predict(self.datasets["dev"])
        write_json_record(f"{self.ckpt_folder}/dev-metrics.json", eval_results)

    def before_eval(self):
        pass

    def do_eval(self):

        trainer = Trainer(
            self.model,
            self.training_args,
            data_collator=self.tokenizer.encode_plus,
            compute_metrics=lambda x: compute_metrics_token(
                x, self.datasets["test"].labels
            ),
        )

        pred, label, eval_results = trainer.predict(self.datasets["test"])
        write_json_record(f"{self.ckpt_folder}/test-metrics.json", eval_results)