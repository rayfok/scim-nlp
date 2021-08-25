# Generalized HuggingFace Trainer 

## Overview 
The HuggingFace `Trainer` provides a nice interface for training the BERT-based models, yet its support for configurable and reusable experiments is really mediocre. Simply using the `Trainer` class makes it extremely complicated less robust for running large-scale experiments and ablation studies. As such, I implemented an trainer wrapper that can be used to 1) configure, 2) run, and 3) manage experiments *easily* and *robustly*. 

It has the following features: 
1. An inheritable template `BaseTrainer` class that streamlines the whole training process. 
2. Structured and reusable configuration based on `yacs`, see `configs.py`. 

## Usage

Consider the following cases: 
1. You have task A and task B: they have slightly different settings like different tokenizer and dataloaders. Within each task, there might be different datasets (X,Y). 
3. For each task, you would like to try with a variety of models (a,b,c)
4. For each model, there's also different possible variants like (u,v,w)

The trainer made it structured and easy for you to run the set of experiments.

1. For each task, you might need to create a new script (let's say `train_A.py`) based on the existing template `token_classification.py`. You need to: 
    1. overwrite the `load_dataset` and `load_model_tokenizer` based on the requirements of the task. 
    2. add additional arguments in `create_argparser` for handling the variants. See the example [here](https://github.com/allenai/scienceparseplus/blob/0e87d4abd5d5e23529096521d0b3211c5c9672e9/tools/train/base_trainer.py#L217).
    3. also remember to change the `ckpt_folder` name for different variants. See the example [here](https://github.com/allenai/scienceparseplus/blob/0e87d4abd5d5e23529096521d0b3211c5c9672e9/tools/train/base_trainer.py#L282).
    4. modify the `do_train` and `do_eval` when necessary. 
2. Create configuration for each datasets, like those shown in the `config/` folder. 
3. Run the following command: 
    ```bash
    python train_<A>.py --config dataset_<X>.yml --model_name <a> --<extra configs for variant u>
    ```

Some concrete examples: 
1. Running the token_classification task (A) using the docbank dataset (X), with the bert-base-uncased model (a), and freeze the base model weight, you can do: 
    ```bash
    python token_classification.py --config configs/docbank/base.yml --model_name bert-base-uncased --freeze_base_model
    ```
2. Running the token_classification_with_font_embedding task (B) using the publaynet dataset (Y), with the layoutlm-base-uncased model (b), and without any modification, you can do: 
    ```bash
    python token_classification_font_emb.py --config configs/pubolaynet/base.yml --model_name microsoft/layoutlm-base-uncased
    ```