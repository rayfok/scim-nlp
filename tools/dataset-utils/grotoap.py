import sys
import os
import argparse
from glob import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from pdf2image import convert_from_path

sys.path.append("../../src")
from scienceparseplus.datasets.grotoap import *

from coco import *

np.random.seed(452)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def load_json(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def write_json(data, filename):
    with open(filename, "w") as fp:
        json.dump(data, fp)


def create_json_record_for_files(all_files, id2label=None):

    token_level_data = []
    files = []

    pbar = tqdm(all_files)
    for filename in pbar:
        _filename = os.path.basename(filename).rstrip(".csv")
        pbar.set_description(_filename)

        df = pd.read_csv(filename)
        df = df.dropna(axis=0, subset=["text"])
        df = df[~df.is_block & ~df.is_line & ~df.text.str.isspace()]

        if len(df) == 0:
            continue

        df["block_id"] = df["block_id"].fillna(-1).astype("int")
        df["line_id"] = df["line_id"].fillna(-1).astype("int")

        row_item = {
            "words": df["text"].tolist(),
            "bbox": df.apply(
                lambda row: (row["x_1"], row["y_1"], row["x_2"], row["y_2"]), axis=1
            ).tolist(),
            "labels": df["category"].tolist(),
            "block_ids": df["block_id"].tolist(),
            "line_ids": df["line_id"].tolist(),
        }
        token_level_data.append(row_item)
        files.append(_filename)

    label_counts = {}
    for ele in token_level_data:
        for label in ele["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    if id2label is None:
        label2id = {label: idx for idx, label in enumerate(label_counts)}
        id2label = {idx: label for idx, label in enumerate(label_counts)}
    else:
        label2id = {label: idx for idx, label in id2label.items()}

    for ele in token_level_data:
        ele["labels"] = [label2id[l] for l in ele["labels"]]

    return {
        "data": token_level_data,
        "labels": id2label,
        "files": files,
        "label_counts": label_counts,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--store_token_table", action="store_true")
    parser.add_argument("--store_json", action="store_true")
    parser.add_argument("--store_coco", action="store_true")
    parser.add_argument("--mono_level", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=45, help="desc")

    parser.add_argument(
        "--raw_file", type=str, default="../datasets/grotoap2/grotoap2/", help="desc"
    )
    parser.add_argument(
        "--token_table_path",
        type=str,
        default="../datasets/grotoap2/token-table",
        help="desc",
    )
    parser.add_argument(
        "--json_path", type=str, default="../datasets/grotoap2/grouped", help="desc"
    )
    parser.add_argument(
        "--coco_path", type=str, default="../datasets/grotoap2/coco", help="desc"
    )

    args = parser.parse_args()

    if args.store_token_table:
        # Convert from source to page token table
        dataset = GrotoapDatasetWithFont(args.raw_file)
        if args.debug:
            dataset.all_xml_files = dataset.all_xml_files[:400]
        dataset.convert_to_page_token_table(args.token_table_path, n_jobs=args.n_jobs)

    if args.store_json:
        # Convert to JSON tables
        target_folder = args.json_path
        if args.debug:
            target_folder += "-debug"
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print("Creating the save Folder")

        all_files = glob(f"{args.token_table_path}/*.csv")
        if args.debug:
            num_samples = 400
            print(f"In Debug Mode: Sample only {num_samples} pages.")
        else:
            num_samples = len(all_files)
        indices = np.random.permutation(num_samples)

        train_samples, val_samples = int(num_samples * TRAIN_RATIO), int(
            num_samples * VAL_RATIO
        )

        file_subsets = {
            "train": [all_files[idx] for idx in indices[:train_samples]],
            "val": [
                all_files[idx]
                for idx in indices[train_samples : train_samples + val_samples]
            ],
            "test": [all_files[idx] for idx in indices[train_samples + val_samples :]],
        }

        used_files = {}

        id2label = None
        for subset in ["train", "val", "test"]:
            print(f"Working on {subset}")
            results = create_json_record_for_files(file_subsets[subset], id2label)

            save_path = (
                f"{target_folder}/{'dev' if subset == 'val' else subset}-token.json"
            )
            print(f"Storing to {save_path}")
            write_json(results, save_path)

            id2label = results["labels"]
            used_files[subset] = results["files"]

        write_json(used_files, f"{target_folder}/train-test-split.json")

    if args.store_coco:
        train_test_split = load_json(f"{args.json_path}/train-test-split.json")
        all_labels = load_json(f"{args.json_path}/labels.json")

        coco_objects = defaultdict(dict)
        
        for level in ["line", "block"]:
            for subset in ["train", "val", "test"]:
                save_path=args.coco_path
                annotation_name=f"{level}-{subset}" if not args.debug else f"debug-{level}-{subset}"

                if args.mono_level:
                    all_labels = {0: level}
                    annotation_name = annotation_name + "-mono"

                coco_objects[level][subset] = COCOBuilder(
                    save_path=save_path,
                    annotation_name=annotation_name,
                    categories=all_labels,
                )

        savename2subset = {
            savename: subset
            for subset, savenames in train_test_split.items()
            for savename in savenames
        }
        fileid_pages = defaultdict(list)
        
        all_file_ids = list(savename2subset.keys())
        if args.debug:
            all_file_ids = all_file_ids[:100]

        for ele in all_file_ids:
            a, b, c = ele.split("-")
            fileid_pages[(a, b)].append(c)

        pbar = tqdm(fileid_pages.items(), total=len(fileid_pages))
        for fileid, pids in pbar:
            pbar.set_description(f"Working on {'-'.join(fileid)}")
            pdf_filename = f"{args.raw_file}/dataset/{'/'.join(fileid)}.pdf"
            
            try:
                pdf_images = convert_from_path(pdf_filename, dpi=72)
            except KeyboardInterrupt:
                exit()
            except:
                print("Fail to load the PDF file")
                continue 
            
            if not args.debug and len(pids) != len(pdf_images):
                print(len(pids), len(pdf_images))
                print(pids)
                print("Incompatible PDF and Annotation")
                continue
            
            fileid = "-".join(fileid)

            for pid in pids:
                csv_filename = f"{args.token_table_path}/{fileid}-{pid}.csv"
                df = pd.read_csv(csv_filename)
                if len(df) == 0: 
                    print(f"Skip for empty page {fileid}-{pid}")
                    continue
                
                page_image = pdf_images[int(pid)]
                savename = f"{fileid}-{pid}"
                subset = savename2subset[savename]

                for level in ["line", "block"]:
                    
                    if args.mono_level:
                        level_anno = df[df[f"is_{level}"]][["id", "x_1", "y_1", "x_2", "y_2"]].copy()
                        level_anno["category"] = level
                    else:
                        page_annotations=df[df[f"is_{level}"]][
                            ["id", "x_1", "y_1", "x_2", "y_2", "category"]
                        ].copy(),

                    coco_objects[level][subset].add_annotation(
                        page_name=savename,
                        page_image=page_image,
                        page_annotations=level_anno,
                    )
        
        for level in ["line", "block"]:
            for subset in ["train", "val", "test"]:
                coco_objects[level][subset].export()