import os
import re
import argparse
import sys 
import json 

from tqdm import tqdm
import pandas as pd

sys.path.append("../../src")
from scienceparseplus.datasets.publaynet import cvt_token_level_gp, NpEncoder


FONT_CLEAN_PATTERN = re.compile(r"(\S{6})\+")
CATEGORYID2STR = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure", 6: "empty"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="default", help="desc")
    parser.add_argument("--subset", type=str, default="train", help="desc")
    parser.add_argument("--save_folder_name", type=str, default="grouped", help="desc")

    parser.add_argument("--use_pdf_token_ordering", action="store_true")
    args = parser.parse_args()

    token_level_data = []
    segment_level_data = []
    files = []

    csv_folder = f"{args.base_path}/token-layout/{args.subset}"
    save_folder = f"{args.base_path}/{args.save_folder_name}/"
    if not os.path.exists(save_folder):
        print(f"Creating the saving folder {save_folder}")
        os.makedirs(save_folder)
    
    print(f"Saving to {save_folder}")

    for filename in tqdm(os.listdir(csv_folder)):

        try:
            df = pd.read_csv(f"{csv_folder}/{filename}")
        except:
            print(f"dataframe loading errors for {filename}")
            continue

        if args.use_pdf_token_ordering:
            df = df.sort_values(by="id")

        df["font"] = df["font"].str.replace(FONT_CLEAN_PATTERN, "")
        df["parent"] = df["parent"].fillna(-1).astype("int")

        row_item = {
            "words": df["text"].tolist(),
            "bbox": df.apply(
                lambda row: (row["x_1"], row["y_1"], row["x_2"], row["y_2"]), axis=1
            ).tolist(),
            "labels": df["category"].tolist(),
            "block_ids": df["parent"].tolist(),
            "fonts": df["font"].tolist(),
        }
        token_level_data.append(row_item)

        segment_data = (
            df[df["parent"] != -1].groupby("parent").apply(cvt_token_level_gp).tolist()
        )
        segment_level_data.extend(segment_data)

        files.append(filename)

    token_level_data = {
        "data": token_level_data,
        "labels": CATEGORYID2STR,
        "files": files,
    }
    segment_level_data = {
        "data": segment_level_data,
        "labels": CATEGORYID2STR,
        "files": files,
    }

    with open(f"{save_folder}/{args.subset}-token.json", "w") as fp:
        json.dump(token_level_data, fp, cls=NpEncoder)

    with open(f"{save_folder}/{args.subset}-segment.json", "w") as fp:
        json.dump(segment_level_data, fp, cls=NpEncoder)