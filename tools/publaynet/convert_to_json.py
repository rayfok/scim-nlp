import argparse
import sys

sys.path.append("../../src")
from scienceparseplus.datasets.publaynet import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="default", help="desc")
    parser.add_argument("--subset", type=str, default="default", help="desc")
    parser.add_argument("--save_path", type=str, default="default", help="desc")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--token_mask", action="store_true")
    args = parser.parse_args()

    dataset = PubLayNetDataset(args.base_path, args.subset)
    dataset.generate_token_data(args.save_path, debug=args.debug, save_token_mask=args.token_mask)