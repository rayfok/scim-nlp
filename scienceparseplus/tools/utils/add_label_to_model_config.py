"""Used for injecting the label configs into the config file.

    Usage:

        python add_label_to_model_config --config_file "config.json" --labels A B C
"""

import json
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file", type=str, default="default", help="The config filename"
)
parser.add_argument(
    "--labels",
    type=str,
    nargs="+",
    help="An ordered list of labels used for modification",
)
parser.add_argument(
    "--force", action="store_true", help="Whether to ask confirmation before save"
)


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config_file, "r") as fp:
        config = json.load(fp)

    assert len(config["id2label"]) == len(
        args.labels
    ), "The number of labels should be the same"

    config["id2label"] = {str(idx): label for idx, label in enumerate(args.labels)}
    config["label2id"] = {label: str(idx) for idx, label in enumerate(args.labels)}

    pprint(config)
    print("===" * 50)

    if args.force:
        with open(args.config_file, "w") as fp:
            json.dump(config, fp)
        print(f"File saved to {args.config_file}")
        exit()

    x = input("Correct? [Y/N]")
    if x.strip().lower() == "Y":
        print()
        with open(args.config_file, "w") as fp:
            json.dump(config, fp)
        print(f"File saved to {args.config_file}")
    else:
        print(f"\nExited without save")