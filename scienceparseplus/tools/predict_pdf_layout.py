import json
import sys
import os
import argparse
from copy import deepcopy
from abc import abstractmethod
from collections import Counter

import layoutparser as lp
from tqdm import tqdm

sys.path.append("../src/")
from scienceparseplus.pdftools import *
from scienceparseplus.modeling.layoutlm import *
from scienceparseplus.modeling.visual_model import *
from scienceparseplus.modeling.pipeline_model import *

cvt_labels_from_coco_to_docbank = {
    "Title": "title",
    "Author": "author",
    "Abstract": "abstract",
    "Section": "section",
    "Paragraph": "paragraph",
    "ListItem": "list",
    "Footer": "footer",
    "BibItem": "reference",
    "Equation": "equation",
    "Figure": "figure",
    "Table": "table",
    "Caption": "caption",
}

docbank_orig_label_map = [
    "abstract",
    "author",
    "caption",
    "date",
    "equation",
    "figure",
    "footer",
    "list",
    "paragraph",
    "reference",
    "section",
    "table",
    "title",
]


def build_layout_detection_model():
    print(f"Currently using the pdfplumber PDF extractor")
    pdf_extractor = PDFExtractor("PDFPlumber")

    print("Loading models...")
    block_predictorA = ObjectDetectionBlockPredictor(
        config_path="models/publaynet/mask_rcnn_R_50_FPN_3x/config.yaml",
        model_path="models/publaynet/mask_rcnn_R_50_FPN_3x/model_final.pth",
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.55,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST",
            0.4,
        ],
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
    )  # pre-trained on publaynet

    block_predictorB = ObjectDetectionBlockPredictor(
        config_path="models/MFD/fast_rcnn_R_50_FPN_3x/config.yaml",
        model_path="models/MFD/fast_rcnn_R_50_FPN_3x/model_final.pth",
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.6,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST",
            0.2,
        ],
        label_map={1: "equation"},
    )  # For detecting equations, pre-trained on the MFD Dataset

    layoutlm_predictor = LayoutLMTokenPredictor(
        "models/DocBank_LayoutLM_base",
        max_seq_length=512,
        tokenizer_path="models/DocBank_LayoutLM_base",
        label_map=docbank_orig_label_map,
    )

    predictor = V4ModelPredictor(
        pdf_extractor=pdf_extractor,
        block_predictor=[block_predictorA, block_predictorB],
        block_classifier=layoutlm_predictor,
        model_name="v4-layoutlm",
        visualizer=None,
    )

    return predictor

def cvt_bbox_to_xywh(bbox):
    return bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help="desc", nargs="+")
    parser.add_argument("--save_folder", type=str, default=None, help="desc")
    parser.add_argument("--pawls_format", action="store_true")
    parser.add_argument("--pawls_anno_name", type=str, default="anno.json", help="desc")

    args = parser.parse_args()

    predictor = build_layout_detection_model()
    save_path = args.save_folder
    pbar = tqdm(args.filename)

    if not args.pawls_format:
        for filename in pbar:
            if save_path is None:
                save_path = os.path.dirname(filename)

            pbar.set_description(f"Parsing PDF File {filename}")
            results = predictor.predict_pdf(filename, save_path=save_path)
            pbar.set_description("Successfully processed the target PDF")

    else:
        print("Saving the parsed PDF file layouts in PAWLS pre-annotate format.")

        all_pdf_data = {}
        for filename in pbar:
            pbar.set_description(f"Parsing PDF File {filename}")
            results = predictor.predict_pdf(filename, save_path=None)

            all_pdf_data[os.path.basename(filename)] = [
                {
                    "page": {"height":ele['height'], "width":ele['width'], "index":ele['index']},
                    "blocks": [
                        [*cvt_bbox_to_xywh(bundle["bbox"]), bundle["type"].title()]
                        for bundle in ele["layout"]["bundles"]
                    ]
                }
                for ele in results
            ]
        
        if save_path is None:
            save_path = "."

        with open(os.path.join(save_path, args.pawls_anno_name), "w") as fp:
            json.dump(all_pdf_data, fp)