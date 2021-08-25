import sys
import os
import argparse
from glob import glob
import json
from collections import defaultdict
import subprocess
import tempfile
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

CERMINE_IMP_PATH = "cermine-impl-1.13-jar-with-dependencies.jar"
CERMINE_PROG_NAME = "pl.edu.icm.cermine.PdfBxStructureExtractor"

def process_single_pdf(filepath):

    target_xmlpath = filepath.replace(".pdf", ".cxml")

    source_dir, pdf_name = os.path.dirname(filepath), os.path.basename(filepath)

    with tempfile.TemporaryDirectory() as tempdir:

        tmp_filepath = os.path.join(tempdir, pdf_name)
        tmp_xmlpath = tmp_filepath.replace(".pdf", ".cxml")
        os.symlink(filepath, tmp_filepath)

        subprocess.call(
            ["java", "-cp", cermine_imp_path, cermine_prog_name, "-path", tempdir]
        )

        if os.path.exists(tmp_xmlpath):
            shutil.move(tmp_xmlpath, target_xmlpath)
            return None
        else:
            return filepath

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir", type=str, default="../datasets/grotoap2/grotoap2/", help="desc"
    )
    parser.add_argument("--n_jobs", type=int, default=2, help="desc")
    parser.add_argument("--issue_table", type=str, default="problematic_pdfs.csv")
    parser.add_argument("--cermine_imp_path", type=str, default=CERMINE_IMP_PATH)
    parser.add_argument("--cermine_prog_name", type=str, default=CERMINE_PROG_NAME)
    args = parser.parse_args()

    cermine_imp_path = args.cermine_imp_path
    cermine_prog_name = args.cermine_prog_name

    all_pdfs_to_process = glob(f"{args.target_dir}/*.pdf")
    pdfs_need_to_process = [ele for ele in tqdm(all_pdfs_to_process) if not os.path.exists(ele.replace('.pdf', '.cxml'))]
    print(f"Total number of pdfs to process {len(pdfs_need_to_process)}")

    results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_single_pdf)(pdf)
            for pdf in tqdm(pdfs_need_to_process)
        )
    pd.Series([ele for ele in results if ele is not None]).to_csv(args.issue_table)
