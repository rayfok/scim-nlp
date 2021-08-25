import os
import requests
import pandas as pd
from tqdm import tqdm

HEADERS = {"User-Agent": "Mozilla/5.0"}
ARXIV_PAPER_PDF_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"


def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    Ref to https://www.codementor.io/@aviaryan/downloading-files-from-urls-in-python-77q3bs0un
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get("content-type")
    if "text" in content_type.lower():
        return False
    if "html" in content_type.lower():
        return False
    return True


def try_download_pdf_to_target_dir(arxiv_link, arxiv_id, target_dir):

    download_path = f"{target_dir}/{arxiv_id}.pdf"

    if os.path.exists(download_path):
        return True
    try:
        r = requests.get(arxiv_link, headers=HEADERS)
        with open(download_path, "wb") as fp:
            fp.write(r.content)
        return True
    except requests.ConnectionError:
        return False


if __name__ == "__main__":

    target_dir = "../datasets/docbank/source_pdfs"
    os.makedirs(target_dir, exist_ok=True)

    df = pd.read_csv(
        "../datasets/docbank/pdf_scraping_table.csv",
        dtype={"arxiv_id": "str"},
        index_col=None,
    )

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df)):     
            if row['downloaded']: continue 

            arxiv_link = ARXIV_PAPER_PDF_TEMPLATE.format(arxiv_id=row["arxiv_id"])

            if is_downloadable(arxiv_link):

                if try_download_pdf_to_target_dir(arxiv_link, row["arxiv_id"], target_dir):
                    df.iloc[idx, -1] = True
                else:
                    print("ERROR 2: Fail to download the PDF")
            else: 
                print("ERROR 1: Invalid Download link")
    except KeyboardInterrupt:
        # Save the scarping progress
        df.to_csv('../datasets/docbank/pdf_scraping_table.csv', index=None)