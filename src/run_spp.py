import requests
import json
import os

# SPP_URL = "http://pdf-layout-detection-service-dev.us-west-2.elasticbeanstalk.com/detect/"
SPP_URL = "http://127.0.0.1:8080/detect"


def get_parsed_arxiv_pdf(arxiv_id: str):
    return get_parsed_url_pdf(f"https://arxiv.org/pdf/{arxiv_id}.pdf")


def get_parsed_url_pdf(url: str):
    r = requests.get(SPP_URL, params={"pdf_url": url})
    if r.ok:
        layout = r.json()["layout"]
    else:
        print(f"Failed with status code: {r.status_code}")
        layout = None
    return layout


def get_parsed_local_pdf(pdf_path: str):
    with open(pdf_path, "rb") as f:
        files = {"pdf_file": (f.name, f, "multipart/form-data")}
        r = requests.post(SPP_URL, files=files)
        if r.ok:
            layout = r.json()["layout"]
        else:
            print(f"Failed with status code: {r.status_code}")
            layout = None
    return layout


def parse_folder_of_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.scandir(input_dir):
        layout = get_parsed_local_pdf(file.path)
        with open(f"{output_dir}/{os.path.splitext(file.name)[0]}.json", "w") as out:
            json.dump(layout, out, indent=2)


if __name__ == "__main__":
    INPUT_DIR = "data/spp-pdf"
    OUTPUT_DIR = "data/spp-parsed"
    parse_folder_of_pdfs(INPUT_DIR, OUTPUT_DIR)
