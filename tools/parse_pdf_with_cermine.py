import sys
import os 
import subprocess

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1

sys.path.append("../src/")
from scienceparseplus.datasets.grotoap import CERMINELoader

CERMINE_IMP_NAME = "cermine-impl-1.13-jar-with-dependencies.jar"
CERMINE_PROG_NAME = "pl.edu.icm.cermine.PdfBxStructureExtractor"
CERMINE_LOADER = CERMINELoader()


def get_pdf_pages_and_sizes(filename: str):
    """Ref https://stackoverflow.com/a/47686921"""
    with open(filename, "rb") as fp:
        parser = PDFParser(fp)
        document = PDFDocument(parser)
        num_pages = resolve1(document.catalog["Pages"])["Count"]
        page_sizes = [
            (int(page.mediabox[2]), int(page.mediabox[3]))
            for page in PDFPage.create_pages(document)
        ]
        return num_pages, page_sizes


def _cvt_df_to_dict(row):
    return {
        "id": row["id"],
        "bbox": [row["x_1"], row["y_1"], row["x_2"], row["y_2"]],
        "type": row["category"],
        "text": row["text"],
        "parent_line_id": row["line_id"],
        "parent_block_id": row["block_id"],
    }


def convert_df_to_dict(df):
    return dict(
        blocks=df[df.is_block].apply(_cvt_df_to_dict, axis=1).tolist(),
        lines=df[df.is_line].apply(_cvt_df_to_dict, axis=1).tolist(),
        tokens=df[~df.is_line & ~df.is_block].apply(_cvt_df_to_dict, axis=1).tolist(),
    )


def run_and_parse_cermine(
    pdf_path, cermine_imp_name=CERMINE_IMP_NAME, cermine_prog_name=CERMINE_PROG_NAME
):

    foldername, _ = os.path.dirname(pdf_path), os.path.basename(pdf_path)
    xml_filename = pdf_path.replace(".pdf", ".cxml")

    try:
        subprocess.call(
            ["java", "-cp", cermine_imp_name, cermine_prog_name, "-path", foldername]
        )
        assert os.path.exists(xml_filename)
    except:
        print("Fail to run the CERMINE_ENGINE!")
        return False

    xml_data = CERMINE_LOADER.load_xml(xml_filename)
    num_pages, page_sizes = get_pdf_pages_and_sizes(pdf_path)

    output = [
        {
            "width": page_sizes[pid][0],  # width of this page,
            "height": page_sizes[pid][1],  # width of this page
            "index": pid,
            "layout": convert_df_to_dict(page_data.to_dataframe()),
        }
        for pid, page_data in xml_data.items()
    ]

    return output