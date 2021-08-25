from scienceparseplus.pdftools import *
import layoutparser as lp

TEST_PDF_FILEPATH = "tests/fixtures/test.pdf"
TWO_COLUMN_PDF_FILEPATH = "tests/fixtures/two-column.pdf"


def test_grobid(grobid_host="", enabled=False):
    if enabled:
        extractor = GrobidTokenExtractor()
        extractor.extract(TEST_PDF_FILEPATH)


def test_pdfplumber():
    extractor = PDFPlumberTokenExtractor()
    extractor.extract(TEST_PDF_FILEPATH)


def test_extractor():
    extractor = PDFExtractor("pdfplumber")
    pdf_layouts, page_images = extractor.load_tokens_and_image(
        TEST_PDF_FILEPATH, resize_image=True
    )
    assert pdf_layouts[0]["tokens"][-1].id == len(pdf_layouts[0]["tokens"]) - 1


def test_two_column_structure():
    extractor = PDFExtractor("pdfplumber")
    pdf_layouts, page_images = extractor.load_tokens_and_image(
        TWO_COLUMN_PDF_FILEPATH, resize_image=True
    )
    page_2 = pdf_layouts[1]
    right_side = lp.Layout(page_2["tokens"]).filter_by(
        lp.Interval(page_2["width"] // 2, page_2["width"], axis="x")
    )

    right_side = lp.Layout(right_side).filter_by(
        lp.Interval(page_2["height"] * 0.1, page_2["height"] * 0.9, axis="y")
    )  # Remove the header and the footer

    # Ensure the right handside is approximately on the left side of the page
    assert right_side[0].id >= len(page_2["tokens"]) // 2 * 0.9
