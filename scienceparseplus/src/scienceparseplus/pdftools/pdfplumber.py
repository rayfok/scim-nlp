from typing import List

import pandas as pd
import pdfplumber

from .base import BasePDFTokenExtractor
from ..datamodel import Page, Token, load_as_lp_blocks_from_dict


class PDFPlumberTokenExtractor(BasePDFTokenExtractor):
    NAME = "pdfplumber"

    @staticmethod
    def convert_to_pagetoken(row: pd.Series) -> Token:
        """Convert a row in a DataFrame to pagetoken"""
        return dict(
            text=row["text"],
            x=row["x0"],
            width=row["width"],
            y=row["top"],
            height=row["height"],
            font=row["fontname"]
        )

    def obtain_word_tokens(self, cur_page: pdfplumber.page.Page) -> List[Token]:
        """Obtain all words from the current page.
        Args:
            cur_page (pdfplumber.page.Page):
                the pdfplumber.page.Page object with PDF token information
        Returns:
            List[PageToken]:
                A list of page tokens stored in PageToken format.
        """
        words = cur_page.extract_words(
            x_tolerance=1.5,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            horizontal_ltr=True,
            vertical_ttb=True,
            extra_attrs=["fontname", "size"],
        )

        df = pd.DataFrame(words)
        if len(words) == 0:
            return []
            
        # Avoid boxes outside the page
        df[["x0", "x1"]] = (
            df[["x0", "x1"]].clip(lower=0, upper=int(cur_page.width)).astype("float")
        )
        df[["top", "bottom"]] = (
            df[["top", "bottom"]]
            .clip(lower=0, upper=int(cur_page.height))
            .astype("float")
        )

        df["height"] = df["bottom"] - df["top"]
        df["width"] = df["x1"] - df["x0"]

        word_tokens = df.apply(self.convert_to_pagetoken, axis=1).tolist()
        return word_tokens

    def extract(self, pdf_path: str) -> List[Page]:
        """Extracts token text, positions, and style information from a PDF file.
        Args:
            pdf_path (str): the path to the pdf file.
            include_lines (bool, optional): Whether to include line tokens. Defaults to False.
        Returns:
            PdfAnnotations: A `PdfAnnotations` containing all the paper token information.
        """
        plumber_pdf_object = pdfplumber.open(pdf_path)

        pages = []
        for page_id in range(len(plumber_pdf_object.pages)):
            cur_page = plumber_pdf_object.pages[page_id]

            tokens = self.obtain_word_tokens(cur_page)

            page = dict(
                page=dict(
                    width=float(cur_page.width),
                    height=float(cur_page.height),
                    index=page_id,
                ),
                tokens=tokens,
            )
            pages.append(page)

        del plumber_pdf_object
        return load_as_lp_blocks_from_dict(pages)