import layoutparser as lp

from pdfminer.layout import LAParams, LTTextBox, LTText, LTChar, LTTextLine, LTTextLineHorizontal, LTAnno
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator

from pdf2image import convert_from_path
from collections.abc import Iterable
from itertools import cycle


def union_boxes(layout_elements):
    """Obtaining the bounding boxes and texts of the union of the input tokens

    Args:
        layout_elements (List): A list of layout elments.

    Returns:
        layoutparser.TextBlock: A single text block of the union of the layout elements.
    """
    x_1 = min([ele.block.x_1 for ele in layout_elements])
    y_1 = min([ele.block.y_1 for ele in layout_elements])
    x_2 = max([ele.block.x_2 for ele in layout_elements])
    y_2 = max([ele.block.y_2 for ele in layout_elements])
    text = ''.join([ele.text for ele in layout_elements])
    return lp.TextBlock(lp.Rectangle(x_1, y_1, x_2, y_2), text=text)


def create_word_token_box(wordobj, page, scale_factor=(1, 1)):
    """Create the word token box from wordobj and scale_factor

    Args:
        scale_factor (tuple, optional): 
            (scale_width, scale_height) for the token coordinates. 
            Defaults to (1, 1).

    Returns:
        layoutparser.TextBlock: A single text block for the given token.
    """
    
    # Coordinate conversion - refer to https://stackoverflow.com/a/53360415
    x_1, y_2_orig, x_2, y_1_orig = wordobj.bbox
    y_1 = page.mediabox[3] - y_1_orig
    y_2 = page.mediabox[3] - y_2_orig

    x_1 *= scale_factor[0]
    x_2 *= scale_factor[0]
    y_1 *= scale_factor[1]
    y_2 *= scale_factor[1]

    return lp.TextBlock(lp.Rectangle(x_1, y_1, x_2, y_2), text=wordobj.get_text())


def calculate_scale_factor(page, output_size):
    """Calculate the token bounding box scale factor based on the page size 
    and the output size. 

    Args:
        page (PDFPage): the PDFPage page object
        output_size (tuple): The (width, height) of the output object

    Returns:
        tuple: (scale_width, scale_height)
    """
    if output_size is None:
        return 1, 1
    else:
        return output_size[0]/page.mediabox[2], output_size[1]/page.mediabox[3]


def load_PDF_tokens(filename, verbose=False, output_sizes=None):
    """Load PDF Tokens from the filename and scale it to the output sizes if set. 

    Args:
        filename ([type]): [description]
        verbose (bool, optional): [description]. Defaults to False.
        output_sizes ([type], optional): [description]. Defaults to None.

    Returns:
        List: A list of layoutparser.Layout, each denotes the token layout for a page. 
    """
    if not isinstance(output_sizes, Iterable):
        output_sizes = cycle([output_sizes])

    with open(filename, 'rb') as fp:

        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pages = PDFPage.get_pages(fp)

        page_layouts = []

        for page_idx, (page, output_size) in enumerate(zip(pages, output_sizes)):
            if verbose:
                print(f'Processing page {page_idx}. ')

            page_layout = lp.Layout()
            interpreter.process_page(page)
            layout = device.get_result()

            scale_factor = calculate_scale_factor(page, output_size)

            for lobj in layout:
                if isinstance(lobj, LTTextBox):
                    for tobj in lobj:
                        if tobj.__class__ == LTTextLineHorizontal:
                            current_word = []
                            for wordobj in tobj:
                                if wordobj.__class__ == LTChar:
                                    current_word.append(create_word_token_box(
                                        wordobj, page, scale_factor))
                                elif wordobj.__class__ == LTAnno:
                                    word_block = union_boxes(current_word)
                                    if word_block.text and not word_block.text.isspace():
                                        page_layout.append(word_block)
                                    current_word = []

            if current_word:
                word_block = union_boxes(current_word)
                if word_block.text and not word_block.text.isspace():
                    page_layout.append(word_block)

            page_layouts.append(page_layout)

    return page_layouts


def load_PDF_tokens_with_images(filename, verbose=True):

    page_images = convert_from_path(filename)
    output_sizes = [image.size for image in page_images]
    page_layouts = load_PDF_tokens(filename, output_sizes=output_sizes)

    return page_images, page_layouts
