from scienceparseplus.datasets.layout_dataset import *


def test_cocodataset():
    dataset = COCOLayoutDataset("tests/fixtures/coco_dataset")
    (paper_filename, gt_layouts, page_images, pdf_page_info) = dataset[0]

    assert (
        paper_filename
        == "tests/fixtures/coco_dataset/pdfs/0a238252712f2c7f887d32056d36a26fcefaa634.pdf"
    )

    assert len(gt_layouts) == 17
    assert gt_layouts[0].remaining_tokens == []
