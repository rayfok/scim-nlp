from scienceparseplus.visualization import *
from scienceparseplus.datamodel import *
from PIL import Image
from layoutparser import TextBlock, Rectangle, Layout


def test_draw_functions():
    t = TextBlock(Rectangle(40, 50, 120, 200), type="t")
    a = TextBlock(Rectangle(80, 90, 100, 150), text="a")
    b = TextBlock(Rectangle(80, 90, 100, 450), text="b")

    A = HierarchicalLayout.from_raw_block_token(block_layout=[t], token_layout=[a, b])

    page_img = Image.new("RGB", (600, 300), color="red")
    page_img_backup = page_img.copy()

    canvas = draw_layout(page_img, A)
    canvas = draw_layout(page_img, [a, b])
    canvas = draw_layout(page_img, [])  # Test robustness for empty regions
    canvas = draw_tokens(page_img, A)
    canvas = draw_tokens(page_img, [a, b])

    visualizer = PageLayoutVisualizer(
        block_boundary_width=2,
        token_boundary_width=5,
    )

    canvas = visualizer.create_viz(page_img, A)

    # Assert the original image hasn't been changed
    assert page_img == page_img_backup