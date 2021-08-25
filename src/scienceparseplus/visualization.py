from typing import List, Union, Dict, Any, Tuple
import os
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import layoutparser as lp

from .constants import UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP
from .datamodel import HierarchicalLayout

# A even more dirty hack to get the font files
_lib_path = os.path.dirname(sys.modules[lp.__package__].__file__)
_font_path = os.path.join(_lib_path, "misc", "NotoSerifCJKjp-Regular.otf")


def get_rgba_color(hex_val: str, alpha: float) -> Tuple[int, int, int, int]:
    """Given a color specified by the hex string in `hex_val`,
    and the alpha value ( a float ranging from 0.0 to 1.0 ),
    return a 4-element tuple for the RGBA color.
    """
    h = hex_val.lstrip("#")
    return (*tuple(int(h[i : i + 2], 16) for i in (0, 2, 4)), int(255 * alpha))


def draw_transparent_boxes(
    image: "Image", blocks: List, color_map: Dict = None, alpha: float = 0.25
) -> "Image":
    """Given the image, draw a series of transparent boxes based on the blocks,
    coloring using the specified color_map.
    If the color_map is specified, it will use the
    `UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP`.
    """

    if color_map is None:
        color_map = UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP

    image = image.copy()
    drw = ImageDraw.Draw(image, "RGBA")

    for block in blocks:
        drw.polygon(
            [tuple(ele) for ele in block.points.tolist()],
            get_rgba_color(color_map[block.type], alpha),
        )

    return image


def draw_tokens(
    image: "Image",
    tokens: Union[List, lp.Layout, HierarchicalLayout],
    color_map: Dict[str, str] = None,
    token_boundary_width: int = 0,
    alpha: float = 0.25,
    **kwargs,
) -> "Image":
    """Draw transparent boxes for tokens on a page.

    Args:
        image (Image):
            the page image
        tokens (Union[List, lp.Layout, HierarchicalLayout]):
            the tokens could be either a list of token blocks,
            or from Hierarchical Layout.
        color_map (Dict, optional):
            A category_name to color_hex_string mapping.
            When it is not specified, it will use the
            `UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP`.
        token_boundary_width (int, optional):
            Whether to draw a solid boundary line of token_boundary_width
            around the bbox. If token_boundary_width is 0, it won't draw
            the token box.
            Defaults to 0.
        alpha (float, optional):
            The alpha value for specifying the transparency of the
            token boxes
            Defaults to 0.25.

    Keyword Args:
        They will be used for specifying the features for
        the drawn boundaries.
    """
    if color_map is None:
        color_map = UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP

    if isinstance(tokens, HierarchicalLayout):
        tokens = tokens.get_all_tokens()

    canvas = draw_transparent_boxes(
        image=image, blocks=tokens, color_map=color_map, alpha=alpha
    )
    if token_boundary_width > 0:
        canvas = lp.draw_box(
            canvas,
            tokens,
            box_width=token_boundary_width,
            color_map=color_map,
            **kwargs,
        )
    return canvas


def draw_layout(
    image: "Image",
    layout: Union[List, lp.Layout, HierarchicalLayout],
    color_map: Dict[str, str] = None,
    block_boundary_width: int = 1,
    alpha: float = 0.25,
    **kwargs,
) -> "Image":
    """Draw boxes for both layout blocks and tokens on a page.
    For tokens, it will draw transparent boxes. For blocks,
    it will draw rectangular lins around the box.

    Args:
        image (Image):
            the page image
        layout (Union[List, lp.Layout, HierarchicalLayout]):
            the layout contains both the block and token information
            if it is not a HierarchicalLayout, it will directly
            use the draw_tokens function.
        color_map (Dict, optional):
            A category_name to color_hex_string mapping.
            When it is not specified, it will use the
            `UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP`.
        block_boundary_width (int, optional):
            The width of the block boundary, must be more than 0.
            Defaults to 1.
        alpha (float, optional): [description].
            The alpha value for specifying the transparency of the
            token boxes
            Defaults to 0.25.
    """
    if color_map is None:
        color_map = UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP

    if block_boundary_width <= 0:
        print("The block boundary must be more than 0!")
        block_boundary_width = 1

    if not isinstance(layout, HierarchicalLayout):
        canvas = draw_tokens(
            image=image, tokens=layout, color_map=color_map, alpha=alpha, **kwargs
        )
        return canvas

    tokens = layout.get_all_tokens()
    canvas = draw_tokens(
        image=image, tokens=tokens, color_map=color_map, alpha=alpha, **kwargs
    )

    blocks = layout.get_all_blocks_regions()

    canvas = lp.draw_box(
        canvas,
        [
            ele.set(id=ele.type.capitalize() if ele.type is not None else "Not Used")
            for ele in blocks
        ],
        color_map=color_map,
        box_width=int(block_boundary_width),
        show_element_id=True,
        id_font_size=13,
    )

    return canvas


class PageLayoutVisualizer:
    def __init__(
        self,
        color_map: Dict[str, str] = None,
        print_model_name: bool = True,
        print_model_acc: bool = True,
        **kwargs,
    ):
        """Used for Visualization PDF Page Layout

        Args:
            color_map (Dict, optional):
                A category_name to color_hex_string mapping.
                When not set, it will use the self.DEFAULT_COLOR_MAP automatically.
            print_model_name (bool, optional):
                Whether to print the name of the prediction model on the visualization.
                Defaults to True.
            print_model_acc (bool, optional):
                Whether to print the layout prediction accuracy when available
                Defaults to True.

        Keyword Args:
            The keyword args are used for styling the draw_layout function,
            see the documentation for `draw_layout`.
        """

        if color_map is None:
            self.color_map = UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP

        self.print_model_name = print_model_name
        self.print_model_acc = print_model_acc

        self.bold_font = ImageFont.truetype(_font_path, 25)
        self.tinny_font = ImageFont.truetype(_font_path, 14)

        self.configs = kwargs.copy()

    def create_viz(
        self,
        page_image: Image,
        layout: HierarchicalLayout,
        model_name: str = "",
        model_acc: str = None,
    ):
        """Create visualization of the model output

        Args:
            page_image (Image):
                The image of the page
            block_layout (Union[lp.Layout, List]):
                The block-level layout
            token_layout (Union[lp.Layout, List]):
                The token-level layout
            model_name (str, optional): [description]. Defaults to "".
            model_acc (str, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        page_image = page_image.copy()

        canvas = draw_layout(page_image, layout, self.color_map, **self.configs)
        drw = ImageDraw.Draw(canvas, "RGBA")

        if self.print_model_name:
            bbox = np.array(self.bold_font.getbbox(model_name)) + np.array(
                [400, 20, 400, 20]
            )
            drw.rectangle(
                (bbox).tolist(), fill=(255, 255, 255, 125), outline=(0, 0, 0, 15)
            )
            drw.text([400, 20], model_name, fill=(0, 0, 0), font=self.bold_font)

        if self.print_model_acc and model_acc is not None:
            drw.text(
                [400, 50],
                f"Accuracy: {model_acc:.2%}",
                fill=(0, 0, 0),
                font=self.tinny_font,
            )

        return canvas