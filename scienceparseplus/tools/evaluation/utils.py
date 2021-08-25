from PIL import Image, ImageDraw, ImageFont
import numpy as np
import layoutparser as lp


def get_rgba_color(hex_val, alpha):
    h = hex_val.lstrip("#")
    return (*tuple(int(h[i : i + 2], 16) for i in (0, 2, 4)), int(255 * alpha))


def draw_transparent_boxes(image, blocks, color_map, alpha=0.25):

    image = image.copy()
    drw = ImageDraw.Draw(image, "RGBA")

    for block in blocks:
        drw.polygon(
            [tuple(ele) for ele in block.points.tolist()],
            get_rgba_color(color_map[block.type], alpha),
        )

    return image


class LayoutVisualizer:
    DEFAULT_COLOR_MAP = {
        None: "#C0C0C0",
        "abstract": "#ffb6c1",
        "author": "#02028b",
        "caption": "#00ff03",
        "date": "#0f0000",
        "equation": "#ff0000",
        "figure": "#ff01ff",
        "footer": "#C2C2C2",
        "list": "#302070",
        "paragraph": "#b0c4de",
        "reference": "#309000",
        "section": "#0603ff",
        "table": "#01ffff",
        "title": "#00bfff",
    }

    def __init__(self, color_map=None, print_model_name=True, print_model_acc=True):

        if color_map is None:
            self.color_map = self.DEFAULT_COLOR_MAP

        self.print_model_name = print_model_name
        self.print_model_acc = print_model_acc

        self.bold_font = ImageFont.truetype("./RobotoCondensed-Bold.ttf", 25)
        self.tinny_font = ImageFont.truetype("./RobotoCondensed-Regular.ttf", 14)

    def create_viz(
        self, page_image, block_layout, token_layout, model_name="", model_acc=None
    ):
        """Create visualization of the model output."""

        viz = draw_transparent_boxes(
            page_image, token_layout, color_map=self.color_map, alpha=0.5
        )
        viz = draw_transparent_boxes(viz, block_layout, color_map=self.color_map)
        viz = lp.draw_box(
            viz,
            [
                ele.set(
                    id=ele.type.capitalize() if ele.type is not None else "Not Used"
                )
                for ele in block_layout
            ],
            color_map=self.color_map,
            box_width=1,
            show_element_id=True,
            id_font_size=14,
            id_font_path="./RobotoCondensed-Regular.ttf",
        )

        drw = ImageDraw.Draw(viz, "RGBA")
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
        return viz
