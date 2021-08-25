"""
Modified based on https://github.com/allenai/pawls/blob/master/cli/pawls/preprocessors/pdfplumber.py
"""
import json
from dataclasses import dataclass  # enables inheritance
from typing import List, Tuple, Dict, Union, Any
from copy import copy

import pandas as pd
import layoutparser as lp

__all__ = [
    "Box",
    "Token",
    "Block",
    "PageInfo",
    "Page",
    "BlockBundle",
    "HierarchicalLayout",
    "load_tokens_from_file",
    "load_tokens_from_dict",
]


def union_boxes(boxes: List["Box"]) -> "Box":
    """Find the outside boundary of the given boxes.
    Args:
        boxes (List[Box]):
            A list of Box-like instances
    Returns:
        Box: the unioned box.
    """
    left, top, right, bottom = float("inf"), float("inf"), float("-inf"), float("-inf")
    for box in boxes:
        l, t, r, b = box.coordinates
        left = min(left, l)
        top = min(top, t)
        right = max(right, r)
        bottom = max(bottom, b)
    return Box(left, top, right - left, bottom - top)


def is_in(block_a, block_b):
    """A rewrite of the lp.LayoutElement.is_in function.
    We will use a soft_margin and center function by default.
    """
    return block_a.is_in(
        block_b,
        soft_margin={"top": 1, "bottom": 1, "left": 1, "right": 1},
        center=True,
    )


# fmt: off
# Monkey dispatching for layoutparser
# Adding to_dict method for textblocks
def to_dict(self):

    return {
        "bbox": list(self.coordinates), # ensure it's a list
        "text": self.text,
        "type": self.type,
        "confidence": self.score,
    }

@classmethod
def from_dict(cls, data):
    x_1, y_1, x_2, y_2 = data['bbox']
    return cls(
        block=lp.Rectangle(x_1, y_1, x_2, y_2),
        text = data['text'],
        type = data['type'],
        score = data['confidence']
    )
    
lp.TextBlock.to_dict = to_dict
lp.TextBlock.from_dict = from_dict
# fmt: on


@dataclass
class Box:

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        """Return the center of the token box"""
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def coordinates(self) -> Tuple[float, float, float, float, float]:
        """Returns the left, top, right, bottom coordinates of the box"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def is_in(self, other: "Box", soft_margin: Dict = None) -> bool:
        """Determines whether the center of this box is contained
        inside another box with a soft margin.
        Args:
            other (Box):
                The other box object.
            soft_margin (Dict, optional):
                Alllowing soft margin of the box boundaries. If set, enlarge
                the outside box (other) by the coordinates.
                Defaults to {}.
        """
        other = other.copy()

        x, y = self.center
        if soft_margin is not None:
            other.pad(**soft_margin)
        xa, ya, xb, yb = other.coordinates

        return xa <= x <= xb and ya <= y <= yb

    def pad(self, left=0, top=0, bottom=0, right=0):
        """Change the boundary positions of the box"""

        self.x -= left
        self.y -= top
        self.width += left + right
        self.height += top + bottom

    def copy(self):
        """Create a copy of the box"""
        return self.__class__(**vars(self))

    def scale(self, scale_factor: Union[float, Tuple[float, float]]):
        """Scale the box according to the given scale factor.
        Args:
            scale_factor (Union[float, Tuple[float, float]]):
                it can be either a float, indicating the same scale factor
                for the two axes, or a two-element tuple for x and y axis
                scaling factor, respectively.
        """

        if isinstance(scale_factor, float):
            self.x *= scale_factor
            self.y *= scale_factor
            self.width *= scale_factor
            self.height *= scale_factor

        elif isinstance(scale_factor, tuple):
            scale_x, scale_y = scale_factor
            self.x *= scale_x
            self.y *= scale_y
            self.width *= scale_x
            self.height *= scale_y

    def as_bounds(self) -> Dict[str, float]:
        """Convert the box into a the bounds format."""
        return {
            "left": self.x,
            "top": self.y,
            "right": self.x + self.width,
            "bottom": self.y + self.height,
        }


@dataclass
class Token(Box):
    text: str


@dataclass
class Block(Box):
    label: str


@dataclass
class PageInfo:
    width: float
    height: float
    index: int

    def scale(self, scale_factor: Union[float, Tuple[float, float]]):
        """Scale the page box according to the given scale factor
        Args:
            scale_factor (Union[float, Tuple[float, float]]):
                it can be either a float, indicating the same scale factor
                for the two axes, or a two-element tuple for x and y axis
                scaling factor, respectively.
        """

        if isinstance(scale_factor, float):
            self.width *= scale_factor
            self.height *= scale_factor
        elif isinstance(scale_factor, tuple):
            scale_x, scale_y = scale_factor
            self.width *= scale_x
            self.height *= scale_y


@dataclass
class Layout:
    tokens: List[Union[Token, Block]]

    def scale(self, scale_factor: Union[float, Tuple[float, float]]):
        """Scale the page according to the given scale factor.
        Args:
            scale_factor (Union[float, Tuple[float, float]]):
                it can be either a float, indicating the same scale factor
                for the two axes, or a two-element tuple for x and y axis
                scaling factor, respectively.
        """
        for token in self.tokens:
            token.scale(scale_factor)

    def filter_tokens_by(self, box: Box, soft_margin: Dict = None) -> Dict[int, Token]:
        """Select tokens in the Page that inside the input box"""
        return {
            idx: token
            for idx, token in enumerate(self.tokens)
            if token.is_in(box, soft_margin)
        }


@dataclass
class Page(Layout):
    page: PageInfo

    def scale_like(self, other: "Page"):
        """Scale the page based on the other page."""

        scale_x = other.page.width / self.page.width
        scale_y = other.page.height / self.page.height

        self.scale((scale_x, scale_y))

    def scale(self, scale_factor: Union[float, Tuple[float, float]]):
        """Scale the page according to the given scale factor."""
        super().scale(scale_factor)
        self.page.scale(scale_factor)


@dataclass
class BlockBundle:
    """BlockBundle stores a block box and the containing tokens."""

    block: lp.TextBlock
    tokens: List[lp.TextBlock]

    def to_dict(self) -> Dict[str, Any]:
        block_info = self.block.set(text=self.text).to_dict()
        return {
            "bbox": block_info["bbox"],
            "type": block_info["type"],
            "confidence": block_info["confidence"],
            "tokens": [ele.to_dict() for ele in self.tokens],
        }

    @property
    def text(self) -> str:
        """Return all text in this block, joined by a space."""
        return " ".join([token.text for token in self.tokens])

    @classmethod
    def from_dict(cls, data: Dict) -> "BlockBundle":
        """Load a BlockBundle from a dictionary."""
        tokens = [lp.TextBlock.from_dict(ele) for ele in data["tokens"]]
        block = lp.TextBlock.from_dict(
            dict(
                bbox=data["bbox"],
                text="",
                type=data["type"],
                confidence=data["confidence"],
            )
        )
        return cls(block, tokens)


@dataclass
class HierarchicalLayout:
    """HierarchicalLayout is used to describe the page layout, including:
    a list of BlockBundle and the remaining tokens on this page without being
    included in any BlockBundles."""

    bundles: List[BlockBundle]
    remaining_tokens: List[lp.elements.BaseLayoutElement]

    @classmethod
    def from_raw_block_token(
        cls, block_layout: List, token_layout: List, keep_empty_blocks=False
    ):
        """Creates a HierarchicalLayout with un-grouped blocks and tokens

        Args:
            block_layout (List): A list of blocks
            token_layout (List): A list of tokens
            keep_empty_blocks (bool, optional):
                If set to True, it won't drop block bundles without any tokens.
                Defaults to False.
        """
        bundles = []

        for block in block_layout:
            token_group = []
            remaining_tokens = []
            for token in token_layout:
                if is_in(token, block):
                    token_group.append(copy(token))
                else:
                    remaining_tokens.append(token)
            if len(token_group) > 0 or keep_empty_blocks:
                bundles.append(BlockBundle(block, token_group))
            token_layout = remaining_tokens

        remaining_tokens = [copy(token) for token in token_layout]

        return cls(bundles=bundles, remaining_tokens=remaining_tokens)

    def set_tokens_with_block_class(self):

        for bundle in self.bundles:
            for token in bundle.tokens:
                token.type = bundle.block.type

    def get_all_blocks_regions(self) -> List:
        return [
            bundle.block.set(id=idx, text=bundle.text)
            for idx, bundle in enumerate(self.bundles)
        ]

    def get_all_tokens(self, inherit_block_class=False) -> List[lp.TextBlock]:
        """Fetch all tokens and return a list.

        Args:
            inherit_block_class (bool, optional):
                If True, all tokens will be assigned to the same category
                as their parent blocks.
                Defaults to False.

        Returns:
            List: A list of tokens
        """
        all_tokens = []

        for idx, bundle in enumerate(self.bundles):

            for token in bundle.tokens:

                if inherit_block_class:
                    all_tokens.append(token.set(parent=idx, type=bundle.block.type))
                else:
                    all_tokens.append(token.set(parent=idx))

        remaining_tokens = [token.set(parent=None) for token in self.remaining_tokens]

        return all_tokens + remaining_tokens

    def to_dataframe(
        self, inherit_block_class=False, keep_token_index=True
    ) -> pd.DataFrame:
        """Convert the HierarchicalLayout into a DataFrame.
        There are 10 columns in this table, namely:
        id, x_1, y_1, x_2, y_2, text, category, parent, confidence, and is_block.

        The DataFrame is composed of two parts:
        1. the first several rows are all the "block"s in this layout,
            which are generated from `self.get_all_blocks_regions`.
        2. the other rows are for the tokens, and generated based on
            `self.get_all_tokens`.

        The `is_block` feature in the table will be used for discriminating between
        block rows and token rows.

        For tokens within some blocks, the `parent` feature points to the id of their
        parent blocks.

        Args:
            inherit_block_class (bool, optional):
                If True, all tokens will be assigned to the same category
                as their parent blocks.
                Defaults to False.

            keep_token_index (bool, optional):
                By default, the token extractor should generate ids for the tokens.
                It's better to keep them as they should be deterministic if the
                pdf token extractor is set.
                Defaults to True.

        Returns:
            pd.DataFrame
        """
        _blocks = self.get_all_blocks_regions()
        _tokens = self.get_all_tokens(inherit_block_class=inherit_block_class)

        blocks_to_save = [
            [ele.id, *ele.coordinates, ele.text, ele.type, -1, ele.score, True]
            for ele in _blocks
        ]
        tokens_to_save = [
            [
                ele.id if keep_token_index else idx,
                *ele.coordinates,
                ele.text,
                ele.type,
                ele.parent,
                ele.score,
                False,
            ]
            for idx, ele in enumerate(_tokens, start=len(blocks_to_save))
        ]
        df = pd.DataFrame(
            blocks_to_save + tokens_to_save,
            columns=[
                "id",
                "x_1",
                "y_1",
                "x_2",
                "y_2",
                "text",
                "category",
                "parent",
                "confidence",
                "is_block",
            ],
        )

        if any(hasattr(tok, 'font') for tok in _tokens):
            df['font'] = [None]*len(blocks_to_save) + [tok.font for tok in _tokens]

        return df

    def to_dict(self) -> Dict[str, Any]:
        """Convert the HierarchicalLayout into a DataFrame.

        The dictionary structure looks like this::
            {
                "bundles": [
                    {
                        bundle_information
                    },
                    ...
                ],
                "remaining_tokens": [
                    {
                        token_information
                    },
                    ...
                ]
            }
        """
        return {
            "bundles": [bundle.to_dict() for bundle in self.bundles],
            "remaining_tokens": [
                token.set(parent=None).to_dict() for token in self.remaining_tokens
            ],
        }

    @classmethod
    def from_dict(cls, data) -> "HierarchicalLayout":
        """Load a HierarchicalLayout from a dictionary."""
        remaining_tokens = [
            lp.TextBlock.from_dict(ele) for ele in data["remaining_tokens"]
        ]
        bundles = [BlockBundle.from_dict(bundle) for bundle in data["bundles"]]

        return cls(bundles, remaining_tokens)


def load_tokens_from_file(filename: str) -> List[Page]:
    """Load tokens files into the data model
    Returns:
        List[Page]:
            A list of `Page` object for eac page.
    """

    with open(filename, "r") as fp:
        source_data = json.load(fp)

    return load_tokens_from_dict(source_data)


def load_tokens_from_dict(source_data: Dict[str, Any]) -> List[Page]:

    return [
        Page(
            page=PageInfo(**page_data["page"]),
            tokens=[Token(**token) for token in page_data["tokens"]],
        )
        for page_data in source_data
    ]


def load_as_lp_blocks_from_dict(source_data: Dict[str, Any]) -> List[Dict]:
    """Load the tokens as layoutparser data structures. This is to support
    some old implementations and should be removed in the future.

    Args:
        source_data (Dict[str, Any]):
            A list of page information, sotred in dictionary formats

    Returns:
        List[Dict]:
            A list of page information dictionaries,
            with box information stored in layoutparser data formats.
    """

    def create_tokens(tokens):

        all_token_boxes = []
        for idx, token in enumerate(tokens):

            token_box = lp.TextBlock(
                lp.Rectangle(
                    x_1=token["x"],
                    y_1=token["y"],
                    x_2=token["x"] + token["width"],
                    y_2=token["y"] + token["height"],
                ),
                text=token["text"],
                id=idx, 
            )
            token_box.font = token.get("font")
            all_token_boxes.append(token_box)

        return lp.Layout(all_token_boxes)

    return [
        {
            "height": page_data["page"]["height"],
            "width": page_data["page"]["width"],
            "tokens": create_tokens(page_data["tokens"]),
        }
        for page_data in source_data
    ]