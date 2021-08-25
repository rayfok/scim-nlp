from layoutparser import Rectangle, TextBlock
import pandas as pd

from scienceparseplus.datamodel import *


def to_dict(self):

    return {
        "bbox": list(self.coordinates),  # ensure it's a list
        "text": self.text,
        "type": self.type,
        "confidence": self.score,
    }


TextBlock.to_dict = to_dict


def test_hierarchical_layout():

    t = TextBlock(Rectangle(40, 50, 120, 200), type="t")
    a = TextBlock(Rectangle(80, 90, 100, 150), text="a")
    b = TextBlock(Rectangle(80, 90, 100, 450), text="b")

    A = HierarchicalLayout.from_raw_block_token(block_layout=[t], token_layout=[a, b])
    B = HierarchicalLayout(
        bundles=[
            BlockBundle(
                block=t,
                tokens=[a],
            )
        ],
        remaining_tokens=[b],
    )

    assert A == B

    # Test Export
    assert (A.to_dataframe()["text"] == ["a", "a", "b"]).all()
    assert (A.to_dataframe()["parent"][:2] == [-1, 0]).all()
    assert pd.isna(A.to_dataframe()["parent"].iloc[-1])

    dict_representation = {
        "bundles": [
            {
                "bbox": list(t.coordinates),
                "type": "t",
                "confidence": None,
                "tokens": [a.to_dict()],
            }
        ],
        "remaining_tokens": [b.to_dict()],
    }
    assert A.to_dict() == dict_representation
    assert A.from_dict(dict_representation) == A
    assert A.get_all_tokens() == [
        a.set(parent=0),
        b,
    ]  # Note: a's parent has been changed

    # Test token type setting
    assert A.get_all_blocks_regions() == [t.set(text=a.text, id=0)]
    # Note: the text and id has also been changed.
    assert A.to_dataframe(inherit_block_class=True)["category"].iloc[1] == "t"

    A.set_tokens_with_block_class()
    assert A.bundles[0].tokens[0].type == "t"
    assert a.type != "t"  # The original token type hasn't been changed
    assert A.remaining_tokens[0].type != "t"
