# Provides a series of constants for model loading,
# visualizations, and many others.

UNIVERSAL_LAYOUT_ELEMENT_COLOR_MAP = {
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
    "text": "#b0c4de",
    "reference": "#309000",
    "section": "#0603ff",
    "table": "#01ffff",
    "title": "#00bfff",
}

MAX_BLOCK_EMBEDDING_NUMBER = 32
MAX_2D_POSITION_EMBEDDINGS = 1024

MAX_FONT_NUMBER = 904
# 0 for [PAD]
# 1 for [UNK]
# 2 for [CLS]
# 3 for [MASK]
# 4 for [SEP]
# 899 Unique fonts for DocBank

DEFAULT_MAX_SEQUENCE_LENGTH = 512

IMAGE_FEATURE_SIZE = 1024