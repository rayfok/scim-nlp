from fvcore.common.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = ""
_C.DATASET.BASE_PATH = ""

_C.TRAIN = CN()

# General training configs
_C.TRAIN.BATCH_SIZE_PER_DEVICE = 20
_C.TRAIN.NUM_TRAIN_EPOCHS = 3
_C.TRAIN.METRIC_FOR_BEST_MODEL = "accuracy"
_C.TRAIN.LOAD_BEST_MODEL_AT_END = True
_C.TRAIN.SAVE_TOTAL_LIMIT = 3

# Hyperparameters for optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.LEARNING_RATE = 2e-5
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.01
_C.TRAIN.OPTIMIZER.LABEL_SMOOTHING_FACTOR = 0.0

# Logging configs
_C.TRAIN.LOGGING = CN()
_C.TRAIN.LOGGING.STRATEGY = "steps"
_C.TRAIN.LOGGING.STEPS = 500

# Model saving configs
_C.TRAIN.SAVE = CN()
_C.TRAIN.SAVE.STRATEGY = "steps"
_C.TRAIN.SAVE.STEPS = 2000

# Model evaluation configs
_C.TRAIN.EVAL = CN()
_C.TRAIN.EVAL.STRATEGY = "steps"
_C.TRAIN.EVAL.STEPS = 2000
_C.TRAIN.EVAL.BATCH_SIZE_PER_DEVICE = 20


def get_default_cfg():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()