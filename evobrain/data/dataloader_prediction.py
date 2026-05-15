"""Stub for prediction-task loader.

The published EvoBrain repo references this module from main.py but does not
ship its implementation. We only need the import to succeed; calling the
function from a `--task prediction` run raises a clear error.
"""


def load_dataset_prediction(*args, **kwargs):
    raise NotImplementedError(
        "load_dataset_prediction is not available — the EvoBrain repository "
        "does not include data/dataloader_prediction.py. Only --task detection "
        "is supported.")
