"""BANKING77: mesmo checkpoint, instruções e descrições usados pelo CLEF."""

DATASET = "PolyAI/banking77"
DATASET_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
QUESTION = "Identify the customer's banking intent. Choose exactly one option."


def load(split):
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split=split, revision=DATASET_REVISION, token=False)
    classes = dataset.features["label"].names
    options = {name: name.replace("_", " ").replace("?", "") for name in classes}
    return dataset, classes, options
