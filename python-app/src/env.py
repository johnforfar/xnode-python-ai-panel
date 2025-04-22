import os


def models_dir():
    return os.environ.get("MODELSDIR", os.path.dirname(__file__).parent.parent / "models")

def data_dir():
    return os.environ.get("DATADIR", os.path.dirname(__file__))