import os
from pathlib import Path

def models_dir():
    return os.environ.get("MODELSDIR", str(Path(os.path.dirname(__file__)).parent.parent / "models"))

def data_dir():
    return os.environ.get("DATADIR", os.path.dirname(__file__))