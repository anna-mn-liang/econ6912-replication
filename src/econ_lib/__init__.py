from pathlib import Path
__version__ = "0.0.1"

data_dir = Path(__file__).parent.parent.parent / "data"
outputs_dir = Path(__file__).parent.parent.parent / "outputs"
outputs_dir.mkdir(exist_ok=True, parents=True)