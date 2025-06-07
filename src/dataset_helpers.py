# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
import os
from datasets import load_dataset

SUPPORTED_FORMATS = ["csv", "json", "jsonl", "parquet"]
REQ_COLUMNS = {"System", "User", "Assistant"}


def load_local_dataset(path: str, file_fmt: str):
    """Load a local dataset file with the correct HF loader."""
    if file_fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {file_fmt} – choose one of {', '.join(SUPPORTED_FORMATS)}")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    try:
        if file_fmt == "jsonl":
            return load_dataset("json", data_files={"train": path}, split=None)
        elif file_fmt == "json":
            return load_dataset("json", data_files={"train": path})
        else:
            return load_dataset(file_fmt, data_files={"train": path})
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {path}: {str(e)}")


def ensure_columns(dataset):
    """Verify that required columns exist in *all* splits."""
    missing = REQ_COLUMNS - set(dataset["train"].column_names)
    if missing:
        raise ValueError(f"Dataset must contain columns {', '.join(REQ_COLUMNS)}. Missing: {', '.join(missing)}")
