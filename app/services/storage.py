from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from app.core.config import settings


def ensure_dirs() -> None:
    """Ensure all required data directories exist."""
    Path(settings.DATASET_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.SEED_DIR).mkdir(parents=True, exist_ok=True)


def save_upload_to_dir(upload_file, target_dir: str, safe_name: str) -> str:
    """
    Save an uploaded file to the specified directory.
    
    Args:
        upload_file: FastAPI UploadFile object
        target_dir: Target directory path
        safe_name: Safe filename to use
        
    Returns:
        Full path to the saved file
    """
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(target_dir, safe_name)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return out_path


def write_json(path: str, obj: Any) -> None:
    """
    Write a Python object to a JSON file.
    
    Args:
        path: Output file path
        obj: Object to serialize (dict, list, etc.)
    """
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, default=str)


def read_json(path: str) -> dict:
    """
    Read a JSON file and return the parsed object.
    
    Args:
        path: JSON file path
        
    Returns:
        Parsed JSON object
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
