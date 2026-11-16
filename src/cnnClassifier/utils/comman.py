"""
Backward compatibility bridge for 'comman.py' -> 'common.py'.
All utilities have been cleanly consolidated in cnnClassifier.utils.common.
"""
from cnnClassifier.utils.common import (
    read_yaml,
    create_directories,
    save_json,
    load_json,
    save_bin,
    load_bin,
    get_size,
    decodeImage,
    encodeImageIntoBase64
)

__all__ = [
    "read_yaml",
    "create_directories",
    "save_json",
    "load_json",
    "save_bin",
    "load_bin",
    "get_size",
    "decodeImage",
    "encodeImageIntoBase64"
]
