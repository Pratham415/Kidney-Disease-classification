import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import re


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads YAML file and returns ConfigBox.

    Args:
        path_to_yaml (Path): Path to YAML file.

    Raises:
        ValueError: If YAML file is empty.
        Exception: If reading fails.

    Returns:
        ConfigBox: ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            if content is None:
                raise ValueError("yaml file is empty")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Creates a list of directories if they don't already exist.

    Args:
        path_to_directories (list): List of paths to directories.
        verbose (bool, optional): Whether to log creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves dictionary data to a JSON file.

    Args:
        path (Path): Path to output JSON file.
        data (dict): Data dictionary to be serialized.
    """
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads JSON file data into ConfigBox.

    Args:
        path (Path): Path to JSON file.

    Returns:
        ConfigBox: Loaded data accessible via class attributes.
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves binary file via joblib.

    Args:
        data (Any): Python object to serialize.
        path (Path): Destination file path.
    """
    os.makedirs(path.parent, exist_ok=True)
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads serialized binary data.

    Args:
        path (Path): Path to binary file.

    Returns:
        Any: Deserialized Python object.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Returns file size in KB.

    Args:
        path (Path): Target file path.

    Returns:
        str: Human-readable size string.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring: str, fileName: str):
    """Decodes a base64 encoded image string (handling data URI headers) and writes to disk.

    Args:
        imgstring (str): Base64 encoded string, with or without data:image prefix.
        fileName (str): Destination output image path.
    """
    # Strip potential data URI prefix (e.g. 'data:image/jpeg;base64,')
    if "," in imgstring:
        imgstring = imgstring.split(",", 1)[1]

    # Clean whitespace or line breaks
    imgstring = re.sub(r"\s+", "", imgstring)

    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(croppedImagePath: str) -> bytes:
    """Encodes an image file on disk into base64 bytes.

    Args:
        croppedImagePath (str): Path to image file.

    Returns:
        bytes: Base64 encoded byte stream.
    """
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
