import os
from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np
from cnnClassifier import logger


def load_and_preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Validates, loads, and preprocesses a medical image for EfficientNetV2B0.

    Requirements addressed:
    1. Validates that the input file exists, is a regular file, and is non-empty.
    2. Loads the image safely via OpenCV.
    3. Preserves 3 channels (RGB format, never converts to grayscale).
    4. Resizes to target_size (224x224) using high-quality interpolation
       (cv2.INTER_AREA for downscaling, cv2.INTER_CUBIC for upscaling).
    5. Converts to float32 without manual 1/255 rescaling (EfficientNetV2B0
       handles normalization internally).
    6. Adds batch dimension, producing shape: (1, 224, 224, 3).

    Args:
        image_path: Path to the image file (str or Path).
        target_size: Target (height, width), defaults to (224, 224).

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 224, 224, 3) and dtype float32.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is not a regular file, is empty, or cannot be decoded as an image.
        RuntimeError: If preprocessing encounters an unexpected error.
    """
    path_obj = Path(image_path)

    # 1. Validation: File existence and non-empty check
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file does not exist: '{image_path}'")

    if not path_obj.is_file():
        raise ValueError(f"Path is not a regular file: '{image_path}'")

    if path_obj.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): '{image_path}'")

    # 2. Safe loading using OpenCV
    try:
        img = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise ValueError(f"Failed to read image file '{image_path}': {str(e)}") from e

    if img is None:
        raise ValueError(
            f"Unsupported or corrupted image file: '{image_path}' could not be decoded as an image."
        )

    try:
        # 3. Ensure 3-channel RGB format (Do NOT convert to grayscale)
        if img.ndim == 2:
            # Single-channel grayscale input -> duplicate to 3-channel RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3:
            channels = img.shape[2]
            if channels == 3:
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                # BGRA -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif channels == 1:
                # 3D with 1 channel -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(
                    f"Unsupported number of image channels: {channels}. Expected 1, 3, or 4 channels."
                )
        else:
            raise ValueError(f"Unsupported image dimensions: shape {img.shape}")

        # 4. Resize to target size (224, 224) using high-quality interpolation
        target_h, target_w = target_size
        h, w = img_rgb.shape[:2]

        if h == target_h and w == target_w:
            resized = img_rgb
        else:
            if h > target_h or w > target_w:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_CUBIC

            resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=interpolation)

        # 5. Convert to float32 (keep [0, 255] raw pixel scale for EfficientNetV2B0)
        img_array = resized.astype(np.float32)

        # 6. Add batch dimension -> (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        logger.info(
            f"Image '{image_path}' preprocessed successfully: original shape {(h, w, img.shape[2] if img.ndim == 3 else 1)} -> batch shape {img_batch.shape}"
        )
        return img_batch

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Preprocessing failure for '{image_path}': {str(e)}") from e
