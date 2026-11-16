import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, Union
import base64
from cnnClassifier import logger
from cnnClassifier.utils.image_processing import load_and_preprocess_image


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Dynamically identifies the last convolutional/activation layer in the model.
    For EfficientNetV2B0, this is typically 'top_activation' or 'top_conv'.
    """
    for layer in reversed(model.layers):
        if 'top_activation' in layer.name:
            return layer.name
        if 'top_conv' in layer.name:
            return layer.name
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Activation)):
            return layer.name
    # Fallback search
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            return layer.name
    raise ValueError("Could not find a convolutional layer in the model.")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None
) -> Tuple[np.ndarray, int, float]:
    """
    Calculates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmap.

    Mathematical steps:
    1. Forward pass to extract feature map A of the last conv layer and model output y.
    2. Compute gradients of predicted class score y_c with respect to A: d(y_c)/d(A^k).
    3. Global Average Pooling over spatial dimensions (height, width) to compute feature map importance weights alpha_k.
    4. Weighted linear combination: L_GradCAM = sum_k (alpha_k * A^k).
    5. Apply ReLU to retain only features that positively influence the target class.
    6. Normalize heatmap between 0 and 1.

    Args:
        model: Trained tf.keras.Model.
        img_array: Preprocessed image tensor of shape (1, 224, 224, 3).
        last_conv_layer_name: Name of target conv layer (auto-detected if None).
        pred_index: Target class index to explain (defaults to model's top predicted class).

    Returns:
        Tuple: (heatmap (7x7 float32 array in [0, 1]), predicted_class_index, confidence)
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError as e:
        logger.warning(f"Layer '{last_conv_layer_name}' not found. Falling back to dynamic search.")
        last_conv_layer_name = get_last_conv_layer_name(model)
        last_conv_layer = model.get_layer(last_conv_layer_name)

    # Build gradient extraction model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.device('/CPU:0'):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            loss = predictions[:, pred_index]

        # Calculate gradients of the class score with respect to feature maps
        grads = tape.gradient(loss, conv_outputs)

        # Global Average Pooling of gradients (importance weight alpha_k)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by alpha_k
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only positive contributions toward target class
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize between 0 and 1
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    confidence = float(predictions[0][pred_index])
    return heatmap.numpy(), pred_index, confidence


def overlay_heatmap_on_image(
    original_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes heatmap to original image dimensions, applies colormap, and blends with original image.

    Args:
        original_img_rgb: RGB image (uint8, shape (H, W, 3)).
        heatmap: 2D float array in [0, 1].
        alpha: Heatmap blend opacity (0.0 = only original image, 1.0 = only heatmap).
        colormap: OpenCV colormap constant (default COLORMAP_JET).

    Returns:
        Tuple: (blended_rgb, colored_heatmap_rgb)
    """
    h, w = original_img_rgb.shape[:2]
    # Resize heatmap to match original image dimensions using bilinear interpolation
    resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * resized_heatmap)

    # Apply colormap (OpenCV returns BGR)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending: (1 - alpha) * original + alpha * heatmap
    blended = cv2.addWeighted(original_img_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
    return blended, colored_heatmap_rgb


def np_image_to_base64(img_rgb: np.ndarray) -> str:
    """Encodes an RGB uint8 image array into a base64 PNG data URL."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode('.png', img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG buffer.")
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"
