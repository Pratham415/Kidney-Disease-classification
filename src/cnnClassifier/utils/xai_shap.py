import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any, Optional
import shap
from skimage.segmentation import slic, mark_boundaries
from cnnClassifier import logger
from cnnClassifier.utils.xai_gradcam import np_image_to_base64


def generate_shap_explanation(
    model: tf.keras.Model,
    img_rgb_224: np.ndarray,
    n_segments: int = 25,
    nsamples: int = 50,
    pred_index: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes genuine SHAP (SHapley Additive exPlanations) values for the CNN prediction.

    Mathematical Basis:
    - Rooted in cooperative game theory (Shapley, 1953; Lundberg & Lee, 2017).
    - Features are defined as superpixel segments of the medical image.
    - Each superpixel's Shapley value phi_i satisfies the 4 foundational axioms:
      1. Efficiency: sum(phi_i) = f(x) - E[f(x)]
      2. Symmetry: if i and j contribute equally to all coalitions, phi_i = phi_j
      3. Dummy: if feature i contributes nothing to all coalitions, phi_i = 0
      4. Additivity: for combined games, Shapley values are additive.

    Args:
        model: Trained tf.keras.Model.
        img_rgb_224: RGB image array of shape (224, 224, 3), float32 or uint8.
        n_segments: Number of superpixel segments (default 25).
        nsamples: Number of coalition evaluations in Kernel SHAP (default 50).
        pred_index: Class index to explain (None = top predicted class).

    Returns:
        Tuple:
            - shap_overlay_rgb: Blended original image with red (positive) and blue (negative) SHAP attribution.
            - shap_heatmap_rgb: Pure colormapped SHAP attribution map.
            - metadata: Dictionary of Shapley attribution values and interpretation.
    """
    img_float = img_rgb_224.astype(np.float32)
    if img_float.ndim == 4:
        img_float = img_float[0]

    # Baseline prediction
    preds = model.predict(np.expand_dims(img_float, axis=0), verbose=0)[0]
    if pred_index is None:
        pred_index = int(np.argmax(preds))

    confidence = float(preds[pred_index])
    class_labels = {0: "Normal", 1: "Tumor"}
    target_label = class_labels.get(pred_index, f"Class {pred_index}")

    # 1. Segment image into contiguous superpixels
    segments = slic(img_float, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
    num_superpixels = int(np.unique(segments).size)

    # Background reference: mean color of the input image
    background_color = img_float.mean(axis=(0, 1))

    # Masking helper: sets superpixels to baseline when coalition indicator is 0
    def mask_image(zs: np.ndarray) -> np.ndarray:
        batch_size = zs.shape[0]
        out = np.empty((batch_size, 224, 224, 3), dtype=np.float32)
        for i in range(batch_size):
            out[i] = img_float
            for sp_id in range(num_superpixels):
                if zs[i, sp_id] == 0:
                    out[i][segments == sp_id] = background_color
        return out

    # Model evaluation wrapper for binary coalition masks
    def f(zs: np.ndarray) -> np.ndarray:
        masked_batch = mask_image(zs)
        return model.predict(masked_batch, verbose=0)

    # 2. Kernel SHAP Explainer
    # Reference baseline is all superpixels off (0 vector)
    explainer = shap.KernelExplainer(f, np.zeros((1, num_superpixels)))
    shap_raw = explainer.shap_values(
        np.ones((1, num_superpixels)),
        nsamples=nsamples,
        silent=True
    )

    # Extract Shapley values for target class
    # shap_raw is a list of length [num_classes], each of shape (1, num_superpixels)
    if isinstance(shap_raw, list):
        target_shap_values = shap_raw[pred_index][0]
    else:
        target_shap_values = shap_raw[0]

    # 3. Project superpixel Shapley values onto 2D pixel space
    pixel_shap = np.zeros((224, 224), dtype=np.float32)
    for sp_id in range(num_superpixels):
        pixel_shap[segments == sp_id] = target_shap_values[sp_id]

    # Normalize SHAP values symmetrically around 0 for diverging colormap
    abs_max = np.max(np.abs(pixel_shap))
    if abs_max > 0:
        norm_shap = pixel_shap / abs_max  # range [-1, 1]
    else:
        norm_shap = pixel_shap

    # Map [-1, 1] to [0, 1] for matplotlib diverging coolwarm colormap
    # Positive SHAP -> Red/Warm (pushes towards Tumor)
    # Negative SHAP -> Blue/Cool (pushes towards Normal)
    import matplotlib.cm as cm
    norm_0_1 = (norm_shap + 1.0) / 2.0
    colored_shap_rgb = np.uint8(255 * cm.coolwarm(norm_0_1)[:, :, :3])

    # Blend with original image
    base_img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(base_img_uint8, 0.55, colored_shap_rgb, 0.45, 0)

    # Add superpixel boundaries
    boundaries_float = mark_boundaries(blended / 255.0, segments, color=(0.9, 0.9, 0.9))
    shap_overlay_rgb = np.clip(boundaries_float * 255.0, 0, 255).astype(np.uint8)

    # Collect metadata on top contributing superpixels
    sorted_sp_indices = np.argsort(np.abs(target_shap_values))[::-1]
    top_sp_info = [
        {
            "superpixel_id": int(idx),
            "shap_value": round(float(target_shap_values[idx]), 5),
            "effect": "increases probability" if target_shap_values[idx] > 0 else "decreases probability"
        }
        for idx in sorted_sp_indices[:5]
    ]

    metadata = {
        "explained_class": target_label,
        "class_index": pred_index,
        "confidence": round(confidence, 4),
        "num_superpixels": num_superpixels,
        "coalition_samples": nsamples,
        "top_shap_contributions": top_sp_info,
        "interpretation": f"Warm/Red regions represent positive Shapley attributions increasing '{target_label}' likelihood; Cool/Blue regions represent negative attributions pushing towards the opposite class."
    }

    return shap_overlay_rgb, colored_shap_rgb, metadata
