import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any, Optional
from lime import lime_image
from skimage.segmentation import mark_boundaries
from cnnClassifier import logger
from cnnClassifier.utils.xai_gradcam import np_image_to_base64


def generate_lime_explanation(
    model: tf.keras.Model,
    img_rgb_224: np.ndarray,
    num_samples: int = 80,
    num_features: int = 5,
    pred_index: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes Local Interpretable Model-Agnostic Explanations (LIME) for image classification.

    Mathematical & Algorithmic Process:
    1. Segments the input image into interpretable, contiguous superpixels using SLIC / Quickshift.
    2. Perturbs the image by turning subsets of superpixels on/off (masked with grey/mean color).
    3. Obtains model prediction probabilities for all perturbed variations: f(z').
    4. Measures distance between original image x and perturbation z' in the superpixel space: pi_x(z').
    5. Fits an interpretable, locally-weighted linear surrogate model (Ridge regression) g:
       min_{g} L(f, g, pi_x) + Omega(g)
    6. Superpixels with high positive weights support the target class prediction; negative weights oppose it.

    Args:
        model: Trained tf.keras.Model.
        img_rgb_224: RGB image array of shape (224, 224, 3), float32 or uint8.
        num_samples: Number of perturbation samples to generate (default 80 for optimal latency/accuracy).
        num_features: Number of top superpixels to highlight.
        pred_index: Class index to explain (None = top predicted class).

    Returns:
        Tuple:
            - overlay_rgb: Image with boundaries and color-coded positive (green) and negative (red) superpixels.
            - positive_only_rgb: Image highlighting only the top supporting superpixels.
            - metadata: Dictionary containing feature weights, top class, and confidence.
    """
    img_float = img_rgb_224.astype(np.float32)
    if img_float.ndim == 4:
        img_float = img_float[0]

    # Predict function for LIME: expects (N, H, W, 3) float32 array
    def predict_fn(images: np.ndarray) -> np.ndarray:
        return model.predict(images.astype(np.float32), verbose=0)

    # Initial prediction
    initial_preds = predict_fn(np.expand_dims(img_float, axis=0))[0]
    if pred_index is None:
        pred_index = int(np.argmax(initial_preds))

    confidence = float(initial_preds[pred_index])
    class_labels = {0: "Normal", 1: "Tumor"}
    target_label = class_labels.get(pred_index, f"Class {pred_index}")

    # Instantiate explainer
    explainer = lime_image.LimeImageExplainer(random_state=42)

    # Generate perturbation explanations
    explanation = explainer.explain_instance(
        img_float.astype(np.double),
        predict_fn,
        labels=(pred_index,),
        hide_color=0,
        num_samples=num_samples,
        random_seed=42
    )

    # 1. Image showing both positive and negative contributors
    temp_both, mask_both = explanation.get_image_and_mask(
        pred_index,
        positive_only=False,
        num_features=num_features,
        hide_rest=False,
        min_weight=0.0
    )

    # Convert temp to uint8 RGB
    base_img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
    overlay_both = base_img_uint8.copy()

    # Highlight positive superpixels with transparent green tint, negative with red tint
    # mask_both == 1: positive feature, mask_both == -1: negative feature
    pos_mask = (mask_both > 0)
    neg_mask = (mask_both < 0)

    # Apply tint
    overlay_both[pos_mask] = (overlay_both[pos_mask] * 0.6 + np.array([0, 220, 80]) * 0.4).astype(np.uint8)
    overlay_both[neg_mask] = (overlay_both[neg_mask] * 0.6 + np.array([230, 50, 50]) * 0.4).astype(np.uint8)

    # Mark superpixel boundaries in bright cyan/yellow
    boundaries_float = mark_boundaries(overlay_both / 255.0, explanation.segments, color=(1, 0.8, 0))
    overlay_rgb = np.clip(boundaries_float * 255.0, 0, 255).astype(np.uint8)

    # 2. Positive-only view
    temp_pos, mask_pos = explanation.get_image_and_mask(
        pred_index,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    pos_boundaries_float = mark_boundaries(base_img_uint8 / 255.0, mask_pos, color=(0, 1, 0))
    positive_only_rgb = np.clip(pos_boundaries_float * 255.0, 0, 255).astype(np.uint8)

    # Collect feature weights
    top_features = explanation.local_exp.get(pred_index, [])
    features_info = [
        {"superpixel_id": int(sp_id), "weight": round(float(weight), 5)}
        for sp_id, weight in top_features[:num_features]
    ]

    metadata = {
        "explained_class": target_label,
        "class_index": pred_index,
        "confidence": round(confidence, 4),
        "num_samples_evaluated": num_samples,
        "num_superpixels_total": int(np.unique(explanation.segments).size),
        "top_features": features_info,
        "interpretation": f"Green regions positively support prediction '{target_label}'; Red regions oppose it."
    }

    return overlay_rgb, positive_only_rgb, metadata
