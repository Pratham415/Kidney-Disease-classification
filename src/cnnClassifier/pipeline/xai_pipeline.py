import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, Union
from cnnClassifier import logger
from cnnClassifier.utils.image_processing import load_and_preprocess_image
from cnnClassifier.utils.model_loader import ModelLoader
from cnnClassifier.utils.xai_gradcam import (
    generate_gradcam_heatmap,
    overlay_heatmap_on_image,
    np_image_to_base64
)
from cnnClassifier.utils.xai_lime import generate_lime_explanation
from cnnClassifier.utils.xai_shap import generate_shap_explanation


class XAIPipeline:
    """
    Unified Explainable AI (XAI) Pipeline providing:
    1. Grad-CAM (Gradient-weighted Class Activation Mapping)
    2. LIME (Local Interpretable Model-Agnostic Explanations)
    3. SHAP (SHapley Additive exPlanations)
    """

    def __init__(
        self,
        filename: Union[str, Path],
        modality: str = "hyperspectral",
        model: Optional[tf.keras.Model] = None
    ):
        self.filename = Path(filename)
        self.modality = modality.lower()

        if self.modality not in ["hyperspectral", "multispectral"]:
            raise ValueError("Modality must be 'hyperspectral' or 'multispectral'")

        if model is not None:
            self.model = model
            self.model_path = None
        else:
            self.model_path = ModelLoader.get_model_path(self.modality)
            self.model = ModelLoader.load_model(self.modality)

        # Load preprocessed image tensor (1, 224, 224, 3)
        self.img_tensor = load_and_preprocess_image(self.filename, target_size=(224, 224))
        # Keep raw uint8 copy of 224x224 RGB image for clean visualization blending
        self.img_rgb_224 = np.clip(self.img_tensor[0], 0, 255).astype(np.uint8)

        # Baseline prediction
        self.predictions = self.model.predict(self.img_tensor, verbose=0)[0]
        self.pred_index = int(np.argmax(self.predictions))
        self.confidence = float(self.predictions[self.pred_index])
        class_labels = {0: "Normal", 1: "Tumor"}
        self.predicted_label = class_labels.get(self.pred_index, "Unknown")

    def explain_gradcam(self, alpha: float = 0.45) -> Dict[str, Any]:
        """Runs Grad-CAM explanation and returns base64 images + interpretation."""
        heatmap, pred_idx, conf = generate_gradcam_heatmap(
            model=self.model,
            img_array=self.img_tensor,
            pred_index=self.pred_index
        )
        overlay_rgb, heatmap_rgb = overlay_heatmap_on_image(
            original_img_rgb=self.img_rgb_224,
            heatmap=heatmap,
            alpha=alpha
        )

        return {
            "method": "Grad-CAM",
            "overlay_image": np_image_to_base64(overlay_rgb),
            "heatmap_image": np_image_to_base64(heatmap_rgb),
            "predicted_class": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "layer_name": "top_activation (EfficientNetV2B0 final conv block)",
            "interpretation": (
                f"Grad-CAM computes gradients of the '{self.predicted_label}' class score with respect "
                "to the final 7x7 convolutional feature maps. Warmer zones (red/yellow) indicate spatial "
                "regions that most strongly activated the CNN filters for this classification."
            ),
            "limitations": (
                "Grad-CAM visualizes coarse activation receptive fields (coarse 7x7 spatial resolution). "
                "It indicates correlation with the neural decision, but does NOT constitute exact clinical "
                "margin demarcation or histological proof of tumor boundaries."
            )
        }

    def explain_lime(self, num_samples: int = 70) -> Dict[str, Any]:
        """Runs LIME explanation and returns base64 images + interpretation."""
        overlay_rgb, positive_only_rgb, metadata = generate_lime_explanation(
            model=self.model,
            img_rgb_224=self.img_rgb_224,
            num_samples=num_samples,
            pred_index=self.pred_index
        )

        return {
            "method": "LIME",
            "overlay_image": np_image_to_base64(overlay_rgb),
            "positive_only_image": np_image_to_base64(positive_only_rgb),
            "predicted_class": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "metadata": metadata,
            "interpretation": (
                f"LIME segments the scan into superpixels and tests {num_samples} local perturbations. "
                f"Green superpixels positively supported the '{self.predicted_label}' decision; "
                "red superpixels opposed it, indicating how local tissue features tilt the prediction."
            ),
            "limitations": (
                "LIME relies on local surrogate linear models and random superpixel sampling. Results may vary "
                "slightly between random seeds and reflect local boundary sensitivities rather than global truth."
            )
        }

    def explain_shap(self, nsamples: int = 40, n_segments: int = 25) -> Dict[str, Any]:
        """Runs SHAP explanation and returns base64 images + interpretation."""
        shap_overlay_rgb, shap_heatmap_rgb, metadata = generate_shap_explanation(
            model=self.model,
            img_rgb_224=self.img_rgb_224,
            n_segments=n_segments,
            nsamples=nsamples,
            pred_index=self.pred_index
        )

        return {
            "method": "SHAP",
            "overlay_image": np_image_to_base64(shap_overlay_rgb),
            "heatmap_image": np_image_to_base64(shap_heatmap_rgb),
            "predicted_class": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "metadata": metadata,
            "interpretation": (
                f"Kernel SHAP computes axiomatic Shapley values across {metadata['num_superpixels']} tissue segments. "
                "Warm (Red) regions denote positive Shapley attributions driving probability towards Tumor; "
                "Cool (Blue) regions denote negative Shapley attributions driving probability towards Normal."
            ),
            "limitations": (
                "Kernel SHAP utilizes an approximation over superpixel coalitions with a baseline background. "
                "High computational cost requires sample budgeting for real-time web applications."
            )
        }

    def explain_all(self) -> Dict[str, Any]:
        """Computes all three XAI explanations for side-by-side comparative analysis."""
        original_b64 = np_image_to_base64(self.img_rgb_224)
        gradcam_res = self.explain_gradcam()
        lime_res = self.explain_lime()
        shap_res = self.explain_shap()

        return {
            "status": "success",
            "modality": self.modality,
            "predicted_class": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "original_image": original_b64,
            "gradcam": gradcam_res,
            "lime": lime_res,
            "shap": shap_res,
            "comparative_analysis": {
                "consensus": (
                    "When Grad-CAM, LIME, and SHAP highlight overlapping anatomical structures, "
                    "confidence in model feature attribution is significantly higher than relying on a single method."
                ),
                "method_differences": (
                    "Grad-CAM evaluates coarse internal conv activations (fastest, white-box). "
                    "LIME evaluates local perturbation sensitivity via linear surrogates (model-agnostic). "
                    "SHAP computes game-theoretic Shapley attributions satisfying axiomatic fairness guarantees."
                ),
                "clinical_disclaimer": (
                    "RESEARCH & EDUCATIONAL USE ONLY: All heatmaps and attributions reflect neural network representations "
                    "and are not certified diagnostic tools, pathology reports, or surgical margins."
                )
            }
        }
