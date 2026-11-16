import numpy as np
import tensorflow as tf
from pathlib import Path
from cnnClassifier.utils.image_processing import load_and_preprocess_image
from cnnClassifier.utils.model_loader import ModelLoader
from cnnClassifier import logger


class PredictionPipeline:
    def __init__(self, filename, modality: str = "hyperspectral", model: tf.keras.Model = None):
        self.filename = filename
        self.modality = modality.lower()

        if self.modality not in ["hyperspectral", "multispectral"]:
            raise ValueError("modality must be 'hyperspectral' or 'multispectral'")

        if model is not None:
            self.model = model
            self.model_path = None
        else:
            self.model_path = ModelLoader.get_model_path(self.modality)
            self.model = ModelLoader.load_model(self.modality)

    def predict(self):
        img_array = load_and_preprocess_image(self.filename, target_size=(224, 224))

        prediction = self.model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        class_labels = {0: "Normal", 1: "Tumor"}
        predicted_label = class_labels[predicted_class_index]

        # --- Confidence gate ---
        CONFIDENCE_THRESHOLD = 0.65   # Tuned based on validation distribution

        is_reliable = confidence >= CONFIDENCE_THRESHOLD

        return [{
            "modality": self.modality,
            "prediction": predicted_label if is_reliable else "Uncertain",
            "confidence": round(confidence, 4),
            "reliable": is_reliable,
            "class_probabilities": {
                "Normal": round(float(prediction[0][0]), 4),
                "Tumor": round(float(prediction[0][1]), 4)
            },
            "warning": None if is_reliable else "Low model confidence — image may not resemble training data (non-kidney scan, wrong modality, or poor quality)."
        }]
