import numpy as np
import tensorflow as tf
from pathlib import Path


class PredictionPipeline:
    def __init__(self, filename, modality: str = "hyperspectral"):
        self.filename = filename
        self.modality = modality.lower()

        if self.modality not in ["hyperspectral", "multispectral"]:
            raise ValueError("modality must be 'hyperspectral' or 'multispectral'")

        self.model_path = Path(f"artifacts/training/{self.modality}/model.h5")
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self):
        img = tf.keras.preprocessing.image.load_img(
            self.filename, target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        class_labels = {0: "Normal", 1: "Tumor"}
        predicted_label = class_labels[predicted_class_index]

        # --- Confidence gate ---
        CONFIDENCE_THRESHOLD = 0.65   # tune this based on your validation results

        is_reliable = confidence >= CONFIDENCE_THRESHOLD

        return [{
            "modality": self.modality,
            "prediction": predicted_label if is_reliable else "Uncertain",
            "confidence": round(confidence, 4),
            "reliable": is_reliable,
            "warning": None if is_reliable else "Low model confidence — image may not resemble training data (non-kidney scan, wrong modality, or poor quality)."
        }]