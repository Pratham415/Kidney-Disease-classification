import os
from pathlib import Path
from typing import Dict, Optional
import tensorflow as tf
from cnnClassifier import logger


class ModelLoader:
    """
    Centralized, thread-safe model loader and cache.
    Ensures that heavy Keras models are loaded only once per modality
    and shared across PredictionPipeline and XAIPipeline.
    Supports automatic fallback between artifacts and models directory.
    """
    _cache: Dict[str, tf.keras.Model] = {}

    @classmethod
    def get_model_path(cls, modality: str) -> Path:
        """
        Resolves model file path with priority:
        1. artifacts/training/{modality}/model.h5
        2. models/{modality}/model.h5
        """
        modality = modality.lower()
        candidate_paths = [
            Path(f"artifacts/training/{modality}/model.h5"),
            Path(f"models/{modality}/model.h5")
        ]

        for p in candidate_paths:
            if p.exists() and p.is_file():
                return p

        searched = ", ".join(f"'{p}'" for p in candidate_paths)
        raise FileNotFoundError(
            f"Trained model for modality '{modality}' was not found. Searched locations: {searched}. "
            "Please ensure model weights are placed in either artifacts/training/ or models/."
        )

    @classmethod
    def load_model(cls, modality: str, force_reload: bool = False) -> tf.keras.Model:
        """
        Retrieves cached model or loads it from disk if not yet cached.
        """
        modality = modality.lower()
        if modality not in ["hyperspectral", "multispectral"]:
            raise ValueError(f"Invalid modality '{modality}'. Expected 'hyperspectral' or 'multispectral'.")

        model_path = cls.get_model_path(modality)
        cache_key = f"{modality}::{model_path.resolve()}"

        if force_reload or cache_key not in cls._cache:
            logger.info(f"Loading {modality} model weights into cache from '{model_path}'...")
            cls._cache[cache_key] = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully cached {modality} model.")

        return cls._cache[cache_key]

    @classmethod
    def clear_cache(cls):
        """Clears the in-memory model cache."""
        cls._cache.clear()
        logger.info("Cleared in-memory model cache.")
