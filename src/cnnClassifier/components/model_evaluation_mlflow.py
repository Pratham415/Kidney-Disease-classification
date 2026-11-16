import os
import json
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import mlflow
import mlflow.keras
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.score = None
        self.metrics_summary = {}

    def _valid_generator(self):
        datagenerator_kwargs = dict(validation_split=0.20)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # No rescale=1./255 — EfficientNetV2 handles normalization internally
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self, modality: str = "hyperspectral"):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()

        # 1. Standard evaluation for loss and categorical accuracy
        logger.info(f"Evaluating loss and accuracy for {modality} on validation set...")
        eval_result = self.model.evaluate(self.valid_generator, verbose=1)
        loss, accuracy = float(eval_result[0]), float(eval_result[1])

        # 2. Detailed classification metrics (Precision, Recall, F1, Confusion Matrix)
        logger.info(f"Computing precision, recall, F1-score, and confusion matrix for {modality}...")
        self.valid_generator.reset()
        num_batches = int(np.ceil(self.valid_generator.samples / self.valid_generator.batch_size))

        y_true = []
        y_pred = []
        for _ in range(num_batches):
            x_batch, y_batch = next(self.valid_generator)
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        cm = confusion_matrix(y_true, y_pred).tolist()
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        self.score = [loss, accuracy]
        self.metrics_summary = {
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "validation_samples": int(len(y_true)),
            "evaluation_split": "validation_split_0.20 (No separate test split in repository)"
        }

        logger.info(f"{modality.upper()} Evaluation Results: {self.metrics_summary}")
        self.save_score(modality)

    def save_score(self, modality: str = "hyperspectral"):
        scores_file = Path("scores.json")
        current_scores = {}
        if scores_file.exists():
            try:
                with open(scores_file, "r") as f:
                    current_scores = json.load(f)
            except Exception:
                current_scores = {}

        # Store modality-specific section as well as root-level backward compatible keys
        current_scores[modality] = self.metrics_summary
        current_scores["loss"] = self.score[0]
        current_scores["accuracy"] = self.score[1]

        save_json(path=scores_file, data=current_scores)
        logger.info(f"Updated scores in {scores_file} for modality '{modality}'.")

    def log_into_mlflow(self, modality: str):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', self.config.mlflow_uri)

        if not mlflow_uri:
            logger.warning("MLFLOW_TRACKING_URI is not set. Skipping remote MLflow tracking.")
            return

        mlflow.set_tracking_uri(mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name=f"{modality}_evaluation"):
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1],
                "precision": self.metrics_summary.get("precision", 0.0),
                "recall": self.metrics_summary.get("recall", 0.0),
                "f1_score": self.metrics_summary.get("f1_score", 0.0)
            })

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model, "model", registered_model_name=f"EfficientNetV2B0_{modality}"
                )
            else:
                mlflow.keras.log_model(self.model, "model")
