import os
import mlflow
import mlflow.keras
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.comman import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

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

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    import os   # add this import at top of file if not already there

    def log_into_mlflow(self, modality: str):
        os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/Pratham415/Kidney-Disease-classification.mlflow'
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'Pratham415'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = 'bfa6a840bebc6c0209bfc000f3bd10b4491e8hgf'

        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name=f"{modality}_evaluation"):
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model, "model", registered_model_name=f"EfficientNetV2B0_{modality}"
                )
            else:
                mlflow.keras.log_model(self.model, "model")