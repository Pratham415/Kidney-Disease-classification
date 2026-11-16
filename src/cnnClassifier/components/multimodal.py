import tensorflow as tf
from typing import List, Dict, Any, Tuple
from cnnClassifier import logger


# Verified standard renal clinical biomarkers for multimodal fusion
CLINICAL_FEATURE_NAMES = [
    "age",                       # Patient age in years
    "serum_creatinine_mg_dl",    # Serum Creatinine (normal ~0.7 - 1.3 mg/dL)
    "blood_urea_nitrogen_mg_dl", # BUN (normal ~7 - 20 mg/dL)
    "egfr_ml_min",               # Estimated Glomerular Filtration Rate (normal >= 90)
    "systolic_bp_mmhg",          # Systolic Blood Pressure (normal < 120 mmHg)
    "hemoglobin_g_dl"            # Hemoglobin level (normal 13.5 - 17.5 g/dL)
]

# Clinical biomarker reference statistics and physiological validation boundaries
CLINICAL_FEATURE_SPECS = {
    "age": {
        "unit": "years",
        "min": 1.0,
        "max": 120.0,
        "default": 55.0,
        "mean": 55.0,
        "std": 15.0,
        "description": "Patient age in completed years"
    },
    "serum_creatinine_mg_dl": {
        "unit": "mg/dL",
        "min": 0.1,
        "max": 25.0,
        "default": 1.1,
        "mean": 1.2,
        "std": 0.8,
        "description": "Serum creatinine level"
    },
    "blood_urea_nitrogen_mg_dl": {
        "unit": "mg/dL",
        "min": 1.0,
        "max": 150.0,
        "default": 15.0,
        "mean": 18.0,
        "std": 10.0,
        "description": "Blood urea nitrogen (BUN)"
    },
    "egfr_ml_min": {
        "unit": "mL/min/1.73m^2",
        "min": 0.0,
        "max": 200.0,
        "default": 85.0,
        "mean": 80.0,
        "std": 25.0,
        "description": "Estimated Glomerular Filtration Rate"
    },
    "systolic_bp_mmhg": {
        "unit": "mmHg",
        "min": 50.0,
        "max": 260.0,
        "default": 120.0,
        "mean": 125.0,
        "std": 18.0,
        "description": "Systolic blood pressure"
    },
    "hemoglobin_g_dl": {
        "unit": "g/dL",
        "min": 3.0,
        "max": 25.0,
        "default": 14.0,
        "mean": 14.0,
        "std": 2.0,
        "description": "Total blood hemoglobin concentration"
    }
}


def build_multimodal_kidney_classifier(
    image_input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_clinical_features: int = 6,
    num_classes: int = 2,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    r"""
    Constructs a dual-branch Multimodal Fusion Architecture for Renal Disease Diagnosis.

    Architectural Schema:
    --------------------------------------------------------------------------
    [Branch A: CT Imaging Scan]               [Branch B: Clinical Biomarkers]
         Input (224, 224, 3)                         Input (6 tabular values)
                 |                                               |
       EfficientNetV2B0 Backbone                        Dense (64, ReLU)
                 |                                               |
      GlobalAveragePooling2D (1280)                     BatchNormalization
                 |                                               |
         Dense (128, ReLU)                                Dropout (0.2)
                 |                                               |
           Dropout (0.2)                                Dense (32, ReLU)
                 \                                             /
                  \                                           /
                   ------> [Feature Fusion Layer] <-----------
                               Concatenate (160)
                                      |
                              Dense (64, ReLU)
                                      |
                                Dropout (0.3)
                                      |
                           Dense (2, Softmax Output)
    --------------------------------------------------------------------------

    Returns:
        tf.keras.Model: Dual-input multimodal Keras model.
    """
    # 1. Imaging Branch
    image_input = tf.keras.layers.Input(shape=image_input_shape, name="image_input")
    base_cnn = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        input_tensor=image_input,
        weights="imagenet",
        include_top=False
    )
    # Freeze base CNN feature extraction layers
    base_cnn.trainable = False

    x_img = tf.keras.layers.GlobalAveragePooling2D(name="img_global_pool")(base_cnn.output)
    x_img = tf.keras.layers.Dense(128, activation="relu", name="img_dense_128")(x_img)
    x_img = tf.keras.layers.Dropout(0.2, name="img_dropout")(x_img)

    # 2. Structured Tabular Clinical Branch
    clinical_input = tf.keras.layers.Input(shape=(num_clinical_features,), name="clinical_input")
    x_clin = tf.keras.layers.Dense(64, activation="relu", name="clin_dense_64")(clinical_input)
    x_clin = tf.keras.layers.BatchNormalization(name="clin_bn")(x_clin)
    x_clin = tf.keras.layers.Dropout(0.2, name="clin_dropout")(x_clin)
    x_clin = tf.keras.layers.Dense(32, activation="relu", name="clin_dense_32")(x_clin)

    # 3. Multimodal Feature Fusion
    fusion = tf.keras.layers.Concatenate(name="multimodal_fusion")([x_img, x_clin])
    x_fused = tf.keras.layers.Dense(64, activation="relu", name="fusion_dense_64")(fusion)
    x_fused = tf.keras.layers.Dropout(0.3, name="fusion_dropout")(x_fused)

    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction_output")(x_fused)

    model = tf.keras.Model(
        inputs=[image_input, clinical_input],
        outputs=output,
        name="Multimodal_Kidney_Classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model
