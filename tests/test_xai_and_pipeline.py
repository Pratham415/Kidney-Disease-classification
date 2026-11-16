import os
import sys
import base64
import tempfile
import numpy as np
import pytest
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

from cnnClassifier.utils.image_processing import load_and_preprocess_image
from cnnClassifier.utils.model_loader import ModelLoader
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.pipeline.xai_pipeline import XAIPipeline
from cnnClassifier.components.multimodal import (
    build_multimodal_kidney_classifier,
    CLINICAL_FEATURE_NAMES,
    CLINICAL_FEATURE_SPECS
)
from cnnClassifier.pipeline.multimodal_pipeline import MultimodalPrototypePipeline
from app import app

TEST_IMAGE = "inputImage.jpg"


# ============================================================================
# UNIT TESTS: Image Preprocessing & Utility Validation
# ============================================================================

def test_image_preprocessing_valid_shape_and_type():
    """Verifies image preprocessor produces (1, 224, 224, 3) float32 tensor."""
    img_tensor = load_and_preprocess_image(TEST_IMAGE, target_size=(224, 224))
    assert img_tensor.shape == (1, 224, 224, 3)
    assert img_tensor.dtype == np.float32
    # EfficientNetV2 expects unscaled [0, 255] pixels
    assert np.max(img_tensor) > 1.0


def test_image_preprocessing_nonexistent_file():
    """Verifies FileNotFoundError on nonexistent image path."""
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_image("nonexistent_medical_scan_123.jpg")


def test_image_preprocessing_empty_file():
    """Verifies ValueError when attempting to process a 0-byte file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        with pytest.raises(ValueError, match="empty"):
            load_and_preprocess_image(tmp.name)


def test_decode_image_with_data_uri_header():
    """Verifies decodeImage strips 'data:image/jpeg;base64,' prefix properly."""
    with open(TEST_IMAGE, "rb") as f:
        raw_b64 = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{raw_b64}"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        decodeImage(data_uri, tmp_name)
        assert os.path.exists(tmp_name)
        assert os.path.getsize(tmp_name) > 0
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# ============================================================================
# UNIT TESTS: Model Loader & Caching
# ============================================================================

def test_model_loader_caching():
    """Verifies ModelLoader returns identical cached instances on repeated calls."""
    m1 = ModelLoader.load_model("hyperspectral")
    m2 = ModelLoader.load_model("hyperspectral")
    assert m1 is m2


def test_model_loader_invalid_modality():
    """Verifies ValueError when requesting unknown modality."""
    with pytest.raises(ValueError):
        ModelLoader.load_model("xray_unknown")


# ============================================================================
# UNIT TESTS: Prediction Pipeline & Confidence Logic
# ============================================================================

def test_prediction_pipeline_hyperspectral():
    """Verifies prediction format, label mapping, and confidence score."""
    pipeline = PredictionPipeline(TEST_IMAGE, modality="hyperspectral")
    results = pipeline.predict()
    assert isinstance(results, list)
    assert len(results) == 1
    res = results[0]
    assert res["modality"] == "hyperspectral"
    assert res["prediction"] in ["Normal", "Tumor", "Uncertain"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "class_probabilities" in res
    assert "Normal" in res["class_probabilities"]
    assert "Tumor" in res["class_probabilities"]


def test_prediction_pipeline_multispectral():
    """Verifies multispectral prediction pipeline."""
    pipeline = PredictionPipeline(TEST_IMAGE, modality="multispectral")
    results = pipeline.predict()
    assert isinstance(results, list)
    res = results[0]
    assert res["modality"] == "multispectral"
    assert res["prediction"] in ["Normal", "Tumor", "Uncertain"]


def test_prediction_confidence_gate_uncertain():
    """Verifies confidence gate flags prediction as 'Uncertain' when confidence < 0.65."""
    # Mock model outputting 51% Normal vs 49% Tumor (uncertain prediction)
    mock_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Lambda(lambda x: tf.constant([[0.51, 0.49]], dtype=tf.float32))
    ])
    pipeline = PredictionPipeline(TEST_IMAGE, modality="hyperspectral", model=mock_model)
    results = pipeline.predict()
    res = results[0]
    assert res["prediction"] == "Uncertain"
    assert res["reliable"] is False
    assert res["warning"] is not None


# ============================================================================
# INTEGRATION TESTS: Explainable AI (Grad-CAM, LIME, SHAP)
# ============================================================================

def test_xai_gradcam():
    """Verifies Grad-CAM heatmap, overlay, and layer targeting."""
    xai = XAIPipeline(TEST_IMAGE, modality="hyperspectral")
    res = xai.explain_gradcam(alpha=0.5)
    assert res["method"] == "Grad-CAM"
    assert res["overlay_image"].startswith("data:image/png;base64,")
    assert res["heatmap_image"].startswith("data:image/png;base64,")
    assert "top_activation" in res["layer_name"]
    assert res["confidence"] >= 0.0


def test_xai_lime():
    """Verifies LIME superpixel perturbation and metadata generation."""
    xai = XAIPipeline(TEST_IMAGE, modality="hyperspectral")
    res = xai.explain_lime(num_samples=25)
    assert res["method"] == "LIME"
    assert res["overlay_image"].startswith("data:image/png;base64,")
    assert "metadata" in res
    assert res["metadata"]["num_samples_evaluated"] == 25
    assert len(res["metadata"]["top_features"]) > 0


def test_xai_shap():
    """Verifies Kernel SHAP game-theoretic attribution explanation."""
    xai = XAIPipeline(TEST_IMAGE, modality="hyperspectral")
    res = xai.explain_shap(nsamples=20, n_segments=10)
    assert res["method"] == "SHAP"
    assert res["overlay_image"].startswith("data:image/png;base64,")
    assert res["heatmap_image"].startswith("data:image/png;base64,")
    assert "metadata" in res
    assert res["metadata"]["num_superpixels"] > 0


def test_xai_explain_all():
    """Verifies unified explain_all pipeline returns all 3 methods."""
    xai = XAIPipeline(TEST_IMAGE, modality="hyperspectral")
    # Use light settings to run quickly
    res = xai.explain_all()
    assert res["status"] == "success"
    assert "gradcam" in res
    assert "lime" in res
    assert "shap" in res
    assert "comparative_analysis" in res


# ============================================================================
# UNIT TESTS: Multimodal Architecture & Clinical Validation
# ============================================================================

def test_multimodal_architecture_construction():
    """Verifies dual-branch model graph has correct input shapes and output dimensions."""
    model = build_multimodal_kidney_classifier(
        image_input_shape=(224, 224, 3),
        num_clinical_features=len(CLINICAL_FEATURE_NAMES),
        num_classes=2
    )
    assert len(model.inputs) == 2
    assert model.inputs[0].shape.as_list() == [None, 224, 224, 3]
    assert model.inputs[1].shape.as_list() == [None, len(CLINICAL_FEATURE_NAMES)]
    assert model.outputs[0].shape.as_list() == [None, 2]


def test_multimodal_clinical_validation_valid():
    """Verifies clinical biomarker parsing and z-score standardization."""
    pipeline = MultimodalPrototypePipeline()
    valid_data = {
        "age": 62,
        "serum_creatinine_mg_dl": 1.4,
        "blood_urea_nitrogen_mg_dl": 22.0,
        "egfr_ml_min": 75.0,
        "systolic_bp_mmhg": 135.0,
        "hemoglobin_g_dl": 13.8
    }
    tensor, meta = pipeline.validate_clinical_inputs(valid_data, scale=True)
    assert tensor.shape == (1, 6)
    assert tensor.dtype == np.float32
    assert len(meta["imputed_features"]) == 0
    assert len(meta["scaled_values"]) == 6


def test_multimodal_clinical_validation_missing_imputation():
    """Verifies missing biomarkers are imputed using baseline physiological values."""
    pipeline = MultimodalPrototypePipeline()
    partial_data = {"age": 45}  # Missing 5 biomarkers
    tensor, meta = pipeline.validate_clinical_inputs(partial_data, scale=True)
    assert tensor.shape == (1, 6)
    assert len(meta["imputed_features"]) == 5
    assert "serum_creatinine_mg_dl" in meta["imputed_features"]


def test_multimodal_clinical_validation_out_of_range():
    """Verifies ValueError when clinical inputs exceed physiological limits."""
    pipeline = MultimodalPrototypePipeline()
    # Age -5 is impossible
    with pytest.raises(ValueError, match="out-of-range"):
        pipeline.validate_clinical_inputs({"age": -5})

    # Systolic BP 500 mmHg is non-viable
    with pytest.raises(ValueError, match="out-of-range"):
        pipeline.validate_clinical_inputs({"systolic_bp_mmhg": 500})


def test_multimodal_clinical_validation_non_numeric():
    """Verifies ValueError when clinical input cannot be cast to float."""
    pipeline = MultimodalPrototypePipeline()
    with pytest.raises(ValueError, match="Invalid clinical input"):
        pipeline.validate_clinical_inputs({"age": "not_a_number"})


# ============================================================================
# INTEGRATION TESTS: Flask Web Application Routes & Error Handling
# ============================================================================

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_flask_home_route(client):
    """Verifies GET / loads the HTML console."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Renal Spectral Diagnostic Console" in response.data or b"html" in response.data


def test_flask_health_route(client):
    """Verifies GET /health returns status 200 with service metadata and disclaimer."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "disclaimer" in data


def test_flask_multimodal_schema_route(client):
    """Verifies GET /multimodal/schema returns architecture specification."""
    response = client.get("/multimodal/schema")
    assert response.status_code == 200
    data = response.get_json()
    assert "clinical_features" in data
    assert len(data["clinical_features"]) == 6


def test_flask_multimodal_validate_route(client):
    """Verifies POST /multimodal/validate endpoint with valid and invalid data."""
    valid_payload = {
        "age": 58,
        "serum_creatinine_mg_dl": 1.2,
        "blood_urea_nitrogen_mg_dl": 16.0,
        "egfr_ml_min": 88.0,
        "systolic_bp_mmhg": 122.0,
        "hemoglobin_g_dl": 14.5
    }
    res = client.post("/multimodal/validate", json={"clinical_data": valid_payload})
    assert res.status_code == 200
    assert res.get_json()["status"] == "valid"

    # Out-of-bounds test
    invalid_payload = {"age": 180}
    res_bad = client.post("/multimodal/validate", json={"clinical_data": invalid_payload})
    assert res_bad.status_code == 400
    assert "error" in res_bad.get_json()


def test_flask_predict_api_valid(client):
    """Verifies POST /predict route with valid base64 image."""
    with open(TEST_IMAGE, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.post("/predict", json={"image": b64, "modality": "hyperspectral"})
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert "prediction" in data[0]
    assert "disclaimer" in data[0]


def test_flask_predict_api_missing_image(client):
    """Verifies POST /predict returns 400 when image is omitted."""
    response = client.post("/predict", json={"modality": "hyperspectral"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_flask_predict_api_invalid_modality(client):
    """Verifies POST /predict returns 400 when modality is invalid."""
    with open(TEST_IMAGE, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    response = client.post("/predict", json={"image": b64, "modality": "mri_invalid"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_flask_explain_api_gradcam(client):
    """Verifies POST /explain route with Grad-CAM method."""
    with open(TEST_IMAGE, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.post("/explain", json={
        "image": b64,
        "modality": "hyperspectral",
        "method": "gradcam"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["method"] == "Grad-CAM"
    assert "overlay_image" in data
    assert "disclaimer" in data
