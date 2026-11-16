import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from cnnClassifier import logger
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.utils.model_loader import ModelLoader
from cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = Path("artifacts/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_IMAGE_PATH = "inputImage.jpg"

CLINICAL_DISCLAIMER = (
    "RESEARCH & EDUCATIONAL PROTOTYPE ONLY: This software is intended solely for scientific "
    "demonstration, research, and educational purposes. It is NOT a certified medical diagnostic "
    "device and does not provide clinical diagnoses. Findings must always be reviewed by a "
    "licensed medical professional."
)


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/health", methods=['GET'])
def healthRoute():
    return jsonify({
        "status": "healthy",
        "service": "Kidney Disease Classification API",
        "supported_modalities": ["hyperspectral", "multispectral"],
        "xai_methods": ["Grad-CAM", "LIME", "Kernel SHAP"],
        "multimodal_status": "Experimental Architecture Specification",
        "disclaimer": CLINICAL_DISCLAIMER
    }), 200


@app.route("/train", methods=['GET'])
def trainRoute():
    try:
        os.system("dvc repro")
        return jsonify({"message": "Training pipeline completed successfully via DVC!"}), 200
    except Exception as e:
        logger.exception(f"Training pipeline execution failed: {e}")
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route("/predict", methods=['POST'])
def predictRoute():
    temp_path = None
    try:
        data = request.get_json(silent=True)

        if not data or 'image' not in data:
            return jsonify({"error": "No image payload provided in JSON request."}), 400

        image_str = data['image']
        if not image_str or not isinstance(image_str, str):
            return jsonify({"error": "Image payload must be a non-empty base64 string."}), 400

        modality = data.get('modality', 'hyperspectral').lower()
        if modality not in ['hyperspectral', 'multispectral']:
            return jsonify({"error": "Modality must be 'hyperspectral' or 'multispectral'"}), 400

        # Unique file handling to prevent concurrent upload collisions
        req_id = uuid.uuid4().hex[:8]
        temp_path = UPLOAD_DIR / f"upload_{req_id}.jpg"

        decodeImage(image_str, str(temp_path))
        # Also maintain inputImage.jpg for backward compatibility
        decodeImage(image_str, DEFAULT_IMAGE_PATH)

        classifier = PredictionPipeline(temp_path, modality=modality)
        result = classifier.predict()

        # Append disclaimer
        for r in result:
            r["disclaimer"] = CLINICAL_DISCLAIMER

        return jsonify(result), 200

    except (ValueError, FileNotFoundError) as e:
        logger.warning(f"Prediction input validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"Prediction processing error: {e}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.route("/explain", methods=['POST'])
def explainRoute():
    temp_path = None
    try:
        from cnnClassifier.pipeline.xai_pipeline import XAIPipeline

        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided in JSON request."}), 400

        image_str = data['image']
        if not image_str or not isinstance(image_str, str):
            return jsonify({"error": "Image payload must be a non-empty base64 string."}), 400

        modality = data.get('modality', 'hyperspectral').lower()
        method = data.get('method', 'all').lower()

        if modality not in ['hyperspectral', 'multispectral']:
            return jsonify({"error": "Modality must be 'hyperspectral' or 'multispectral'"}), 400

        req_id = uuid.uuid4().hex[:8]
        temp_path = UPLOAD_DIR / f"xai_{req_id}.jpg"
        decodeImage(image_str, str(temp_path))
        decodeImage(image_str, DEFAULT_IMAGE_PATH)

        xai = XAIPipeline(temp_path, modality=modality)

        if method == "gradcam":
            alpha = float(data.get('alpha', 0.45))
            result = xai.explain_gradcam(alpha=alpha)
        elif method == "lime":
            samples = int(data.get('samples', 60))
            result = xai.explain_lime(num_samples=samples)
        elif method == "shap":
            samples = int(data.get('samples', 35))
            result = xai.explain_shap(nsamples=samples)
        else:
            result = xai.explain_all()

        result["disclaimer"] = CLINICAL_DISCLAIMER
        return jsonify(result), 200

    except (ValueError, FileNotFoundError) as e:
        logger.warning(f"XAI input validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"XAI computation error: {e}")
        return jsonify({"error": f"XAI computation failed: {str(e)}"}), 500
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.route("/multimodal/schema", methods=['GET'])
def multimodalSchemaRoute():
    try:
        from cnnClassifier.pipeline.multimodal_pipeline import MultimodalPrototypePipeline
        pipeline = MultimodalPrototypePipeline()
        summary = pipeline.get_architecture_summary()
        summary["disclaimer"] = CLINICAL_DISCLAIMER
        return jsonify(summary), 200
    except Exception as e:
        logger.exception(f"Multimodal schema error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/multimodal/validate", methods=['POST'])
def multimodalValidateRoute():
    """Validates and scales clinical biomarker inputs without fabricating patient diagnoses."""
    try:
        from cnnClassifier.pipeline.multimodal_pipeline import MultimodalPrototypePipeline
        data = request.get_json(silent=True) or {}
        clinical_data = data.get("clinical_data", data)
        pipeline = MultimodalPrototypePipeline()
        tensor, meta = pipeline.validate_clinical_inputs(clinical_data, scale=True)
        return jsonify({
            "status": "valid",
            "feature_tensor_shape": list(tensor.shape),
            "metadata": meta,
            "disclaimer": CLINICAL_DISCLAIMER
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
