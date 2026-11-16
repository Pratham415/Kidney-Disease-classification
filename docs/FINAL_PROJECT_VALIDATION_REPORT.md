# Final Project Validation & Quality Assurance Report

**Project:** Kidney Disease Classification  
**Date of Report:** November 16, 2026  
**Auditor:** Senior Machine Learning & MLOps Engineer, Python Backend Developer  
**Validation Suite:** 26 automated unit and integration tests (`tests/test_xai_and_pipeline.py`)  
**Execution Outcome:** **26 / 26 PASSED (100% Pass Rate in 13.17s)**  
**Bytecode Verification:** `python -m compileall src app.py main.py` passed with 0 errors.

---

## 1. Features Successfully Verified

| Feature / Subsystem | Status | Verification Detail |
| :--- | :---: | :--- |
| **EfficientNetV2B0 Inference** | Verified | Validated 224x224x3 unscaled raw float32 preprocessing matching native internal Rescaling/Normalization layers. |
| **Model Weight Caching** | Verified | `ModelLoader` singleton in `src/cnnClassifier/utils/model_loader.py` eliminates redundant disk loads and shares models across pipelines. |
| **Fallback Weight Resolution** | Verified | Checks `artifacts/training/{modality}/model.h5` then falls back to `models/{modality}/model.h5`. |
| **Confidence Gate** | Verified | Predictions with $< 0.65$ confidence are safely labeled as `Uncertain` with an alert message. |
| **Grad-CAM Explainability** | Verified | Computes exact gradients via `tf.GradientTape()` on `top_activation` conv layer; generates base64 heatmap & overlay. |
| **LIME Explainability** | Verified | Segments scans into superpixels, samples perturbations, fits linear surrogate, color-codes support/oppose regions. |
| **Kernel SHAP Explainability** | Verified | Axiomatically fair Shapley values computed across superpixel coalitions; Diverging coolwarm colormap visualization. |
| **Unified XAI Pipeline** | Verified | `explain_all()` returns comparative synthesis and consensus analysis without crashing. |
| **Multimodal Architecture** | Verified | Dual-branch Keras model graph constructed: 128-d imaging branch + 32-d clinical branch $\rightarrow$ 160-d late fusion. |
| **Clinical Biomarker Validation**| Verified | Physiological range boundaries enforced; missing values imputed with medical baselines; z-score scaling applied. |
| **Flask REST & Web App** | Verified | Endpoints `/`, `/health`, `/predict`, `/explain`, `/multimodal/schema`, `/multimodal/validate` verified. |
| **Concurrent File Safety** | Verified | Unique temporary UUIDs prevent race condition overwrites on image uploads. |
| **Security Token Sanitization** | Verified | Compromised DagsHub access tokens removed from active repo files (`research/trials.ipynb`, `egg-info`). |
| **Cross-Platform Dependencies** | Verified | PEP 508 environment markers in `requirements.txt` cleanly support Apple Silicon macOS and Linux/Docker. |
| **CI/CD Automation** | Verified | GitHub Actions `.github/workflows/main.yaml` upgraded to run real automated tests and compileall before deployment. |

---

## 2. Features Partially Implemented & Dataset Limitations

1. **Multimodal Learning (Experimental Architectural Prototype):**
   - **Status:** Architectural specification, dual-branch Keras model graph, and clinical boundary validation are fully implemented and tested.
   - **Limitation:** The current repository dataset contains only 2D CT scan images (20,000 images) and lacks paired electronic health record (EHR) tabular patient records.
   - **Integrity Compliance:** In accordance with scientific ethics, **no fake patient data was created** and no ungrounded claims of multimodal performance improvements are made. The model remains an experimental prototype ready for paired EHR cohort ingestion.

2. **Holdout Test Set Absence:**
   - **Status:** The dataset contains `normal` and `tumor` directories split 80/20 into training and validation sets.
   - **Limitation:** There is no independent holdout test set from external hospital cohorts. All evaluation metrics are strictly reported as **Validation Split Metrics**.

---

## 3. Features Not Tested (Out of Scope for Local Run)

- **AWS ECR & EC2 Automated Deployment:** Self-hosted runner steps in `.github/workflows/main.yaml` require live AWS IAM credentials and a running EC2 instance. The workflow definitions were upgraded and validated for syntax and dependency correctness.
- **Physical Retraining of Full 10 Epochs:** Retraining from scratch takes ~2 hours and was not required because verified production weights already exist with solid validation scores.

---

## 4. Comprehensive Model Evaluation Results

### Validation Split Benchmark (Saved Production Weights)
Evaluated on a balanced validation split (640 samples per modality, 320 Normal, 320 Tumor):

| Metric | Hyperspectral Modality | Multispectral Modality | Clinical Diagnostic Impact |
| :--- | :---: | :---: | :--- |
| **Loss** | `0.2096` | `0.3812` | Lower entropy on hyperspectral scans. |
| **Accuracy** | **94.22%** | **83.75%** | Hyperspectral exhibits superior overall accuracy. |
| **Precision (Tumor)** | **100.00%** | **78.57%** | Hyperspectral achieved zero false positives on evaluation batch. |
| **Recall (Tumor)** | **88.44%** | **92.81%** | Multispectral captures more tumors (higher sensitivity). |
| **F1-Score (Tumor)** | **93.86%** | **85.10%** | Robust harmonic balance across both modalities. |

---

## 5. Security Findings & History Remediation Advice

### Findings:
1. **Active Files:** A DagsHub MLflow token (`fbea9abf7cb94ad9ddad624ef8b47700d5f8f126`) was found in `research/trials.ipynb` and `6824692c47a369aa6f9eac5b10041d5c8edbcef0` in `src/CNN_Classifier.egg-info/PKG-INFO`.
   - **Remediation Action:** All active working tree occurrences have been replaced with placeholders (`<YOUR_DAGSHUB_TOKEN>`).
2. **Git Commit History:** The tokens persist in past commit objects:
   - Commit `4a5fd9dcf33d3664030b5f7bec9a21a599d0ac50` (`test_mlflow_connection.py`)
   - Commit `e711c69a7e4e3f63386e8be6fd9efe68634e2b99` (`README.md`)

### Mandatory Security Recommendation:
1. **Immediate Token Revocation:** Log in to [DagsHub](https://dagshub.com) $\rightarrow$ User Settings $\rightarrow$ Tokens, and revoke the compromised token. Generate a new token and store it exclusively in your local `.env` file.
2. **Optional Git History Rewriting:** If the repository is public and you wish to purge the historic commit blobs, tools such as `git-filter-repo` or BFG Repo-Cleaner can purge the history. Because this requires rewriting Git commit hashes and force-pushing, it must only be executed with explicit approval.

---

## 6. Recommended Next Steps

1. **Paired Clinical Data Collaboration:** Partner with a clinical nephrology department to curate a de-identified cohort of paired CT scans and lab records (Creatinine, BUN, eGFR) to train and evaluate the multimodal classifier.
2. **3D Volumetric Extensibility:** Extend the 2D slice classifier to 3D convolutional architectures (e.g. 3D ResNet or Med3D) for volumetric DICOM CT scan evaluation.
3. **External Hospital Validation:** Validate the existing EfficientNetV2B0 models on an external medical center's kidney imaging repository to evaluate out-of-distribution generalizability.
