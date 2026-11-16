# Multimodal Deep Learning Architecture & Clinical Biomarker Integration

## 1. Clinical Motivation

While computed tomography (CT) imaging provides essential macroscopic spatial visualization of renal structures and tumors, nephrologists and oncologists rely heavily on structured laboratory panels to assess renal physiological health. 

Isolated image-based classification overlooks critical physiological variables:
- Elevated serum creatinine and Blood Urea Nitrogen (BUN) indicate compromised renal filtration.
- Depressed estimated Glomerular Filtration Rate (eGFR) quantifies Chronic Kidney Disease (CKD) staging.
- Systolic hypertension and anemia (low hemoglobin) correlate with vascular renal pathologies.

To advance diagnostic accuracy toward holistic clinical intelligence, this project defines a **Dual-Branch Multimodal Fusion Architecture** combining CT imaging scans with six standard renal biomarkers.

---

## 2. Architectural Blueprint

```mermaid
flowchart TD
    subgraph Branch A: CT Scan Image (224x224x3)
        I1["Input CT Tensor\n(None, 224, 224, 3)"] --> I2["EfficientNetV2B0\n(Frozen Pretrained Feature Extractor)"]
        I2 --> I3["GlobalAveragePooling2D\n(None, 1280)"]
        I3 --> I4["Dense(128, activation='relu')\n(None, 128)"]
        I4 --> I5["Dropout(rate=0.2)\n(None, 128)"]
    end

    subgraph Branch B: Clinical Biomarkers (6 Features)
        C1["Input Tabular Biomarkers\n(None, 6)"] --> C2["Dense(64, activation='relu')\n(None, 64)"]
        C2 --> C3["BatchNormalization()\n(Stabilizes feature distribution)"]
        C3 --> C4["Dropout(rate=0.2)\n(None, 64)"]
        C4 --> C5["Dense(32, activation='relu')\n(None, 32)"]
    end

    subgraph Late Feature Fusion & Classification
        I5 --> F1["Feature Concatenation\n(128 + 32 = 160 dimensions)"]
        C5 --> F1
        F1 --> F2["Dense(64, activation='relu')\n(None, 64)"]
        F2 --> F3["Dropout(rate=0.3)\n(None, 64)"]
        F3 --> F4["Dense(2, activation='softmax')\n(None, 2)"]
    end
```

---

## 3. Clinical Biomarker Specification & Normalization Schema

The clinical branch processes six physiological features with strict boundary validation and z-score scaling:

| Feature Name | Unit | Biological Valid Range | Reference Baseline | Mean ($\mu$) | Std ($\sigma$) | Clinical Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `age` | Years | $[1.0, 120.0]$ | $55.0$ | $55.0$ | $15.0$ | Renal cell carcinoma incidence increases with age. |
| `serum_creatinine_mg_dl` | mg/dL | $[0.1, 25.0]$ | $1.1$ | $1.2$ | $0.8$ | Primary biomarker of glomerular clearance breakdown. |
| `blood_urea_nitrogen_mg_dl` | mg/dL | $[1.0, 150.0]$ | $15.0$ | $18.0$ | $10.0$ | Marker of nitrogen retention and renal hypoperfusion. |
| `egfr_ml_min` | mL/min/1.73m² | $[0.0, 200.0]$ | $85.0$ | $80.0$ | $25.0$ | Standard index of overall renal filtration function. |
| `systolic_bp_mmhg` | mmHg | $[50.0, 260.0]$ | $120.0$ | $125.0$ | $18.0$ | Secondary hypertension often signals renal parenchyma ischemia. |
| `hemoglobin_g_dl` | g/dL | $[3.0, 25.0]$ | $14.0$ | $14.0$ | $2.0$ | Anemia correlates with erythropoietin deficiency in renal disease. |

### 3.1 Normalization Equation
To avoid gradients being dominated by features with large absolute ranges (e.g. eGFR up to 200 vs creatinine ~1.0), all inputs undergo z-score standardization:
$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

---

## 4. Current Dataset Status & Scientific Transparency Statement

> **CRITICAL CLINICAL & METHODOLOGICAL DISCLOSURE:**  
> The current public repository dataset contains exclusively imaging data (20,000 CT scans). **No paired electronic health record (EHR) tabular patient records exist in the current public release.**

In strict adherence to scientific ethics:
- **No synthetic or fake patient data has been fabricated** to train the multimodal model.
- **No artificial performance claims** (e.g. claiming multimodal yields "+5% accuracy") are reported.
- The multimodal pipeline in `src/cnnClassifier/components/multimodal.py` and `src/cnnClassifier/pipeline/multimodal_pipeline.py` serves as a **verified architectural specification, validation schema, and ready-to-train model graph**.

### 4.1 Prerequisites for Production Multimodal Training
Before this multimodal classifier can be trained and evaluated:
1. **Paired Cohort Acquisition:** Ingestion of de-identified DICOM CT volumes paired with confirmed laboratory panels from hospital electronic health record systems.
2. **IRB / Ethics Clearance:** Full institutional ethical approval for multimodal health data utilization.
3. **Missing Value Protocol:** Handling real-world clinical data sparsity through multiple imputation by chained equations (MICE) rather than naive zero-filling.
