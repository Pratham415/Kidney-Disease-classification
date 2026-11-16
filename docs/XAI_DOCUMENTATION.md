# Explainable Artificial Intelligence (XAI) Documentation

## 1. Executive Summary & Rationale

In high-stakes clinical AI deployment, deep neural networks are often criticized as opaque "black boxes." A high accuracy score without interpretability fails to satisfy regulatory and clinical standards:
- It obscures dataset bias and "Clever Hans" effects (e.g. models learning hospital-specific watermarks, scan borders, or background couch markers).
- It provides no transparent validation for radiologists reviewing AI output.
- It prevents doctors from verifying anatomical tumor margins.

To address these challenges, this project incorporates **Triangulated Explainable AI**, implementing three complementary interpretability algorithms:
1. **Grad-CAM (Gradient-weighted Class Activation Mapping):** White-box internal feature map activation.
2. **LIME (Local Interpretable Model-Agnostic Explanations):** Model-agnostic superpixel perturbation.
3. **Kernel SHAP (SHapley Additive exPlanations):** Axiomatic cooperative game theory.

---

## 2. Methodology A: Grad-CAM

- **File:** `src/cnnClassifier/utils/xai_gradcam.py`
- **Method Category:** White-Box, Gradient-Based Saliency Mapping
- **Target Convolutional Layer:** `top_activation` (EfficientNetV2B0 final conv stage, shape: $7 \times 7 \times 1280$).

### 2.1 Mathematical Formulation
1. Let $y^c$ be the raw score (logit or softmax output) for class $c$ (where $c \in \{\text{Normal}, \text{Tumor}\}$).
2. Let $A^k_{i, j}$ be the activation of channel $k$ in the target feature map at spatial location $(i, j)$ of size $7 \times 7$.
3. Gradients of the class score with respect to feature maps are computed using TensorFlow's `tf.GradientTape()`:
   $$\frac{\partial y^c}{\partial A^k_{i, j}}$$
4. Global Average Pooling over the spatial dimensions $(i, j)$ yields channel importance weights $\alpha^c_k$:
   $$\alpha^c_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i, j}}$$
5. A weighted linear combination of forward activation maps is computed and passed through a Rectified Linear Unit (ReLU) to isolate features positively influencing class $c$:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha^c_k A^k \right)$$
6. The resulting $7 \times 7$ heatmap is normalized to $[0, 1]$, upsampled to $224 \times 224$ via bilinear interpolation, colormapped via `cv2.COLORMAP_JET`, and overlaid onto the scan with alpha blending:
   $$I_{\text{blended}} = (1 - \alpha) I_{\text{original}} + \alpha I_{\text{heatmap}}$$

### 2.2 Strengths & Clinical Limitations
- **Strengths:** Sub-second latency (~50ms); direct insight into the internal representations of the network.
- **Limitations:** Spatial resolution is constrained to the receptive field of the final conv layer ($7 \times 7$), providing a coarse regional localization rather than cellular/histological tumor boundaries.

---

## 3. Methodology B: LIME

- **File:** `src/cnnClassifier/utils/xai_lime.py`
- **Method Category:** Black-Box, Perturbation-Based Local Surrogate
- **Segmentation Algorithm:** SLIC / Quickshift Superpixels

### 3.1 Mathematical Formulation
LIME tests the local decision boundary around a specific scan $x$ by generating perturbed instances:
1. The scan $x$ is segmented into $d$ contiguous superpixel segments: $x' \in \{0, 1\}^d$.
2. Perturbed binary vectors $z' \in \{0, 1\}^d$ are sampled by randomly toggling superpixels on and off ($N = 60\text{--}80$ samples). Toggled-off superpixels are masked with neutral mean pixel color.
3. The original trained model predicts class probabilities for all perturbed scans: $f(z)$.
4. An exponential kernel measures proximity in superpixel Hamming space:
   $$\pi_x(z) = \exp\left( - \frac{D(x, z)^2}{\sigma^2} \right)$$
5. An interpretable linear model $g(z') = w_g^T z'$ is fit to minimize the fidelity loss:
   $$\mathcal{L}(f, g, \pi_x) = \sum_{z, z'} \pi_x(z) \left( f(z) - g(z') \right)^2$$
6. Superpixels with $w_g > 0$ positively push towards the predicted class (highlighted in transparent **Green**); superpixels with $w_g < 0$ oppose the verdict (highlighted in **Red**).

### 3.2 Strengths & Clinical Limitations
- **Strengths:** Completely model-agnostic; verifies whether surrounding background artifacts (e.g., bone, scan borders) artificially steer predictions.
- **Limitations:** Superpixel boundaries depend on low-level color gradients and random seeds; surrogate explanations represent local approximations.

---

## 4. Methodology C: Kernel SHAP

- **File:** `src/cnnClassifier/utils/xai_shap.py`
- **Method Category:** Axiomatic Cooperative Game Theory
- **Theoretical Foundation:** Shapley Values (Shapley, 1953; Lundberg & Lee, 2017)

### 4.1 Mathematical Formulation
Treats superpixel segments as "players" in a cooperative coalition game where the "payout" is the model's predicted class probability:
1. Each superpixel $i$ receives a unique Shapley attribution value $\phi_i(v)$:
   $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$
2. **Guaranteed Axioms:**
   - **Efficiency:** The sum of all superpixel attributions equals the difference between the scan's prediction and the expected base prediction: $\sum_i \phi_i = f(x) - \mathbb{E}[f(x)]$.
   - **Symmetry:** If two tissue segments contribute identically across all possible segment combinations, their attributions are equal.
   - **Dummy:** If a segment adds zero marginal value to every coalition, its attribution is strictly zero ($\phi_i = 0$).
   - **Additivity:** Attributions are additive across ensemble components.
3. Attributions are normalized symmetrically in $[-1, 1]$ and mapped onto a diverging `coolwarm` colormap:
   - **Warm / Red:** Positive Shapley values (pushes prediction towards Tumor).
   - **Cool / Blue:** Negative Shapley values (pushes prediction towards Normal).

### 4.2 Strengths & Clinical Limitations
- **Strengths:** Mathematically grounded and axiomatically fair; prevents arbitrary feature weighting.
- **Limitations:** Computationally intensive ($O(2^{|N|})$ exact evaluation requires sample budgeting in real-time settings).

---

## 5. Multi-Method Triangulation Strategy

| Dimension | Grad-CAM | LIME | Kernel SHAP |
| :--- | :--- | :--- | :--- |
| **Model Access** | White-Box (Gradients) | Black-Box (Predictions) | Black-Box (Predictions) |
| **Resolution** | Coarse ($7 \times 7$ conv grid) | Superpixel segments | Superpixel coalitions |
| **Output Interpretation** | Relative activation magnitude | Linear regression feature weights | Axiomatic Shapley values |
| **Latency** | Extremely Fast (~50ms) | Moderate (~1.5s) | Compute-Intensive (~2.5s) |
| **Primary Clinical Value** | Validates internal conv focus | Checks artifact sensitivity | Quantifies segment marginal impact |

**Triangulation Consensus:** When Grad-CAM activations, LIME positive superpixels, and warm SHAP attributions all converge on the identical renal parenchyma zone, clinical trust in the prediction is substantially higher than relying on any single explanation technique.
