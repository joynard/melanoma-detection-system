# Melanoma Risk Detection Application Using Weighted Ensemble CNN with Genetic Algorithm Optimization

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-Ionic%20Angular-3880FF.svg)](https://ionicframework.com/)
[![Framework](https://img.shields.io/badge/AI-TensorFlow%20%2F%20Keras-FF6F00.svg)](https://tensorflow.org/)

This repository contains the complete source code, deep learning model pipelines, and comprehensive documentation for my Undergrad Thesis project: 

**Image-Based Early Melanoma Risk Detection System**. 

The project is engineered using a decoupled *client-server* architecture, leveraging a high-performance *hybrid mobile application* for the user interface and an optimized *deep learning backend* for intensive computational inference.

---

## 📌 Table of Contents
1. [Background](#-background)
2. [Key Features & System Pipeline](#-key-features--system-pipeline)
3. [Dataset & Base Model Specifications](#-dataset--base-model-specifications)
4. [Genetic Algorithm (GA) Optimization](#-genetic-algorithm-ga-optimization)
5. [Project Architecture](#-project-architecture)
6. [Installation & Setup](#-installation--setup)
7. [Evaluation Metrics](#-evaluation-metrics)
8. [Disclaimer](#-disclaimer)

---

## Background
Melanoma is the most aggressive and malignant form of skin cancer, carrying a high fatality rate due to its rapid metastatic potential to vital organs. The critical bottleneck in early clinical screening is the extreme visual ambiguity between early-stage melanoma and benign lesions, such as *melanocytic nevi* (common moles), which introduces substantial human observer bias during manual dermoscopy examinations.

In Indonesia, this issue is exacerbated by systemic healthcare constraints, including a low physician-to-population ratio (approximately 0.47 per 1.000 citizens) and geographical maldistribution of medical facilities. This system resolves these challenges by providing an automated, accessible, and highly communicative early-screening tool. By mapping complex continuous AI predictions into three discrete risk tiers (Low, Medium, High), it fosters public health awareness and encourages timely, proactive professional dermatological consultations.

---

## Key Features & System Pipeline

The system processes an incoming user-submitted dermoscopic image through a robust sequential pipeline:

1. **Input Quality Retention**: Photos with sub-optimal exposure, excessive motion blur, or critically low resolutions are automatically rejected at the client side to mitigate model prediction bias.
2. **Automated Lesion Segmentation (Otsu's Thresholding)**: Isolates the clinical region of interest (*foreground*) from healthy skin (*background*) by adaptively maximizing the between-class variance:
   $$\sigma^2_B(T) = \omega_1(T) \cdot \omega_2(T) \cdot [\mu_1(T) - \mu_2(T)]^2$$
   The bounding box coordinates ($x_{min}, x_{max}, y_{min}, y_{max}$) are derived from the resulting binary mask to execute localized lesion cropping.
3. **Image Restoration & Hair Removal Artifact Mitigation**:
   * Hair strands are extracted using a mathematical morphological *Blackhat* filter: $\text{Blackhat}(f) = (f \bullet b) - f$.
   * Occulted pixels under hair structures are seamlessly reconstructed using the `INPAINT_TELEA` algorithm, which propagates pixel values inward from the mask boundaries via the *Fast Marching Method* (FMM).
4. **Weighted Ensemble Inference**: The restored $224 \times 224$ image is evaluated concurrently across three distinct pretrained CNN backbones to generate separate class probability distributions.
5. **Rule-Based Risk Categorization**: To optimize clinical sensitivity and minimize catastrophic False Negatives, the continuous probability output ($P_{final}$) is mapped into human-readable risk tiers:
   * **Low Risk** ($0.00 \le P_{final} \le 0.39$)
   * **Medium Risk** ($0.40 \le P_{final} \le 0.69$)
   * **High Risk** ($0.70 \le P_{final} \le 1.00$)

---

## Dataset & Base Model Specifications

### HAM10000 Dataset
The models were trained on the large-scale HAM10000 public dermoscopy dataset, which features a highly imbalanced class distribution across 10,015 images:
* **Melanocytic Nevi (NV)**: 6,705 samples *(Majority Class)*
* **Melanoma (MEL)**: 1,113 samples
* **Benign Keratosis-like Lesions (BKL)**: 1,099 samples
* **Basal Cell Carcinoma (BCC)**: 514 samples
* **Actinic Keratoses (AKIEC)**: 327 samples
* **Vascular Lesions (VASC)**: 142 samples
* **Dermatofibroma (DF)**: 115 samples

*Note: Extreme class imbalance is resolved during the training phase using target-specific stochastic data augmentation techniques (Random Rotation from $-45^\circ$ to $+45^\circ$, Horizontal Flips, and Vertical Flips) on minority classes until they match the size of the majority class.*

### Base Model Backbones
Three architectures with complementary inductive biases were chosen for the ensemble:
1. **ResNet50**: Implements deep residual *skip connections* (bottleneck blocks) to eradicate vanishing gradients during backpropagation across 50 parameter-weighted layers.
2. **DenseNet121**: Maximizes dense feature reuse by executing structural *feature map concatenation* across all preceding layers within a dense block.
3. **EfficientNetB0**: Utilizes structured *compound scaling* to uniformly balance network depth, width, and resolution, relying on *MBConv* blocks with integrated *Squeeze-and-Excitation* (SE) optimization for lightweight mobile deployment.

---

## Genetic Algorithm (GA) Optimization

Rather than using *simple averaging*—which uniformizes base models regardless of their specific error rates—this system implements a heuristic **Genetic Algorithm** to search for the optimal parameter weight combination vector $[w_1, w_2, w_3]$ that maximizes the *macro-averaged F1-Score* on the validation dataset.

* **Chromosome Representation**: A 1D continuous floating-point array `[w1, w2, w3]` subjected to strict boundary constraints: $w_1 (\text{ResNet50}) + w_2 (\text{DenseNet121}) + w_3 (\text{EfficientNetB0}) = 1.0$, where $w_i \ge 0$.
* **Fitness Function Evaluation**: Evaluated empirically against the validation split's macro F1-score metric to ensure unbiased performance assessment across minority categories.
* **Selection Strategy**: Higher fitness scores translate to an exponentially higher probability of selection as parent chromosomes for subsequent generations.
* **Arithmetic Crossover**: Parent vectors are recombined using continuous arithmetic blending governed by a balanced scaling coefficient ($\alpha = 0.5$):
  $$w_{offspring} = (\alpha \cdot w_{parent1}) + ((1 - \alpha) \cdot w_{parent2})$$
* **Stochastic Mutation**: Introduces controlled genetic drift via a calibrated *mutation rate*, re-adjusting inner weights while maintaining the cumulative sum boundary of $1.0$ to successfully bypass local optima traps.
* **Termination Criterion**: Concludes automatically at the 100th generation or upon statistical fitness value convergence.

### Ensemble Agregation Formula:
$$P_{final} = (w_1 \times P_1) + (w_2 \times P_2) + (w_3 \times P_3)$$
*Example Evaluation:*
$$P_{final} = (0.15 \times 0.80) + (0.55 \times 0.90) + (0.30 \times 0.70) = 0.825 \rightarrow \mathbf{High\ Risk}$$

---

## Project Architecture

```text
melanoma-weighted-ensemble-ga/
│
├── mobile-client/         # Front-End Application (Ionic Angular Cross-Platform)
│   ├── src/
│   ├── package.json
│   └── ionic.config.json
│
├── server-backend/        # Machine Learning Production API Gateway (FastAPI)
│   ├── app/
│   │   ├── main.py        # REST API endpoints & Otsu/Telea preprocessing pipelines
│   │   └── core/          # Weighted Ensemble mathematical integration engine
│   ├── requirements.txt
│   └── Dockerfile
│
├── model-training/        # Experimental ML Pipelines & Heuristic GA Search
│   ├── training_cnn.py    # Fine-tuning scripts for ResNet50, DenseNet121, EfficientNetB0
│   ├── genetic_search.py  # Global heuristic GA optimization script for ensemble weights
│   └── notebook/          # Interactive Jupyter Notebook experiments (.ipynb)
│
└── README.md              # Root repository documentation
```
---

# Installation & Setup

## 1. Server Backend (FastAPI)

Ensure you have Python 3.10+ installed along with the necessary C++ build tools required for OpenCV native dependencies.

```bash
cd server-backend

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 2. Mobile Client (Ionic Angular)

Ensure Node.js and the Ionic CLI are installed and properly configured on your system.

```bash
cd mobile-client

npm install

ionic serve
```

# Evaluation Metrics

Classification reliability across all seven dermatological classes is evaluated using a Confusion Matrix (`TP`, `TN`, `FP`, `FN`) and the following metrics:

### Accuracy

Measures the overall proportion of correct predictions across all classes.

### Precision

Measures the positive predictive value of the model, indicating how many predicted positives are actually correct.

### Recall (Sensitivity)

Measures the proportion of actual positive cases that are correctly identified by the model.

> This metric is particularly important for minimizing False Negatives, thereby reducing the clinical risk of missing melanoma cases.

### Macro-Averaged F1-Score

Represents the harmonic mean of precision and recall across all classes.

This metric serves as the primary optimization objective during the Genetic Algorithm (GA) search process, helping prevent the model from overfitting to the heavily represented **Nevus** class.

# Disclaimer

This application is intended solely as an **early skin lesion screening tool** for risk detection purposes.

It **does not** replace professional medical diagnosis, clinical biopsy procedures, or consultation with a licensed dermatologist. Any suspicious findings should be evaluated by a qualified healthcare professional.

# License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

Developed by **Alexander Fabiano Joynard Lapod (2026)**.
