# HER2 Affinity Prediction

Predicting HER2-binding affinity from trastuzumab CDRH3 sequence variants using classical machine learning and ESM-2 protein language model representations, with a containerized FastAPI service for inference.

This project investigates how **sequence representation** and **model complexity** affect antibody affinity classification. Simple sequence features are compared with pretrained ESM-2 embeddings using Logistic Regression and XGBoost, with leakage-aware evaluation designed to reduce information sharing between closely related sequences.

The final model uses position-specific one-hot encoding with XGBoost and is packaged as a tested inference pipeline with saved model artifacts, input validation, a REST API, and Docker deployment.

## Prediction Task

The dataset contains approximately **524,000 unique trastuzumab CDRH3 variants**, each 10 amino acids long, with experimentally measured HER2-binding affinity.

The task is formulated as binary classification:

- **Positive:** high-affinity variants
- **Negative:** mid- and low-affinity variants

Closely related antibody sequences can make random train/test splits overly optimistic. The analysis therefore uses predefined sequence clusters to maintain separation between related variants.

Because of class imbalance, **Average Precision (AP)** is used as the primary evaluation metric, with ROC-AUC reported as a secondary metric.

## Modeling Approach

Models were developed incrementally to separate the effects of sequence representation from model complexity.

Four representations were investigated:

1. **Edit distance** — overall mutational distance from the reference sequence.
2. **Position-specific one-hot encoding** — 10 positions × 20 amino acids, producing 200 sparse features.
3. **Mean-pooled ESM-2** — residue embeddings averaged across the sequence.
4. **Flattened ESM-2** — residue embeddings concatenated across positions, preserving positional information and producing 3,200 dense features.

Logistic Regression and XGBoost were evaluated across the sequence representations to examine whether richer pretrained representations remained beneficial when the classifier itself could model nonlinear relationships.

## Results

### Representation Comparison

| Representation | Model | ROC-AUC | AP |
|---|---|---:|---:|
| One-hot | Logistic Regression | 0.9644 | 0.9365 |
| ESM mean-pooled | Logistic Regression | 0.9495 | 0.9046 |
| ESM flattened | Logistic Regression | **0.9857** | **0.9736** |
| One-hot | XGBoost | 0.99130 | 0.98445 |
| ESM mean-pooled | XGBoost | 0.96398 | 0.93294 |
| ESM flattened | XGBoost | **0.99154** | **0.98458** |

*All representation comparisons were performed on the validation set.*

Preserving residue position was important across both model classes. Flattened ESM substantially outperformed mean-pooled ESM, indicating that mean pooling removes information important for this fixed-length CDRH3 prediction task.

With Logistic Regression, flattened ESM also substantially outperformed one-hot encoding (**0.9736 vs. 0.9365 AP**). With XGBoost, however, the difference nearly disappeared (**0.98458 vs. 0.98445 AP**).

This suggests that pretrained ESM representations provide substantial additional predictive value to a linear classifier, while a nonlinear model can extract much of the relevant sequence signal directly from position-specific residue identity.

## Final Model

Because one-hot XGBoost essentially matched flattened ESM XGBoost while using only **200 sparse features instead of 3,200 dense features**, the simpler one-hot representation was selected for the final model.

Hyperparameters were tuned on the training set using randomized search with **5-fold Stratified Group K-Fold cross-validation**. `cluster_centre` was used as the grouping variable to keep related sequences within the same fold, and Average Precision was used as the optimization metric.

The selected XGBoost configuration was:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.3 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.8 |

The tuned model achieved:

- **Grouped CV AP:** 0.9510
- **Validation AP:** 0.9918
- **Validation ROC-AUC:** 0.9956

After model and hyperparameter selection were complete, the final model was retrained on the combined training and validation data and evaluated on the held-out test set.

### Final Held-Out Test Performance

| Metric | Score |
|---|---:|
| **Average Precision** | **0.9836** |
| **ROC-AUC** | **0.9913** |

## Key Findings

- **Residue position matters.** Flattened ESM substantially outperformed mean-pooled ESM with both Logistic Regression and XGBoost.
- **ESM helps a linear model.** Flattened ESM improved Logistic Regression AP from **0.9365 to 0.9736** compared with one-hot features.
- **Model complexity changes the value of the representation.** With XGBoost, one-hot and flattened ESM were essentially tied at **0.98445 vs. 0.98458 AP**.
- **Simple representations can be highly competitive.** Position-specific one-hot encoding combined with a nonlinear learner captured nearly the same predictive signal as the substantially larger ESM representation.
- **Edit distance alone is an incomplete predictor.** Overall mutational distance was substantially less predictive than models retaining residue identity and position.

Overall, the results highlight an interaction between **representation and model capacity**: richer pretrained protein representations can provide substantial value to simple models, but their advantage may diminish when a nonlinear learner can extract relevant interactions directly from well-structured sequence features.

## Compute and Engineering

The flattened ESM training representation had shape `(367,044, 3,200)`. Because the dense representation exceeded the practical memory capacity of the local machine, these experiments were run on a **Google Cloud Compute Engine VM with 8 vCPUs and 32 GB RAM**.

Embeddings were stored as disk-backed **NumPy memmaps**, allowing the large representation to be generated and reused without loading the complete array into memory.

Reusable functionality for data loading, feature generation, ESM embedding, model training, inference, and API serving is organized under `src/her2clf/` rather than being implemented entirely inside notebooks.

## Repository Structure

```text
her2-affinity-prediction/
├── artifacts/
│   ├── onehot_encoder.joblib
│   └── xgboost_model.json
├── data/
│   └── raw/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── figures/
│   ├── esm_flat_pr_curve.png
│   ├── esm_flat_roc_curve.png
│   └── esm_flat_xgb_calibration.png
├── notebooks/
│   └── modeling_analysis.ipynb
├── src/
│   └── her2clf/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py
│       ├── esm.py
│       ├── features.py
│       ├── inference.py
│       └── models.py
├── tests/
│   ├── test_api.py
│   └── test_inference.py
├── Dockerfile
├── pyproject.toml
├── .gitignore
└── README.md
```

## Inference API

The final one-hot XGBoost model is exposed through a FastAPI inference service.

The API accepts a 10-residue CDRH3 sequence, validates the input, applies the saved position-specific one-hot encoder, and returns the predicted probability and binary affinity classification.

Example request:

```json
{
  "sequence": "ARDYWGQGTL"
}
```

Example response:

```json
{
  "sequence": "ARDYWGQGTL",
  "probability_high": 0.0032255966,
  "predicted_class": "non-binder"
}
```

Input validation requires exactly 10 residues drawn from the 20 standard amino acids.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and dependencies:

```bash
pip install -e .
```

Run the test suite:

```bash
pytest
```

Run the API locally:

```bash
python -m uvicorn her2clf.api:app --reload
```

Interactive API documentation is then available at:

`http://localhost:8000/docs`

## Docker

The inference API can also be run in a Docker container, providing a reproducible runtime independent of the local Python environment.

Build the image:

```bash
docker build -t her2-affinity-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 her2-affinity-api
```

The API is then available at:

`http://localhost:8000`

with interactive documentation at:

`http://localhost:8000/docs`

The production container includes only the dependencies required for the final one-hot XGBoost inference pipeline rather than the substantially larger PyTorch/Transformers stack used during ESM experimentation.