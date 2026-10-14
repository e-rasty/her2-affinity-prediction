from pathlib import Path
import joblib
import os
from xgboost import XGBClassifier

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.json"
ENCODER_PATH = ARTIFACTS_DIR / "onehot_encoder.joblib"


def load_artifacts():
    encoder=joblib.load(ENCODER_PATH)

    model=XGBClassifier()
    model.load_model(MODEL_PATH)

    return encoder, model


def validate_sequence(sequence):
    if type(sequence) != str:
        raise TypeError("Sequence must be a string.")

    sequence = sequence.strip().upper()

    if len(sequence) != 10:
        raise ValueError("Sequence must contain exactly 10 amino acids.")

    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    for amino_acid in sequence:
        if amino_acid not in valid_amino_acids:
            raise ValueError("Invalid amino acid.")

    return sequence

def predict(sequence, encoder, model):
    sequence=validate_sequence(sequence)

    X= encoder.transform([list(sequence)])
    probability= model.predict_proba(X)[0,1]
    
    if probability >= 0.5:
        predicted_class = "binder"
    else:
        predicted_class = "non-binder"

    return {
        "sequence": sequence, 
        "probability_high": float(probability), 
        "predicted_class": predicted_class
    }
