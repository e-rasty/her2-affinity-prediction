import pytest
from her2clf.inference import validate_sequence, load_artifacts, predict

def test_wrong_length():
    with pytest.raises(ValueError):
        validate_sequence("ARDY")

def test_invalid_amino_acid():
    with pytest.raises(ValueError):
        validate_sequence("ARDXWGQGTL")

def test_valid_sequence():
    sequence = validate_sequence("ARDYWGQGTL")
    assert sequence == "ARDYWGQGTL"

def test_sequence_is_cleaned():
    sequence = validate_sequence("  ardywgqgtl  ")
    assert sequence == "ARDYWGQGTL"

def test_prediction():
    encoder, model = load_artifacts()

    result = predict("ARDYWGQGTL", encoder, model)

    assert "sequence" in result
    assert "probability_high" in result
    assert "predicted_class" in result