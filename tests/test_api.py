from fastapi.testclient import TestClient

from her2clf.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "HER2 affinity prediction API"
    }


def test_predict():
    response = client.post(
        "/predict",
        json={"sequence": "ARDYWGQGTL"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sequence"] == "ARDYWGQGTL"
    assert "probability_high" in data
    assert data["predicted_class"] in ["binder", "non-binder"]

def test_predict_invalid_amino_acid():
    response = client.post(
        "/predict",
        json={"sequence": "ARDXWGQGTL"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid amino acid."
    }
