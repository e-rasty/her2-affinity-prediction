from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from her2clf.inference import load_artifacts, predict

app=FastAPI()
encoder, model = load_artifacts()

class SequenceRequest(BaseModel):
    sequence: str

@app.get("/")
def root():
    return {"message": "HER2 affinity prediction API"}

@app.post("/predict")
def predict_sequence(request: SequenceRequest):
    try:
        return predict(request.sequence, encoder, model)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    
