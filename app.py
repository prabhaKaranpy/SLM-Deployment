import torch
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model ONCE at startup
model = torch.load("model.pt", map_location="cpu")
model.eval()

class InputData(BaseModel):
    x: list[float]

@app.get("/")
def health():
    return {"status": "Model API running"}

@app.post("/predict")
def predict(data: InputData):
    with torch.no_grad():
        inp = torch.tensor(data.x)
        output = model(inp)
    return {"output": output.tolist()}
