# app.py

import torch
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel

from model import GPT
from config import config

app = FastAPI()

# ---- TOKENIZER (FROM NOTEBOOK) ----
enc = tiktoken.get_encoding("gpt2")

# ---- LOAD MODEL ----
device = torch.device("cpu")

state_dict = torch.load(
    "best_model_params.pt",
    map_location=device,
    weights_only=True
)

model = GPT(config)
model.load_state_dict(state_dict)
model.eval()

# ---- API SCHEMA ----
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 300

class GenerateResponse(BaseModel):
    output: str

# ---- ENDPOINT ----
@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    idx = torch.tensor(
        enc.encode(req.prompt),
        dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():
        out = model.generate(idx, min(req.max_tokens, 300))

    text = enc.decode(out[0].tolist())
    return {"output": text}
