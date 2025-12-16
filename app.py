# app.py

import torch
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel

from model import GPT
from config import config

# -------------------- APP --------------------
app = FastAPI(title="SLM Text Generation API")

# -------------------- TOKENIZER --------------------
# GPT-2 tokenizer (matches training)
enc = tiktoken.get_encoding("gpt2")

# -------------------- MODEL LOADING --------------------
device = torch.device("cpu")

state_dict = torch.load(
    "best_model_params.pt",
    map_location=device
)

model = GPT(config)
model.load_state_dict(state_dict)
model.eval()

# -------------------- SCHEMAS --------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 300

class GenerateResponse(BaseModel):
    output: str

# -------------------- HEALTH CHECK --------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }

# -------------------- GENERATION ENDPOINT --------------------
@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    # Encode input text
    idx = torch.tensor(
        enc.encode(req.prompt),
        dtype=torch.long
    ).unsqueeze(0)

    # Generate tokens
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=min(req.max_tokens, 300)
        )

    # Decode output tokens
    text = enc.decode(out[0].tolist())
    return {"output": text}




# # app.py

# import torch
# import tiktoken
# from fastapi import FastAPI
# from pydantic import BaseModel

# from model import GPT
# from config import config

# app = FastAPI()

# # ---- TOKENIZER (FROM NOTEBOOK) ----
# enc = tiktoken.get_encoding("gpt2")

# # ---- LOAD MODEL ----
# device = torch.device("cpu")

# state_dict = torch.load(
#     "best_model_params.pt",
#     map_location=device
# )

# model = GPT(config)
# model.load_state_dict(state_dict)
# model.eval()

# # ---- API SCHEMA ----
# class GenerateRequest(BaseModel):
#     prompt: str
#     max_tokens: int = 300

# class GenerateResponse(BaseModel):
#     output: str

# # ---- ENDPOINT ----
# @app.post("/generate", response_model=GenerateResponse)
# def generate_text(req: GenerateRequest):
#     idx = torch.tensor(
#         enc.encode(req.prompt),
#         dtype=torch.long
#     ).unsqueeze(0)

#     with torch.no_grad():
#         out = model.generate(idx, min(req.max_tokens, 300))

#     text = enc.decode(out[0].tolist())
#     return {"output": text}
