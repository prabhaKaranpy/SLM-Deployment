# Small Language Model - Deployment 

A lightweight, containerized API for serving a custom Small Language Model (SLM). This project implements a GPT-style transformer architecture using PyTorch and exposes it via a fast, asynchronous REST API using FastAPI.

Designed for easy deployment on Docker and Hugging Face Spaces.

##  Features

* **Custom GPT Architecture:** A decoder-only transformer model implemented from scratch (Causal Self-Attention, LayerNorm, GELU).
* **FastAPI Backend:** High-performance, easy-to-use API with automatic Swagger documentation.
* **Dockerized:** Fully containerized environment for consistent deployment anywhere.
* **CPU Optimized:** Configured to run efficiently on standard CPU hardware.
* **Tokenizer:** Uses OpenAI's `tiktoken` (GPT-2 encoding) for efficient tokenization.

##  Tech Stack

* **Language:** Python 3.10
* **ML Framework:** PyTorch
* **API Framework:** FastAPI & Uvicorn
* **Container:** Docker

---

##  Project Structure

```bash
├── app.py                 # FastAPI application and endpoints
├── model.py               # GPT Model architecture definition
├── config.py              # Model configuration parameters
├── best_model_params.pt   # Pre-trained model weights (Git LFS)
├── Dockerfile             # Container instructions
├── requirements.txt       # Python dependencies
└── .gitattributes         # Git LFS tracking for model files
```
##  Getting Started

### Prerequisites
* Python 3.10+
* Git & Git LFS (Large File Storage)

**Important:** You must install Git LFS to download the model weights (`.pt` file) correctly.


## ☁️ Deployment on Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Select **Docker** as the SDK.
3. Push this repository to the Space.
4. The `Dockerfile` handles the setup automatically.

*Note: Ensure your Hugging Face token has `WRITE` permissions when pushing LFS files.*

** Live API:** [https://huggingface.co/spaces/prabha36/slm-deploy](https://huggingface.co/spaces/prabha36/slm-deploy)
