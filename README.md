# 🧠 Small Language Model (SLM) – Text Generation API

A **custom-trained Small Language Model (SLM)** deployed as a **production-ready FastAPI service** using **Docker** and hosted on **Hugging Face Spaces**.

This project demonstrates the **complete end-to-end pipeline**:
model training → serialization → API serving → containerization → cloud deployment.

---

## 🚀 Features

- Custom **GPT-style Transformer** (trained from scratch)
- Token-level **text generation**
- REST API built with **FastAPI**
- **Dockerized** for reproducible deployment
- Hosted on **Hugging Face Spaces**
- Configurable generation length (up to 300 tokens)

---

## 🧩 Model Configuration

- **Architecture**: GPT-style Decoder Transformer
- **Vocabulary Size**: `50257` (GPT-2 tokenizer)
- **Context Length (block_size)**: `128`
- **Transformer Layers**: `6`
- **Attention Heads**: `6`
- **Embedding Dimension**: `384`
- **Framework**: PyTorch

Model weights:
best_model_params.pt

yaml
Copy code

> ⚠️ This is a **Small Language Model** created for learning, experimentation, and deployment practice.

---

## 📡 API Usage

### Endpoint
POST /generate

bash
Copy code

### Request
```json
{
  "prompt": "Once upon a time",
  "max_tokens": 300
}
Response
json
Copy code
{
  "output": "Once upon a time there was a..."
}
🏗 Project Structure
graphql
Copy code
.
├── app.py                  # FastAPI application
├── model.py                # GPT model architecture
├── config.py               # Model configuration
├── best_model_params.pt    # Trained model weights
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
├── .gitattributes          # Git LFS for .pt files
└── README.md
🐳 Run Locally with Docker
bash
Copy code
docker build -t slm-api .
docker run -p 7860:7860 slm-api
API will be available at:

bash
Copy code
http://localhost:7860/generate
☁️ Deployment
Platform: Hugging Face Spaces

Runtime: Docker

Port: 7860

The API stays live as long as the Space is active.

🎯 Why This Project?
This project focuses on real-world ML engineering, not just training:

Designing a transformer model

Handling model serialization

Serving inference via REST API

Debugging deployment & memory issues

Shipping a production-style ML service

🛠 Tech Stack
Python

PyTorch

FastAPI

Docker

Hugging Face Spaces

tiktoken (GPT-2 tokenizer)

🔮 Future Enhancements
Streaming token generation

Web frontend (React / Next.js)

Model quantization for lower memory usage

Authentication & rate limiting

Prompt templates

📜 License
This project is intended for educational and experimental purposes.

🙌 Author
Prabha Karan
