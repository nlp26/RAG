from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

OLLAMA_BASE_URL = "http://localhost:11434/v1/completions"
OLLAMA_MODEL = "llama3.2:1b"

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/api/ask")
def ask_ollama(data: Query):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": data.prompt,
        "stream": False  # if True, you must chunk and stream responses
    }
    response = requests.post(OLLAMA_BASE_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        return {"response": result.get("completion") or result}
    else:
        return {"error": response.text}

# Optional: for root GET request (health check)
@app.get("/")
def read_root():
    return {"hello": "Ollama microservice running"}

# To launch:
# uvicorn ollama_api:app --reload
