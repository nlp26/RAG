from fastapi import FastAPI, File, UploadFile, Form
import pdfplumber
import requests
import os

OLLAMA_BASE_URL = "http://localhost:11434/v1/completions"
OLLAMA_MODEL = "llama3.2:1b"

app = FastAPI()

def extract_text_from_pdfplumber(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

@app.post("/api/ask_pdf")
async def ask_ollama_with_pdf(
        prompt: str = Form(...),
        file: UploadFile = File(...)
    ):
    pdf_text = extract_text_from_pdfplumber(file.file)
    # Use the extracted text as context for the prompt
    ollama_prompt = f"Document:\n{pdf_text}\n\nQuestion: {prompt}\n\nAnswer:"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": ollama_prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_BASE_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        return {"response": result.get("completion") or result}
    else:
        return {"error": response.text}

@app.get("/")
def read_root():
    return {"hello": "Ollama PDF microservice running"}

# Run with: uvicorn ollama_pdf_api:app --reload
# Test with:
# curl -X POST "http://localhost:8000/api/ask_pdf" \
# -F 'prompt=Summarize the document' \
# -F 'file=@/absolute/path/to/your.pdf'