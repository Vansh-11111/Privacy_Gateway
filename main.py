
from fastapi import FastAPI

app = FastAPI(
    title="LLM Gateway",
    description="A starter API gateway",
    version="0.1.0",
)


@app.get("/")
def home():
    return {"message": "LLM Gateway is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions():
    return {
        "message": "Gateway endpoint is ready",
        "next_step": "Connect an LLM provider"
    }
