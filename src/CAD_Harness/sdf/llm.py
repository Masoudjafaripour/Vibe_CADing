"""LLM backends. Each returns an llm_fn: callable(prompt: str) -> str.

- "qwen":   local Hugging Face model (default Qwen/Qwen3-8B), see qwen_llm.py
- "openai": any OpenAI-compatible /chat/completions endpoint (OpenAI,
            vLLM, Ollama, LM Studio, ...), configured by environment:
              OPENAI_BASE_URL  (default https://api.openai.com/v1)
              OPENAI_API_KEY   (optional for local servers)
              SDF_LLM_MODEL    (or pass --model)
"""
import json
import os
import urllib.error
import urllib.request

BACKENDS = ("qwen", "openai")


def openai_llm(model: str = None, base_url: str = None, api_key: str = None,
               temperature: float = 0.2, max_tokens: int = 4096, timeout: float = 300):
    model = model or os.environ.get("SDF_LLM_MODEL")
    if not model:
        raise ValueError("The openai backend needs a model: pass --model or set SDF_LLM_MODEL.")
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def llm_fn(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(f"{base_url}/chat/completions", body, headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LLM request failed ({e.code}): {e.read().decode()[:500]}") from None
        return data["choices"][0]["message"]["content"]

    return llm_fn


def default_backend() -> str:
    return "openai" if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL") else "qwen"


def get_llm(backend: str = None, model: str = None):
    backend = backend or default_backend()
    if backend == "openai":
        return openai_llm(model)
    if backend == "qwen":
        from qwen_llm import MODEL_NAME, make_qwen_llm  # imports torch; keep lazy
        return make_qwen_llm(model or MODEL_NAME)
    raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")
