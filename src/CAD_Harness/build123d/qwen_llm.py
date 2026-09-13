"""Qwen3-8B-backed llm_fn for harness.run_harness.

Lazily loads the model on first call so importing this module (or
running the harness with a mock llm_fn) never requires a GPU/download.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-8B"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        _model = _model.to("cuda" if torch.cuda.is_available() else "cpu")
    return _tokenizer, _model


def qwen_llm(prompt: str, max_new_tokens: int = 512) -> str:
    tokenizer, model = _load()
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return generated
