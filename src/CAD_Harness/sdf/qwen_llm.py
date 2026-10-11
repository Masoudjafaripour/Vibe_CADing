"""Local Qwen (HF transformers) llm_fn for harness.run_harness.

Lazily loads the model on first call so importing this module (or
running the harness with a mock llm_fn) never requires a GPU/download.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-8B"

_loaded = {}


def _load(model_name):
    if model_name not in _loaded:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        _loaded[model_name] = (tokenizer, model)
    return _loaded[model_name]


def make_qwen_llm(model_name: str = MODEL_NAME, max_new_tokens: int = 3072):
    def llm_fn(prompt: str) -> str:
        tokenizer, model = _load(model_name)
        messages = [{"role": "user", "content": prompt}]
        # Thinking off: SDF code is long, and reasoning tokens would eat the budget
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
    return llm_fn


qwen_llm = make_qwen_llm()
