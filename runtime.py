"""Model loading shared by training and inference."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from triage import MODEL_ID, MODEL_REVISION


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
        attn_implementation="eager",
    )
    return model, tokenizer


def device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.inference_mode()
def generate(model, tokenizer, prompt):
    inputs = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
