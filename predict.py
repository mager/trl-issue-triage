"""Try one issue with the base model or a saved adapter."""
import argparse
from pathlib import Path
import torch
from peft import PeftModel
from runtime import device, generate, load_base
from triage import messages, parse_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tokenizer = load_base()
    prompt = messages(args.text)
    if len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True)) > 224:
        parser.error("Issue is too long for this short-input tutorial; shorten it first.")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device(args.device)).eval()
    raw = generate(model, tokenizer, prompt)
    print(raw)
    if parse_label(raw) is None:
        raise SystemExit("Invalid output: send this issue to a human; do not silently assign a label.")


if __name__ == "__main__":
    main()
