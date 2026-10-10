"""Compare the original model and adapter on the same held-out issues and prompt."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from runtime import device, generate, load_base
from triage import MODEL_ID, MODEL_REVISION, ROOT, load_split, messages, parse_label, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error("Adapter not found. Run train.py first.")
    torch.set_num_threads(4)
    rows = load_split(args.split)
    model, tokenizer = load_base()
    model = PeftModel.from_pretrained(model, args.adapter).to(device(args.device)).eval()
    report = {"model": MODEL_ID, "revision": MODEL_REVISION, "device": str(model.device),
              "split": args.split, "prompt": messages("<issue text>"),
              "decoding": {"do_sample": False, "max_new_tokens": 32},
              "data_sha256": hashlib.sha256((ROOT / "data" / f"{args.split}.jsonl").read_bytes()).hexdigest(),
              "adapter_run": json.loads((args.adapter / "run.json").read_text())}
    def run():
        predictions = []
        for row in rows:
            raw = generate(model, tokenizer, messages(row["text"]))
            predictions.append({"id": row["id"], "text": row["text"], "expected": row["label"],
                                "predicted": parse_label(raw), "raw": raw})
        return {"metrics": score(predictions), "predictions": predictions}
    with model.disable_adapter():
        report["baseline"] = run()
    report["fine_tuned"] = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name in ("baseline", "fine_tuned"):
        print(name, json.dumps(report[name]["metrics"]))
    print(f"Raw predictions and metadata: {args.output}")


if __name__ == "__main__":
    main()
