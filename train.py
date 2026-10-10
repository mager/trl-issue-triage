"""Train a LoRA adapter. Validation loss is diagnostic; test data is never loaded."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from runtime import load_base
from triage import MODEL_ID, MODEL_REVISION, ROOT, load_split, training_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/triage"))
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use 2 for a wiring check, not a quality result")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of automatic CUDA/MPS")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty; choose a fresh --output directory to preserve the previous run.")
    if args.epochs <= 0 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("Use positive epochs and max-steps > 0 (or -1 for the full run).")
    set_seed(42)
    torch.set_num_threads(4)
    model, tokenizer = load_base()
    splits = {name: training_rows(load_split(name)) for name in ("train", "validation")}
    # Fail instead of silently truncating away the answer we intend to teach.
    for name, rows in splits.items():
        for row in rows:
            tokens = tokenizer.apply_chat_template(row["prompt"] + row["completion"])
            if len(tokens) > 256:
                raise ValueError(f"{name} example exceeds 256 tokens; shorten it or increase max_length.")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(splits["train"]),
        eval_dataset=Dataset.from_list(splits["validation"]),
        peft_config=LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM", bias="none",
        ),
        args=SFTConfig(
            output_dir=str(args.output), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            per_device_eval_batch_size=2, learning_rate=2e-4,
            max_length=256, completion_only_loss=True, packing=False,
            bf16=False, fp16=False, optim="adamw_torch", gradient_checkpointing=False,
            dataloader_pin_memory=False, use_cpu=args.cpu,
            eval_strategy="epoch", save_strategy="no", logging_steps=10,
            report_to="none", push_to_hub=False, seed=42, data_seed=42,
        ),
    )
    trainer.model.print_trainable_parameters()
    # Assert that TRL masks the prompt but retains the answer, using the actual collator.
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["labels"][0]
    assert (labels == -100).any() and (labels != -100).any(), "Unexpected completion mask"
    supervised = tokenizer.decode(labels[labels != -100])
    assert '"label"' in supervised and "Label this software issue" not in supervised, supervised
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    metadata = {
        "model": MODEL_ID, "revision": MODEL_REVISION, "seed": 42,
        "device": str(trainer.args.device), "elapsed_seconds": elapsed,
        "epochs_requested": args.epochs, "max_steps": args.max_steps,
        "optimizer_steps": trainer.state.global_step,
        "metrics": result.metrics, "history": trainer.state.log_history,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "trl", "peft", "datasets", "accelerate")},
        "data_sha256": {name: hashlib.sha256((ROOT / "data" / f"{name}.jsonl").read_bytes()).hexdigest()
                        for name in ("train", "validation")},
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Adapter saved to {args.output}; {elapsed:.1f} seconds on {trainer.args.device}.")


if __name__ == "__main__":
    main()
