# TRL issue triage

A small, runnable introduction to supervised fine-tuning: teach
`HuggingFaceTB/SmolLM2-135M-Instruct` to label software issues with JSON using
Hugging Face TRL and a LoRA adapter.

```text
"Saving my profile returns a 500 error."
                     ↓
              {"label":"bug"}
```

This is a teaching dataset and experiment, not a production triage service.
Labels are `bug`, `feature`, `question`, and `docs`. Documentation issues take
precedence over the other labels. There is no GitHub API integration and the
scripts never modify real issues.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone
this repository. Python 3.11 is selected by `.python-version`; uv can
download it if needed. The first run also downloads public model weights from
Hugging Face. Allow a few GB of disk space for the environment and caches.
No API key, paid service, or Hugging Face login is required for this model.

```bash
git clone https://github.com/mager/trl-issue-triage.git
cd trl-issue-triage
uv sync --locked
uv run pytest -q

# Two optimizer steps check loading, masking, training, and adapter saving.
uv run python train.py --max-steps 2 --output outputs/smoke

# The full experiment: 8 epochs, 80 optimizer steps.
uv run python train.py --output outputs/triage

# Same held-out issues and prompt, adapter disabled vs. enabled.
uv run python evaluate.py --adapter outputs/triage --output outputs/comparison.json

# Try your own short issue. Omit --adapter to use the original model.
uv run python predict.py --adapter outputs/triage \
  "Saving my profile returns a 500 error."
```

Use a fresh `--output` directory for each training run. The script refuses to
overwrite an existing nonempty run. A smoke adapter proves the pipeline works;
it is not a meaningful trained classifier. Evaluation prints aggregate scores
and saves every raw answer so you can inspect errors.

## Hardware and versions

The recorded run uses an Apple M4 Pro with 24 GB unified memory, Python 3.11,
and PyTorch's MPS device. CUDA is selected automatically when available, then
MPS, then CPU. The complete pipeline has been exercised on MPS, and the two-step
training check also passed on CPU. CUDA has not been tested. Other hardware
can differ in speed and numerical results. Force CPU with `train.py --cpu` and
`evaluate.py --device cpu` / `predict.py --device cpu`.

The model is loaded in float32 with eager attention. This deliberately avoids
bitsandbytes, FlashAttention, mixed precision, and CUDA-only training extras.
LoRA reduces trainable parameters but the base model still needs to be loaded.
This is not a promise that arbitrary larger models will fit on your laptop.

The project pins TRL **1.14.0**, Transformers **5.19.0**, PEFT **0.21.2**, PyTorch
**2.14.1**, Datasets **5.1.0**, and Accelerate **1.15.0**. `uv.lock` also records
transitive dependencies. TRL 1.15.0's fused LM-head path failed in our MPS smoke
run with `AttributeError: 'NoneType' object has no attribute 'apply'`; 1.14.0
completed the run. Do not remove the pin without repeating the smoke test.

The model revision is pinned in `triage.py`; no remote model code is trusted.
Training saves the adapter and tokenizer, not a standalone copy of the base
weights. Loading the adapter still requires the matching base model.

## What's in the repo

| File | Purpose |
| --- | --- |
| `data/train.jsonl` | 80 authored issues, 20 per label |
| `data/validation.jsonl` | 12 separate issues for development diagnostics |
| `data/test.jsonl` | 20 separate issues for the final comparison |
| `triage.py` | Shared prompt, label policy, data format, strict parser, metrics |
| `runtime.py` | Pinned model loading and greedy generation |
| `train.py` | TRL `SFTTrainer`, completion-only loss, LoRA, run metadata |
| `evaluate.py` | Baseline and adapter comparison with every raw prediction |
| `predict.py` | Inference on one short issue |
| `results/comparison.json` | Recorded experiment, including training history |
| `tests/` | Data separation, output validation, and metric checks |

All 112 examples were authored for this tutorial with AI assistance. No real
customer tickets or private repository data are included. They are short,
mostly unambiguous, balanced English examples and are intentionally easy to
inspect. Their quality and coverage are the main limitation of the experiment.
Code and authored data use the MIT license; the base model retains its own
[Apache 2.0 license](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).

## What training does

The JSONL rows contain `id`, `text`, and `label`. `training_rows()` converts each
to a conversational prompt/completion pair. The prompt has the task instruction
and issue text; the completion is the assistant's JSON answer. TRL applies the
tokenizer's chat template. `completion_only_loss=True` masks prompt tokens from
the loss; a runtime check inspects the actual collator output before training.
Examples longer than 256 tokens fail explicitly rather than silently losing
their answer to truncation.

LoRA freezes the original model weights and trains low-rank updates. Here it
targets linear layers at rank 8, with alpha 16 and dropout 0.05. The run trains
2,442,240 parameters, about 1.78% of the model plus adapters. Each optimizer
step accumulates four batches of two examples: eight issues on one device.
Eight epochs over 80 training examples produce 80 optimizer steps.

Validation loss is logged each epoch. We use the final adapter from a fixed
eight-epoch run, not a model selected on the test set. The learning rate is
`2e-4` with the trainer's linear decay. Neither training nor generation uploads
your examples to a hosted inference service. Downloads and package/model
clients still make network requests; this is not an offline installation.

## Read the results critically

Recorded on 2026-10-10, using the default eight-epoch run:

| Metric | Original model | Adapter |
| --- | ---: | ---: |
| Valid schema | 0/20 | 20/20 |
| Correct labels | 0/20 | 12/20 |
| Bugs | 0/5 | 5/5 |
| Features | 0/5 | 4/5 |
| Questions | 0/5 | 3/5 |
| Documentation | 0/5 | 0/5 |

Training took 36.5 seconds inside `trainer.train()`, excluding downloads,
loading, and the separate generation comparison. The adapter weights are about
9.4 MiB. All documentation issues were misclassified, so this is not ready to
label real issues automatically. The original model generated prose rather
than the required JSON; its zero score is failure of this particular contract,
not evidence that it cannot recognize issue categories under other prompts.

`results/comparison.json` records the measured run, model revision, dependency
versions, data hashes, prompt, decoding settings, training time, and raw
predictions. The baseline is the same original instruct model with the adapter
disabled. Both passes use greedy decoding and at most 32 new tokens.

There are two scores: strict schema validity and correct label / **all** test
issues. Prose around JSON, extra keys, unknown labels, and malformed JSON are
invalid and count as wrong. The parser allows insignificant JSON whitespace.
It does not repair generated answers. Per-label counts make collapsed
predictions easier to spot. Chance or always predicting one label is 25% on
this balanced four-label test set.

This comparison isolates the effect of the adapter under one fixed prompt. It
does not establish that fine-tuning beats better prompting, few-shot examples,
constrained decoding, a larger model, or a conventional classifier. Try those
baselines on validation data. Keep the test set out of iterative tuning; if
you repeatedly inspect and optimize for it, collect a new final test set.

For real triage, add longer reports, overlapping labels, code snippets,
non-English issues, unclear requests, and adversarial instructions. Split
related reports and duplicates together, and prefer a time-based holdout from
the actual project. Require human review and validate outputs before applying
labels; valid JSON does not mean the decision is correct.

## Learn more

- [TRL](https://huggingface.co/docs/trl/en/index)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [Dataset formats](https://huggingface.co/docs/trl/en/dataset_formats)
- [PEFT / LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [Chat templates](https://huggingface.co/docs/transformers/en/chat_templating)

The documentation's default version can be newer than this pinned example.
The commands and saved results here describe the tested environment.
