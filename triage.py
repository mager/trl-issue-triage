"""Shared task contract. This module deliberately needs only the standard library."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
LABELS = ("bug", "feature", "question", "docs")
SYSTEM = (
    'Label this software issue. Return only a JSON object with one key, "label". '
    'Allowed labels: bug (existing behavior is broken), feature (new behavior requested), '
    'question (asking for help), docs (documentation correction or addition). '
    'For documentation problems choose docs, even if the issue describes an error. '
    'Example output: {"label":"bug"}'
)


def messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def target(label):
    return json.dumps({"label": label}, separators=(",", ":"))


def parse_label(text):
    """Strict validation: no extracting JSON from prose or accepting extra keys."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"label"}:
        return None
    label = value["label"]
    return label if isinstance(label, str) and label in LABELS else None


def load_split(name):
    if name not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {name}")
    return [json.loads(line) for line in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines()]


def training_rows(rows):
    return [{"prompt": messages(row["text"]),
             "completion": [{"role": "assistant", "content": target(row["label"])}]}
            for row in rows]


def score(rows):
    if not rows:
        raise ValueError("Cannot score an empty evaluation")
    valid = sum(row["predicted"] is not None for row in rows)
    correct = sum(row["predicted"] == row["expected"] for row in rows)
    return {"total": len(rows), "valid_json": valid, "correct": correct,
            "valid_json_rate": valid / len(rows), "accuracy": correct / len(rows),
            "per_label": {label: {
                "total": sum(r["expected"] == label for r in rows),
                "correct": sum(r["expected"] == label and r["predicted"] == label for r in rows),
            } for label in LABELS}}
