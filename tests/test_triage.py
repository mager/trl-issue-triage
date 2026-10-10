import pytest
from triage import LABELS, load_split, parse_label, score, target, training_rows


@pytest.mark.parametrize("text", [
    'Here is the answer: {"label":"bug"}', '```json\n{"label":"bug"}\n```',
    '{"label":"bug","reason":"crash"}', '{"label":"urgent"}',
    '{"label":[]}', '["bug"]', 'null', '{}', '{broken',
])
def test_rejects_invalid_output(text):
    assert parse_label(text) is None


def test_valid_labels_and_whitespace():
    for label in LABELS:
        assert parse_label("\n" + target(label) + " ") == label


def test_data_contract_and_no_cross_split_leakage():
    texts, ids = set(), set()
    for split, count in (("train", 80), ("validation", 12), ("test", 20)):
        rows = load_split(split)
        assert len(rows) == count
        assert {r["label"] for r in rows} == set(LABELS)
        for row in rows:
            assert row["id"] not in ids
            assert row["text"].casefold().strip() not in texts
            ids.add(row["id"])
            texts.add(row["text"].casefold().strip())
            assert row["label"] in LABELS and row["text"].strip()


def test_invalid_predictions_count_as_wrong_not_missing():
    metrics = score([
        {"expected": "bug", "predicted": "bug"},
        {"expected": "feature", "predicted": "docs"},
        {"expected": "docs", "predicted": None},
    ])
    assert metrics["accuracy"] == 1 / 3
    assert metrics["valid_json_rate"] == 2 / 3
    assert metrics["per_label"]["docs"] == {"total": 1, "correct": 0}


def test_training_format_keeps_target_out_of_prompt():
    row = training_rows([{"text": "The app crashes.", "label": "bug"}])[0]
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["prompt"][-1]["content"] == "The app crashes."
    assert row["completion"] == [{"role": "assistant", "content": '{"label":"bug"}'}]


def test_empty_evaluation_is_an_error():
    with pytest.raises(ValueError):
        score([])
