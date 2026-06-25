import json
from pathlib import Path

import pytest

from safeguard_harness.datasets import load_jsonl_cases


def test_load_jsonl_cases_accepts_messages_format(tmp_path: Path):
    dataset_path = tmp_path / "messages.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": 7,
                "type": "输出侧",
                "is_mt": 0,
                "label": "unsafe",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "请回答这个问题。"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "这是待评估的回答。"}],
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_jsonl_cases(dataset_path)

    assert len(cases) == 1
    case = cases[0]
    assert case.id == "7"
    assert case.question == "请回答这个问题。"
    assert case.answer == "这是待评估的回答。"
    assert case.label == "unsafe"
    assert case.metadata["raw_id"] == 7
    assert case.metadata["source_format"] == "messages"
    assert case.metadata["type"] == "输出侧"
    assert case.metadata["is_mt"] == 0
    assert case.metadata["messages"][0]["role"] == "user"


def test_load_jsonl_cases_normalizes_mt_alias(tmp_path: Path):
    dataset_path = tmp_path / "messages_mt.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "mt",
                "type": "输出侧",
                "MT": 1,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Q"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    case = load_jsonl_cases(dataset_path)[0]

    assert case.metadata["MT"] == 1
    assert case.metadata["is_mt"] == 1


def test_load_jsonl_cases_extracts_images_from_messages(tmp_path: Path):
    dataset_path = tmp_path / "messages_image.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "img",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_id", "image_id": "/tmp/demo.jpg"},
                            {"type": "text", "text": "这张图安全吗？"},
                        ],
                    }
                ],
                "source": "demo_source",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    case = load_jsonl_cases(dataset_path)[0]

    assert case.question == "这张图安全吗？"
    assert case.modality == "image"
    assert case.attachments == ["/tmp/demo.jpg"]
    assert case.has_image() is True
    assert case.metadata["source"] == "demo_source"


def test_load_jsonl_cases_rejects_messages_without_user_text(tmp_path: Path):
    dataset_path = tmp_path / "bad_messages.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "bad",
                "label": "safe",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "missing user prompt"}],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case requires a non-empty question"):
        load_jsonl_cases(dataset_path)


def test_load_jsonl_cases_accepts_json_array_with_same_case_shape(tmp_path: Path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "safe", "question": "hello", "label": "safe"},
                {"id": 2, "question": "bad", "label": "unsafe"},
            ]
        ),
        encoding="utf-8",
    )

    cases = load_jsonl_cases(dataset_path)

    assert [case.id for case in cases] == ["safe", "2"]
    assert [case.label for case in cases] == ["safe", "unsafe"]
    assert cases[1].metadata["raw_id"] == 2


def test_load_jsonl_cases_accepts_json_object_wrapping_cases(tmp_path: Path):
    dataset_path = tmp_path / "wrapped.json"
    dataset_path.write_text(
        json.dumps({"cases": [{"id": "c1", "question": "hello", "label": "safe"}]}),
        encoding="utf-8",
    )

    cases = load_jsonl_cases(dataset_path)

    assert len(cases) == 1
    assert cases[0].id == "c1"
