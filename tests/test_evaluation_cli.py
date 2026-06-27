import json
import os
import subprocess
import sys
from pathlib import Path

from safeguard_harness.config import load_pipeline
from safeguard_harness.core import Decision, SafetyCase
from safeguard_harness.datasets import load_jsonl_cases
from safeguard_harness.evaluation import evaluate_dataset
from safeguard_harness.orchestration import Pipeline


def write_pipeline(path: Path) -> None:
    path.write_text(
        """
runner: static
methods:
  rules:
    type: dictionary
    high_risk_terms: ["steal token"]
    review_terms: ["bypass"]
  llm:
    type: prompt_binary_model
    prompt_template: "Judge: {question}"
    unsafe_keywords: ["bypass"]
steps:
  - id: rules
    method: rules
    on_unsafe: stop
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )


def test_evaluate_dataset_writes_metrics_predictions_and_error_slices(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    dataset_path = tmp_path / "cases.jsonl"
    output_dir = tmp_path / "run"
    write_pipeline(pipeline_path)
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "safe", "question": "hello", "label": "safe"}),
                json.dumps({"id": 2, "question": "steal token now", "label": "unsafe"}),
                json.dumps({"id": 3, "question": "bypass this", "label": "unsafe"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    cases = load_jsonl_cases(dataset_path)
    summary = evaluate_dataset(pipeline, cases, output_dir, config_snapshot={"runner": "static"})

    assert summary.metrics["total"] == 3
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "errors_false_positive.jsonl").exists()
    assert (output_dir / "errors_false_negative.jsonl").exists()
    deliverable_rows = [
        json.loads(line)
        for line in (output_dir / "deliverable.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert deliverable_rows == [
        {"id": "safe", "result": 0},
        {"id": 2, "result": 1},
        {"id": 3, "result": 1},
    ]


def test_evaluate_dataset_streams_predictions_and_progress(tmp_path: Path):
    output_dir = tmp_path / "stream"

    class ObservingPipeline(Pipeline):
        calls = 0

        def __init__(self):
            super().__init__(runner="static", methods={})

        def judge(self, case: SafetyCase) -> Decision:
            if self.calls == 1:
                assert (output_dir / "predictions.jsonl").read_text(encoding="utf-8").count("\n") == 1
                progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
                assert progress["processed"] == 1
                assert progress["status"] == "running"
            self.calls += 1
            return Decision(
                case_id=case.id,
                label=case.label or "safe",
                unsafe_score=1.0 if case.label == "unsafe" else 0.0,
                confidence=1.0,
            )

    cases = [
        SafetyCase(id="safe", question="hello", label="safe"),
        SafetyCase(id="unsafe", question="bad", label="unsafe"),
    ]

    evaluate_dataset(ObservingPipeline(), cases, output_dir)

    assert (output_dir / "predictions.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (output_dir / "deliverable.jsonl").read_text(encoding="utf-8").count("\n") == 2
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress == {"processed": 2, "total": 2, "status": "completed"}


def test_evaluate_dataset_prints_terminal_progress(tmp_path: Path, capsys):
    output_dir = tmp_path / "terminal_progress"

    class SimplePipeline(Pipeline):
        def __init__(self):
            super().__init__(runner="static", methods={})

        def judge(self, case: SafetyCase) -> Decision:
            return Decision(
                case_id=case.id,
                label=case.label or "safe",
                unsafe_score=1.0 if case.label == "unsafe" else 0.0,
                confidence=1.0,
            )

    cases = [
        SafetyCase(id="safe", question="hello", label="safe"),
        SafetyCase(id="unsafe", question="bad", label="unsafe"),
    ]

    evaluate_dataset(SimplePipeline(), cases, output_dir)

    stderr = capsys.readouterr().err
    assert "[evaluate]" in stderr
    assert "2/2" in stderr
    assert "100.0%" in stderr


def test_evaluate_dataset_marks_progress_failed_on_exception(tmp_path: Path):
    output_dir = tmp_path / "failed"

    class FailingPipeline(Pipeline):
        calls = 0

        def __init__(self):
            super().__init__(runner="static", methods={})

        def judge(self, case: SafetyCase) -> Decision:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("model failed to load")
            return Decision(
                case_id=case.id,
                label=case.label or "safe",
                unsafe_score=0.0,
                confidence=1.0,
            )

    cases = [
        SafetyCase(id="safe", question="hello", label="safe"),
        SafetyCase(id="unsafe", question="bad", label="unsafe"),
    ]

    try:
        evaluate_dataset(FailingPipeline(), cases, output_dir)
    except RuntimeError:
        pass
    else:
        raise AssertionError("evaluate_dataset should re-raise pipeline failures")

    assert (output_dir / "predictions.jsonl").read_text(encoding="utf-8").count("\n") == 1
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["processed"] == 1
    assert progress["total"] == 2
    assert progress["status"] == "failed"
    assert "RuntimeError: model failed to load" == progress["error"]


def test_cli_judge_predict_and_evaluate_commands(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    dataset_path = tmp_path / "cases.json"
    predictions_path = tmp_path / "predictions.jsonl"
    deliverable_path = tmp_path / "submission.jsonl"
    run_dir = tmp_path / "eval"
    write_pipeline(pipeline_path)
    dataset_path.write_text(
        json.dumps([{"id": 1, "question": "steal token", "label": "unsafe"}]),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
    }

    judge = subprocess.run(
        [
            sys.executable,
            "-m",
            "safeguard_harness",
            "judge",
            "--pipeline",
            str(pipeline_path),
            "--question",
            "steal token",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    predict = subprocess.run(
        [
            sys.executable,
            "-m",
            "safeguard_harness",
            "predict",
            "--pipeline",
            str(pipeline_path),
            "--input",
            str(dataset_path),
            "--output",
            str(predictions_path),
            "--deliverable-output",
            str(deliverable_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "safeguard_harness",
            "evaluate",
            "--pipeline",
            str(pipeline_path),
            "--dataset",
            str(dataset_path),
            "--output",
            str(run_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert json.loads(judge.stdout)["label"] == "unsafe"
    assert predictions_path.exists()
    assert [json.loads(line) for line in deliverable_path.read_text(encoding="utf-8").splitlines()] == [
        {"id": 1, "result": 1}
    ]
    assert "wrote" in predict.stdout.lower()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "deliverable.jsonl").exists()
    assert "accuracy" in evaluate.stdout.lower()
