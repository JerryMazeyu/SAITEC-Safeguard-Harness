from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from safeguard_harness.config import load_pipeline
from safeguard_harness.core import SafetyCase
from safeguard_harness.datasets import deliverable_result_row, load_jsonl_cases, write_jsonl
from safeguard_harness.evaluation import evaluate_dataset
from safeguard_harness.progress import TerminalProgress


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safeguard-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge = subparsers.add_parser("judge", help="Judge one question through a configured pipeline.")
    judge.add_argument("--pipeline", required=True)
    judge.add_argument("--question", required=True)
    judge.add_argument("--answer")
    judge.add_argument("--image", action="append", default=[])
    judge.add_argument("--id", default="case")
    judge.set_defaults(func=cmd_judge)

    predict = subparsers.add_parser("predict", help="Run batch prediction over a JSON or JSONL file.")
    predict.add_argument("--pipeline", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--deliverable-output")
    predict.set_defaults(func=cmd_predict)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a pipeline on labeled JSON or JSONL data.")
    evaluate.add_argument("--pipeline", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--deliverable-output")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def cmd_judge(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args.pipeline)
    case = SafetyCase.from_dict(
        {
            "id": args.id,
            "question": args.question,
            "answer": args.answer,
            "attachments": args.image,
            "metadata": {"has_image": bool(args.image)},
        }
    )
    decision = pipeline.judge(case)
    print(json.dumps(decision.to_dict(), ensure_ascii=False))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args.pipeline)
    cases = load_jsonl_cases(args.input)
    rows = []
    deliverable_rows = []
    terminal_progress = TerminalProgress("predict", len(cases))
    terminal_progress.start()
    processed = 0
    try:
        decisions_iter = pipeline.judge_many(cases, intermediate_dir=_default_intermediate_dir(args.output))
        for processed, (case, decision) in enumerate(zip(cases, decisions_iter), start=1):
            terminal_progress.update(processed, current=f"case={case.id}")
            rows.append({"case_id": case.id, **decision.to_dict()})
            deliverable_rows.append(deliverable_result_row(case, decision))
    except Exception as exc:
        terminal_progress.fail(processed=processed, error=f"{type(exc).__name__}: {exc}")
        raise
    terminal_progress.finish()
    write_jsonl(args.output, rows)
    deliverable_output = Path(args.deliverable_output) if args.deliverable_output else _default_deliverable_path(args.output)
    write_jsonl(deliverable_output, deliverable_rows)
    print(f"wrote {len(rows)} predictions to {Path(args.output)}")
    print(f"wrote {len(deliverable_rows)} deliverable results to {deliverable_output}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args.pipeline)
    cases = load_jsonl_cases(args.dataset)
    summary = evaluate_dataset(
        pipeline,
        cases,
        args.output,
        config_snapshot=pipeline.raw_config,
        deliverable_output=args.deliverable_output,
    )
    print(json.dumps({"accuracy": summary.metrics["accuracy"], "metrics": summary.metrics}, ensure_ascii=False))
    return 0


def _default_deliverable_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    suffix = output.suffix or ".jsonl"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_deliverable{suffix}")


def _default_intermediate_dir(output_path: str | Path) -> Path:
    output = Path(output_path)
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_intermediate")
