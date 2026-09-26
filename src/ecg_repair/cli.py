"""Portable local commands; no embedded model endpoint or credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import load_memory, read_jsonl, repair_reports


def _write_new(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ecg-repair")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="run a synthetic verification and memory/repair example")
    repair = commands.add_parser("repair", help="repair stored reports with user-supplied memory")
    repair.add_argument("--reports", type=Path, required=True)
    repair.add_argument("--instruments", type=Path, required=True)
    repair.add_argument("--memory", type=Path)
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--alpha", type=float, default=100.0)
    repair.add_argument("--margin", type=float, default=2.0)
    repair.add_argument("--base-model-id", default="base-ecg-model")
    measure = commands.add_parser("measure", help="measure canonical ECG NPZ files (requires [ecg])")
    measure.add_argument("inputs", type=Path, nargs="+")
    measure.add_argument("--output", type=Path, required=True)
    retrieve = commands.add_parser("retrieve", help="search a user-supplied passage index")
    retrieve.add_argument("--index", type=Path, required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    if hasattr(args, "output") and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.command == "demo":
        from .demo import run_demo
        print(json.dumps(run_demo(), indent=2, allow_nan=False))
    elif args.command == "repair":
        results = repair_reports(
            read_jsonl(args.reports), read_jsonl(args.instruments),
            load_memory(args.memory) if args.memory else (),
            alpha=args.alpha, margin=args.margin, base_model_id=args.base_model_id,
        )
        _write_new(args.output, results)
        print(f"Wrote {len(results)} repair records to {args.output}")
    elif args.command == "measure":
        from ecgcf.io.storage import load_npz
        from .tools import measure_ecg
        _write_new(args.output, (measure_ecg(load_npz(path)) for path in args.inputs))
        print(f"Wrote measurements to {args.output}")
    elif args.command == "retrieve":
        from dataclasses import asdict
        from ecg_agent.clinical_rag.index import ClinicalKnowledgeIndex
        if args.top_k <= 0:
            parser.error("--top-k must be positive")
        index = ClinicalKnowledgeIndex.from_jsonl([args.index])
        print(json.dumps([asdict(hit) for hit in index.retrieve(args.query, k=args.top_k)], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
