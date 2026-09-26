import json
from dataclasses import asdict, replace

import pytest

from ecg_repair.demo import demo_inputs, run_demo
from ecg_repair.pipeline import build_repair_agent, index_records, repair_reports
from ecg_repair.cli import main


def test_demo_uses_real_repair_components_and_is_auditable():
    result = run_demo()["results"][0]
    assert result["committed_actions"] == 1
    assert result["release_audit"]["passed"]


def test_no_memory_does_not_authorize_diagnostic_addition():
    reports, instruments, _ = demo_inputs()
    result = repair_reports(reports, instruments, candidate_claims=("pr_prolonged",))[0]
    assert result["committed_actions"] == 0
    assert result["release_audit"]["passed"]


def test_nonpositive_advantage_preserves_diagnoses():
    reports, instruments, memory = demo_inputs()
    memory = [replace(row, reward=-4.0) for row in memory]
    result = repair_reports(reports, instruments, memory, candidate_claims=("pr_prolonged",))[0]
    assert result["committed_actions"] == 0


def test_query_fold_is_excluded_before_fitting():
    reports, instruments, memory = demo_inputs()
    memory = [replace(row, fold=0) for row in memory]
    result = repair_reports(reports, instruments, memory, candidate_claims=("pr_prolonged",))[0]
    assert result["committed_actions"] == 0


def test_query_record_in_other_fold_is_rejected():
    reports, instruments, memory = demo_inputs()
    memory[0] = replace(memory[0], record_id=reports[0]["record_id"])
    with pytest.raises(ValueError, match="query record"):
        repair_reports(reports, instruments, memory)


def test_other_query_folds_can_supply_historical_outcomes():
    reports, instruments, memory = demo_inputs()
    reports.append({**reports[0], "record_id": memory[0].record_id, "fold": 1})
    agent = build_repair_agent(
        index_records(reports), index_records(instruments), memory, query_fold=0,
    )
    assert 0 not in agent.outcome_policy.policies[0].memory_folds


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        index_records([{"record_id": "a"}, {"record_id": "a"}])


def test_missing_instrument_is_not_silently_skipped():
    reports, _, memory = demo_inputs()
    with pytest.raises(ValueError, match="missing instruments"):
        repair_reports(reports, [], memory)


def test_file_based_cli_roundtrip_and_no_overwrite(tmp_path):
    reports, instruments, memory = demo_inputs()
    paths = [tmp_path / name for name in ("reports.jsonl", "instruments.jsonl", "memory.jsonl")]
    history = []
    for episode in memory:
        row = asdict(episode)
        row["selected_claims"] = sorted(row["selected_claims"])
        history.append(row)
    for path, rows in zip(paths, (reports, instruments, history)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "repaired.jsonl"
    args = ["repair", "--reports", str(paths[0]), "--instruments", str(paths[1]),
            "--memory", str(paths[2]), "--output", str(output)]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result["release_audit"]["passed"]
    assert result["committed_actions"] == 1
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == original
