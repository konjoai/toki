"""Tests for toki.__main__ — python -m toki CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from toki.__main__ import main


def test_generate_command_runs(capsys):
    main(["generate", "--count", "5", "--seed", "42"])
    captured = capsys.readouterr()
    assert "Generated" in captured.out or "prompts" in captured.out.lower()


def test_generate_with_output_saves_file(tmp_path, capsys):
    out_file = str(tmp_path / "prompts.json")
    main(["generate", "--count", "5", "--seed", "42", "--output", out_file])
    assert Path(out_file).exists()
    data = json.loads(Path(out_file).read_text())
    assert isinstance(data, list)
    assert len(data) > 0
    captured = capsys.readouterr()
    assert "Saved" in captured.out


def test_evaluate_command_runs(capsys):
    main(["evaluate", "--seed", "42"])
    captured = capsys.readouterr()
    assert "Total prompts" in captured.out
    assert "Mean score" in captured.out


def test_run_command_runs(tmp_path, capsys):
    main([
        "run",
        "--name", "cli_test",
        "--model", "mock",
        "--seed", "42",
        "--output-dir", str(tmp_path),
    ])
    captured = capsys.readouterr()
    assert "Experiment" in captured.out
    assert "Pre-score" in captured.out


def test_list_command_empty_dir(tmp_path, capsys):
    main(["list", "--dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert "No experiments found" in captured.out


def test_list_command_finds_experiment(tmp_path, capsys):
    # First run an experiment so there is something to list
    main([
        "run",
        "--name", "list_test",
        "--model", "mock",
        "--seed", "42",
        "--output-dir", str(tmp_path),
    ])
    # Clear captured output from run
    capsys.readouterr()

    # Now list
    main(["list", "--dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert "list_test" in captured.out


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["unknown_command_xyz"])


def test_pipeline_command_runs(tmp_path, capsys):
    main([
        "pipeline",
        "--name", "cli_pipeline",
        "--seed", "5",
        "--iterations", "2",
        "--convergence-threshold", "0.95",
        "--convergence-window", "2",
        "--jailbreak-count", "2",
        "--injection-count", "2",
        "--boundary-count", "1",
        "--output-dir", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert "Pipeline:" in out
    assert "cli_pipeline" in out
    assert "Final score" in out
    # Safe-mock baseline → converges in `window` rounds
    assert "Converged:   True" in out
    # Pipeline artifact written
    found = list(Path(tmp_path).glob("*_cli_pipeline/pipeline.json"))
    assert len(found) == 1


def test_upload_dry_run_writes_card(tmp_path, capsys):
    """The upload --dry-run path should render a dataset card locally with no HF imports."""
    # Build a dataset on disk
    main([
        "generate",
        "--count", "3",
        "--seed", "7",
        "--output", str(tmp_path / "ds.json"),
    ])
    capsys.readouterr()

    card_path = tmp_path / "CARD.md"
    main([
        "upload",
        "--dataset", str(tmp_path / "ds.json"),
        "--repo", "user/toki-adv",
        "--version", "0.4.0",
        "--dry-run",
        "--output-card", str(card_path),
    ])
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "user/toki-adv" in out
    assert card_path.exists()
    text = card_path.read_text()
    assert text.startswith("---\n")
    assert "dataset_version: 0.4.0" in text
