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
