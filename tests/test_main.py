

# ---------------------------------------------------------------------------
# monitor CLI (Sprint 21)
# ---------------------------------------------------------------------------


def test_monitor_command_detects_regression(tmp_path, capsys):
    main([
        "monitor", "--model", "unsafe", "--reference", "safe",
        "--output-dir", str(tmp_path),
    ])
    captured = capsys.readouterr()
    assert "REGRESSION" in captured.out


def test_monitor_command_no_regression(tmp_path, capsys):
    main([
        "monitor", "--model", "safe", "--reference", "safe",
        "--output-dir", str(tmp_path),
    ])
    captured = capsys.readouterr()
    assert "status: ok" in captured.out


def test_monitor_command_json(tmp_path, capsys):
    import json as _json

    main([
        "monitor", "--model", "unsafe", "--json",
        "--output-dir", str(tmp_path),
    ])
    captured = capsys.readouterr()
    data = _json.loads(captured.out)
    assert data["regressed"] is True
    assert data["name"] == "safety_monitor"
