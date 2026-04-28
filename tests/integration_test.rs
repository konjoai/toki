use std::process::Command;

#[test]
#[ignore]
fn test_cli_config_command() {
    let output = Command::new("cargo")
        .args(["run", "--", "config"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to run toki");
    assert!(
        output.status.success(),
        "toki config failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("model_name"),
        "expected model_name in config output"
    );
}

#[test]
#[ignore]
fn test_cli_generate_command() {
    let output = Command::new("cargo")
        .args(["run", "--", "generate", "--count", "5"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to run toki");
    assert!(
        output.status.success(),
        "toki generate failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore]
fn test_cli_pipeline_command() {
    let output = Command::new("cargo")
        .args(["run", "--", "pipeline"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to run toki");
    assert!(
        output.status.success(),
        "toki pipeline failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Pipeline complete"));
}
