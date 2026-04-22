//! Integration tests for the `cli/` module.
//!
//! These tests exercise [`RunHandler::execute`] end-to-end with real
//! configuration files, covering the paths not reachable from unit tests:
//!
//! - `read_model_file` / `peek_root_key` / `deserialise_inner`
//! - `resolve_species_names` / `resolve_export_map`
//! - `RunHandler::execute` — single-species and multi-species dispatch
//! - CSV, plot, and JSON output paths
//!
//! Each test creates a temporary project directory populated with copies of
//! the config files from `examples/config/`.  All outputs land in that
//! directory and are cleaned up automatically by `tempfile::TempDir`.
//!
//! `--project-dir` is passed explicitly in the args map so that
//! `RunHandler::execute` resolves all file paths against the temp directory
//! rather than the current working directory.

use std::collections::HashMap;
use std::path::Path;

use chrom_rs::cli::app::{ChromContext, RunHandler};
use dynamic_cli::{CommandHandler, ExecutionContext};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Copies every listed source file into `dest_dir` under its basename.
fn copy_files(files: &[&str], dest_dir: &Path) {
    for src in files {
        let name = Path::new(src).file_name().expect("file must have a name");
        std::fs::copy(src, dest_dir.join(name))
            .unwrap_or_else(|e| panic!("cannot copy {src}: {e}"));
    }
}

/// Builds the `args` map passed to `RunHandler::execute`.
fn make_args(entries: &[(&str, &str)]) -> HashMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Runs the handler with `project-dir` pointing at `dir`.
///
/// `RunHandler::execute` reads `project-dir` from `args` and calls
/// `set_project_dir` internally — the context starts with the default `.`.
fn run(dir: &Path, args: &HashMap<String, String>) {
    let mut ctx = ChromContext::new();
    RunHandler
        .execute(&mut ctx as &mut dyn ExecutionContext, args)
        .unwrap_or_else(|e| panic!("RunHandler::execute failed: {e}"));
    // suppress unused warning
    let _ = dir;
}

// ── Single-species (LangmuirSingle / TFA) ────────────────────────────────────

/// Gaussian injection + RK4 → single-species dispatch, CSV output.
#[test]
fn test_run_tfa_gaussian_rk4_csv() {
    let dir = tempfile::tempdir().unwrap();
    copy_files(
        &[
            "examples/config/tfa/model.yml",
            "examples/config/tfa/scenario_gaussian.yml",
            "examples/config/tfa/solver_rk4.yml",
        ],
        dir.path(),
    );

    let project_dir = dir.path().to_str().unwrap();
    let args = make_args(&[
        ("project-dir", project_dir),
        ("model", "model.yml"),
        ("scenario", "scenario_gaussian.yml"),
        ("solver", "solver_rk4.yml"),
        ("output-csv", "result.csv"),
    ]);

    run(dir.path(), &args);
    assert!(
        dir.path().join("result.csv").exists(),
        "CSV must be created"
    );
}

/// Gaussian injection + Euler → single-species dispatch, JSON export.
#[test]
fn test_run_tfa_gaussian_euler_json() {
    let dir = tempfile::tempdir().unwrap();
    copy_files(
        &[
            "examples/config/tfa/model.yml",
            "examples/config/tfa/scenario_gaussian.yml",
            "examples/config/tfa/solver_euler.yml",
        ],
        dir.path(),
    );

    let project_dir = dir.path().to_str().unwrap();
    let args = make_args(&[
        ("project-dir", project_dir),
        ("model", "model.yml"),
        ("scenario", "scenario_gaussian.yml"),
        ("solver", "solver_euler.yml"),
        ("export-json", "result.json"),
    ]);

    run(dir.path(), &args);
    assert!(
        dir.path().join("result.json").exists(),
        "JSON must be created"
    );
}

/// Gaussian injection + RK4 → plot output.
#[test]
fn test_run_tfa_gaussian_rk4_plot() {
    let dir = tempfile::tempdir().unwrap();
    copy_files(
        &[
            "examples/config/tfa/model.yml",
            "examples/config/tfa/scenario_gaussian.yml",
            "examples/config/tfa/solver_rk4.yml",
        ],
        dir.path(),
    );

    let project_dir = dir.path().to_str().unwrap();
    let args = make_args(&[
        ("project-dir", project_dir),
        ("model", "model.yml"),
        ("scenario", "scenario_gaussian.yml"),
        ("solver", "solver_rk4.yml"),
        ("output-plot", "result.png"),
    ]);

    run(dir.path(), &args);
    assert!(
        dir.path().join("result.png").exists(),
        "plot must be created"
    );
}

/// All three outputs at once — full single-species output chain.
#[test]
fn test_run_tfa_all_outputs() {
    let dir = tempfile::tempdir().unwrap();
    copy_files(
        &[
            "examples/config/tfa/model.yml",
            "examples/config/tfa/scenario_gaussian.yml",
            "examples/config/tfa/solver_rk4.yml",
        ],
        dir.path(),
    );

    let project_dir = dir.path().to_str().unwrap();
    let args = make_args(&[
        ("project-dir", project_dir),
        ("model", "model.yml"),
        ("scenario", "scenario_gaussian.yml"),
        ("solver", "solver_rk4.yml"),
        ("output-csv", "result.csv"),
        ("output-plot", "result.png"),
        ("export-json", "result.json"),
    ]);

    run(dir.path(), &args);
    assert!(
        dir.path().join("result.csv").exists(),
        "CSV must be created"
    );
    assert!(
        dir.path().join("result.png").exists(),
        "plot must be created"
    );
    assert!(
        dir.path().join("result.json").exists(),
        "JSON must be created"
    );
}

// ── Multi-species (LangmuirMulti / Acids competitive) ────────────────────────

/// Acids competitive + RK4 → multi-species dispatch, CSV output.
#[test]
fn test_run_acids_multi_rk4_csv() {
    let dir = tempfile::tempdir().unwrap();
    copy_files(
        &[
            "examples/config/acids/model.yml",
            "examples/config/acids/scenario_gaussian.yml",
            "examples/config/acids/solver_rk4.yml",
        ],
        dir.path(),
    );

    let project_dir = dir.path().to_str().unwrap();
    let args = make_args(&[
        ("project-dir", project_dir),
        ("model", "model.yml"),
        ("scenario", "scenario_gaussian.yml"),
        ("solver", "solver_rk4.yml"),
        ("output-csv", "result.csv"),
    ]);

    run(dir.path(), &args);
    assert!(
        dir.path().join("result.csv").exists(),
        "multi-species CSV must be created"
    );
}

/// Acids competitive + Euler → multi-species dispatch, all outputs.
#[test]
fn test_run_acids_multi_euler_all_outputs() {
    let dir = tempfile::tempdir().unwrap();
    copy_files(
        &[
            "examples/config/acids/model.yml",
            "examples/config/acids/scenario_gaussian.yml",
            "examples/config/acids/solver_euler.yml",
        ],
        dir.path(),
    );

    let project_dir = dir.path().to_str().unwrap();
    let args = make_args(&[
        ("project-dir", project_dir),
        ("model", "model.yml"),
        ("scenario", "scenario_gaussian.yml"),
        ("solver", "solver_euler.yml"),
        ("output-csv", "result.csv"),
        ("output-plot", "result.png"),
        ("export-json", "result.json"),
    ]);

    run(dir.path(), &args);
    assert!(
        dir.path().join("result.csv").exists(),
        "multi-species CSV must be created"
    );
    assert!(
        dir.path().join("result.png").exists(),
        "multi-species plot must be created"
    );
    assert!(
        dir.path().join("result.json").exists(),
        "multi-species JSON must be created"
    );
}

// ── Error paths ───────────────────────────────────────────────────────────────

/// Missing required option returns an error.
#[test]
fn test_run_missing_model_arg_fails() {
    let dir = tempfile::tempdir().unwrap();
    let project_dir = dir.path().to_str().unwrap();

    let args = make_args(&[
        ("project-dir", project_dir),
        // "model" intentionally absent
        ("scenario", "scenario.yml"),
        ("solver", "solver.yml"),
    ]);

    let mut ctx = ChromContext::new();
    let result = RunHandler.execute(&mut ctx as &mut dyn ExecutionContext, &args);
    assert!(result.is_err(), "missing --model must return an error");
}

/// Path traversal in project-dir is rejected.
#[test]
fn test_run_project_dir_traversal_fails() {
    let args = make_args(&[
        ("project-dir", "../etc"),
        ("model", "model.yml"),
        ("scenario", "scenario.yml"),
        ("solver", "solver.yml"),
    ]);

    let mut ctx = ChromContext::new();
    let result = RunHandler.execute(&mut ctx as &mut dyn ExecutionContext, &args);
    assert!(result.is_err(), "path traversal must return an error");
}
