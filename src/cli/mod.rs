//! Command-line interface for `chrom-rs`.
//!
//! Assembles the `dynamic-cli` application from a declarative YAML
//! configuration embedded at compile time and wires it to the simulation
//! pipeline defined in the [`app`](crate::cli::app) module.
//!
//! # Entry point
//!
//! ```rust,no_run
//! chrom_rs::cli::build_app()
//!     .expect("CLI initialisation failed")
//!     .run();
//! ```
//!
//! # Command surface (v0.6.0, in progress)
//!
//! ```text
//! chrom-rs run [--project-dir <dir>]
//!              --model    <file.yml>              (or --source model    file=<file.yml>)
//!              --scenario <file.yml>              (or --source scenario file=<file.yml>)
//!              --solver   <file.yml>              (or --source solver   file=<file.yml>)
//!              [--output-csv   <file.csv>]        (or --output csv  file=<file.csv>)
//!              [--output-plot  <file.png|.svg>]   (or --output png  file=<file.png>,
//!                                                      --output svg  file=<file.svg>)
//!              [--export-json  <file.json>]       (or --output json file=<file.json>)
//!
//! chrom-rs check [--project-dir <dir>]
//!                [--source model    file=<file.yml>]
//!                [--source scenario file=<file.yml>]
//!                [--source solver   file=<file.yml>]
//!
//! chrom-rs config (alias build)
//!                  [--model single lambda=... langmuir-k=... port-number=...
//!                                  column-length=... n-points=... dz=... fe=... ue=...]
//!                  [--model multi n-points=... porosity=... velocity=...
//!                                 column-length=... dz=... fe=... ue=... stationary-fraction=...]
//!                  [--model species name=... lambda=... langmuir-k=... port-number=...]
//! ```
//!
//! `run` accepts either the legacy scalar options or the repeatable
//! `--source`/`--output` syntax for each role — never both for the same
//! role in the same invocation. `check` only has the repeatable syntax,
//! and every role is optional: with none given it just lists the project
//! directory's config-like files; with any given, it validates exactly
//! those. Both draw on `dynamic-cli` 0.6.0's repeatable-option-with-
//! sub-parameters feature (see [dcli#21](https://github.com/biface/dcli/issues/21)
//! for that feature's own design rationale on the `dynamic-cli` side).
//!
//! `config` (DD-016, [#53](https://github.com/biface/chromatography/issues/53))
//! accumulates unvalidated configuration state across one or more
//! occurrences — in a single invocation or across a chained sequence
//! (`dynamic-cli` 0.9.0 command chaining, DD-026) — into
//! [`ChromContext`](crate::cli::app::ChromContext)'s pending slots.
//! `--model single`/`multi`/`species` are wired up
//! ([#68](https://github.com/biface/chromatography/issues/68),
//! [#69](https://github.com/biface/chromatography/issues/69)); `--solver`
//! and `--scenario` land in later commits (#70–#72), and nothing is
//! validated until a future `save`/`run`.

/// Execution context, command handlers, and simulation helpers.
///
/// All runtime state ([`ChromContext`](crate::cli::app::ChromContext)),
/// input validation, and the `run` command handler
/// ([`RunHandler`](crate::cli::app::RunHandler)) live here.
pub mod app;

/// The `check` command handler ([`CheckHandler`](crate::cli::check::CheckHandler)) —
/// validates configuration files without running a simulation.
pub mod check;

/// Pending-configuration builder types for the interactive `config`/`build`
/// command — accumulated, unvalidated state held in
/// [`ChromContext`](crate::cli::app::ChromContext) across chained
/// invocations. See DD-016 (issue #53) and issue #67.
pub mod builders;

/// The `config`/`build` command handler
/// ([`ConfigHandler`](crate::cli::config_handler::ConfigHandler)) — builds
/// model/solver/scenario configuration interactively. See issue #68 (and
/// #69–72 for the parts not wired up yet).
pub mod config_handler;

use anyhow::anyhow;
use dynamic_cli::config::loader::load_yaml;
use dynamic_cli::{CliApp, CliBuilder};

use app::{ChromContext, RunHandler};
use check::CheckHandler;
use config_handler::ConfigHandler;

// ============================================================================
// Embedded command configuration
// ============================================================================

/// YAML command configuration, embedded at compile time from
/// `src/cli/commands.yml`.
///
/// Parsed once in [`build_app`] via `load_yaml`. Keeping the declarations in
/// YAML lets maintainers adjust help text, aliases, and option metadata
/// without touching Rust code.
const COMMANDS_YML: &str = include_str!("commands.yml");

/// Handler name that must match the `implementation:` field of the `run`
/// command in `commands.yml`.
const RUN_HANDLER_NAME: &str = "run_handler";

/// Handler name that must match the `implementation:` field of the `check`
/// command in `commands.yml`.
const CHECK_HANDLER_NAME: &str = "check_handler";

/// Handler name that must match the `implementation:` field of the `config`
/// command in `commands.yml`.
const CONFIG_HANDLER_NAME: &str = "config_handler";

// ============================================================================
// build_app
// ============================================================================

/// Assembles and returns the fully configured [`CliApp`].
///
/// Parses the embedded command YAML, wires
/// [`RunHandler`], [`CheckHandler`], and a fresh
/// [`ChromContext`], then delegates to
/// `CliBuilder::build`.
///
/// # Errors
///
/// - The embedded YAML is malformed (compile-time regression).
/// - The builder detects a missing required handler.
pub fn build_app() -> anyhow::Result<CliApp> {
    let config =
        load_yaml(COMMANDS_YML).map_err(|e| anyhow!("embedded commands.yml is invalid: {e}"))?;

    CliBuilder::new()
        .config(config)
        .context(Box::new(ChromContext::new()))
        .register_sync_handler(RUN_HANDLER_NAME, Box::new(RunHandler))
        .register_sync_handler(CHECK_HANDLER_NAME, Box::new(CheckHandler))
        .register_sync_handler(CONFIG_HANDLER_NAME, Box::new(ConfigHandler))
        .build()
        .map_err(|e| anyhow!("CLI builder error: {e}"))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_app_succeeds() {
        assert!(build_app().is_ok());
    }

    #[test]
    fn test_commands_yml_is_valid_yaml() {
        use dynamic_cli::config::loader::load_yaml;
        let config = load_yaml(COMMANDS_YML).expect("COMMANDS_YML must be valid");
        assert!(config.commands.iter().any(|c| c.name == "run"));
        assert!(config.commands.iter().any(|c| c.name == "check"));
        assert!(config.commands.iter().any(|c| c.name == "config"));
    }
}
