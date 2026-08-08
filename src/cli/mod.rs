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
//! # Command surface (v0.5.0)
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

/// Execution context, command handlers, and simulation helpers.
///
/// All runtime state ([`ChromContext`](crate::cli::app::ChromContext)),
/// input validation, and the `run` command handler
/// ([`RunHandler`](crate::cli::app::RunHandler)) live here.
pub mod app;

/// The `check` command handler ([`CheckHandler`](crate::cli::check::CheckHandler)) —
/// validates configuration files without running a simulation.
pub mod check;

use anyhow::anyhow;
use dynamic_cli::config::loader::load_yaml;
use dynamic_cli::{CliApp, CliBuilder};

use app::{ChromContext, RunHandler};
use check::CheckHandler;

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
    }
}
