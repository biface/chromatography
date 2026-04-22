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
//! # Command surface (v0.2.0)
//!
//! ```text
//! chrom-rs run [--project-dir <dir>]
//!              --model    <file.yml>
//!              --scenario <file.yml>
//!              --solver   <file.yml>
//!              [--output-csv   <file.csv>]
//!              [--output-plot  <file.png>]
//!              [--export-json  <file.json>]
//! ```

/// Execution context, command handlers, and simulation helpers.
///
/// All runtime state ([`ChromContext`](crate::cli::app::ChromContext)),
/// input validation, and the `run` command handler
/// ([`RunHandler`](crate::cli::app::RunHandler)) live here.
pub mod app;

use anyhow::anyhow;
use dynamic_cli::config::loader::load_yaml;
use dynamic_cli::{CliApp, CliBuilder};

use app::{ChromContext, RunHandler};

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

// ============================================================================
// build_app
// ============================================================================

/// Assembles and returns the fully configured [`CliApp`].
///
/// Parses the embedded command YAML, wires
/// [`RunHandler`] and a fresh
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
        .register_handler(RUN_HANDLER_NAME, Box::new(RunHandler))
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
    }
}
