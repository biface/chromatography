//! Command-line interface for `chrom-rs` (DD-001).
//!
//! See [`app`] for the execution context, command handlers, and helpers.
//!
//! # Entry point
//!
//! ```rust,no_run
//! fn main() {
//!     chrom_rs::cli::build_app()
//!         .expect("CLI initialisation failed")
//!         .run();
//! }
//! ```

pub mod app;

use anyhow::anyhow;
use dynamic_cli::config::loader::load_yaml;
use dynamic_cli::{CliApp, CliBuilder};

use app::{ChromContext, RunHandler};

// ============================================================================
// Embedded command configuration
// ============================================================================

/// Content of `src/cli/commands.yml`, embedded at compile time.
///
/// Parsed once in [`build_app`] via [`load_yaml`]. Keeping the declarations
/// in YAML lets maintainers adjust help text, aliases, and option metadata
/// without touching Rust code.
const COMMANDS_YML: &str = include_str!("commands.yml");

/// Implementation name — must match the `implementation:` field in
/// `commands.yml` for the `run` command.
const RUN_HANDLER_NAME: &str = "run_handler";

// ============================================================================
// build_app
// ============================================================================

/// Assembles and returns the fully configured [`CliApp`].
///
/// Parses the embedded [`COMMANDS_YML`], wires [`RunHandler`] and a fresh
/// [`ChromContext`], then builds the application.
///
/// # Errors
///
/// Returns an error if the embedded YAML is malformed or if the builder
/// detects a missing required handler.
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
        // Vérifie que le YAML embarqué est syntaxiquement correct
        // et contient la commande `run`.
        use dynamic_cli::config::loader::load_yaml;
        let config = load_yaml(COMMANDS_YML).expect("COMMANDS_YML must be valid");
        assert!(config.commands.iter().any(|c| c.name == "run"));
    }
}
