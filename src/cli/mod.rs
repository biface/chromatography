//! Command-line interface for `chrom-rs`.
//!
//! Assembles the `dynamic-cli` application from a declarative YAML
//! configuration embedded at compile time and wires it to the simulation
//! pipeline defined in the [`run`](crate::cli::run) module.
//!
//! # Module layout (since the v0.6.0 consolidation)
//!
//! One file per command handler, plus two shared-concern modules:
//!
//! - [`context`]: [`ChromContext`](context::ChromContext) — the execution
//!   context every handler downcasts to, including the three
//!   pending-configuration builder slots (DD-016).
//! - [`support`]: helpers used by more than one handler
//!   (`to_cli_err`/`resolve_source_optional`/`path_to_str`).
//! - [`builders`]: the pending-configuration builder *types* themselves
//!   (`ModelBuilder`, `SolverBuilder`, `ScenarioBuilder`, ...).
//! - [`run`], [`check`], [`config`], [`save`]: one command each.
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
//!                  [--solver RK4 total-time=... time-steps=... [step=...]]
//!                  [--solver Euler total-time=... time-steps=... [step=...]]
//!                  [--initial-condition zero]
//!                  [--injection default type=... center=... width=... peak-concentration=... time=... amount=...]
//!                  [--injection species-override species=... type=... ...]
//!
//! chrom-rs save --target <model-single|model-multi|solver|scenario>
//!               --file <file.yml> [--project-dir <dir>]
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
//! [`ChromContext`](crate::cli::context::ChromContext)'s pending slots.
//! `--model single`/`multi`/`species`, `--solver`, and
//! `--initial-condition`/`--injection` are all wired up
//! ([#68](https://github.com/biface/chromatography/issues/68),
//! [#69](https://github.com/biface/chromatography/issues/69),
//! [#70](https://github.com/biface/chromatography/issues/70),
//! [#71](https://github.com/biface/chromatography/issues/71)). `save`
//! ([#72](https://github.com/biface/chromatography/issues/72)) serialises
//! any of those pending slots to a real file, validating required fields
//! and erroring by name rather than writing a YAML `null`. `config` and
//! `save` are typically chained in one invocation (DD-026): the pending
//! state built by one or more `config` calls only needs to survive until
//! the matching `save` call later in the same chain.

/// Execution context shared by every command handler — see [`ChromContext`](context::ChromContext).
pub mod context;

/// Helpers shared by more than one command handler — see the module docs.
pub(crate) mod support;

/// The `run` command handler ([`RunHandler`](crate::cli::run::RunHandler)) —
/// orchestrates the full simulation pipeline.
pub mod run;

/// The `check` command handler ([`CheckHandler`](crate::cli::check::CheckHandler)) —
/// validates configuration files without running a simulation.
pub mod check;

/// Pending-configuration builder types for the interactive `config`/`build`
/// command — accumulated, unvalidated state held in
/// [`ChromContext`](crate::cli::context::ChromContext) across chained
/// invocations. See DD-016 (issue #53) and issue #67.
pub mod builders;

/// The `config`/`build` command handler
/// ([`ConfigHandler`](crate::cli::config::ConfigHandler)) — builds
/// model/solver/scenario configuration interactively. See issue #68 (and
/// #69–72 for the parts not wired up yet).
pub mod config;

/// The `save` command handler
/// ([`SaveHandler`](crate::cli::save::SaveHandler)) — serialises a
/// pending `config`/`build` builder slot to a real `model.yml`/
/// `solver.yml`/`scenario.yml` file. See issue #72.
pub mod save;

use anyhow::anyhow;
use dynamic_cli::config::loader::load_yaml;
use dynamic_cli::{CliApp, CliBuilder};

use check::CheckHandler;
use config::ConfigHandler;
use context::ChromContext;
use run::RunHandler;
use save::SaveHandler;

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

/// Handler name that must match the `implementation:` field of the `save`
/// command in `commands.yml`.
const SAVE_HANDLER_NAME: &str = "save_handler";

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
        .register_sync_handler(SAVE_HANDLER_NAME, Box::new(SaveHandler))
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
        assert!(config.commands.iter().any(|c| c.name == "save"));
    }
}
