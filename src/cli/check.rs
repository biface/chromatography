//! `check` command — validates configuration files without running a
//! simulation.
//!
//! Two independent things, both requested explicitly, neither guessed at:
//!
//! 1. **Directory inventory.** Lists every `.yml`/`.yaml`/`.json` file
//!    directly under `--project-dir`, with a warning if fewer than three are
//!    present. No attempt is made to classify a file as "the" model,
//!    scenario, or solver — a directory can legitimately hold several
//!    variants of each (see `examples/config/tfa/`), and guessing which one
//!    the user means would be worse than not guessing at all.
//! 2. **Targeted validation**, for whichever `--source <role> file=...`
//!    occurrences are actually given (zero, one, two, or three — unlike
//!    `run`, no role is required here). Each is loaded with the same
//!    `load_model`/`load_scenario`/`load_solver` functions `run` itself
//!    uses, so "check passes" and "run would accept this file" mean exactly
//!    the same thing — there is no separate, weaker validation path to fall
//!    out of sync with the real one. `scenario` can only be validated
//!    alongside `model` (`load_scenario` needs a live model to check
//!    per-species injections against — see its doc comment), so a
//!    `scenario` given without a `model` is reported as skipped, not
//!    silently ignored and not treated as a failure on its own.
//!
//! `check` has no legacy scalar equivalents (`--model`, `--output-csv`,
//! …) — it's a new command, introduced alongside the new `--source`
//! syntax (both build on `dynamic-cli`'s own repeatable-sub-parameter
//! feature, [dcli#21](https://github.com/biface/dcli/issues/21)), so
//! there's no prior syntax to stay compatible with.

use std::fs;
use std::path::PathBuf;

use anyhow::anyhow;
use dynamic_cli::error::ExecutionError;
use dynamic_cli::{CommandHandler, DynamicCliError, ExecutionContext, ParsedArgs};

use crate::config::{model::load_model, scenario::load_scenario, solver::load_solver};

use super::app::{ChromContext, path_to_str, resolve_source_optional, to_cli_err};

/// `check` command handler.
///
/// See the module-level doc comment for what it does and, as importantly,
/// what it deliberately does not do (no cross-combination testing, no file
/// classification by naming convention or trial parsing).
pub struct CheckHandler;

impl CommandHandler for CheckHandler {
    fn execute(
        &self,
        ctx: &mut dyn ExecutionContext,
        args: &ParsedArgs,
    ) -> dynamic_cli::Result<()> {
        // ── 1. Project directory ─────────────────────────────────────────────
        let chrom_ctx = ctx
            .as_any_mut()
            .downcast_mut::<ChromContext>()
            .ok_or_else(|| {
                DynamicCliError::from(ExecutionError::ContextDowncastFailed {
                    expected_type: "ChromContext".to_string(),
                    suggestion: None,
                })
            })?;

        let project_dir_str = args.get_scalar("project-dir").unwrap_or(".");
        chrom_ctx
            .set_project_dir(project_dir_str)
            .map_err(|e| to_cli_err(anyhow!("{e}")))?;
        let project_dir: PathBuf = chrom_ctx.project_dir().to_path_buf();

        // ── 2. Directory inventory ───────────────────────────────────────────
        println!("Project directory: {}", project_dir.display());

        let mut config_like: Vec<PathBuf> = fs::read_dir(&project_dir)
            .map_err(|e| to_cli_err(anyhow!("reading project directory: {e}")))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| matches!(ext, "yml" | "yaml" | "json"))
            })
            .collect();
        config_like.sort();

        println!("Config-like files found: {}", config_like.len());
        for f in &config_like {
            let name = f
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            println!("  {name}");
        }
        if config_like.len() < 3 {
            println!(
                "Note: fewer than 3 config-like files present — a typical project needs \
                 at least a model, a scenario, and a solver config."
            );
        }
        println!();

        // ── 3. Targeted validation ───────────────────────────────────────────
        let model_file = resolve_source_optional(args, "model").map_err(to_cli_err)?;
        let scenario_file = resolve_source_optional(args, "scenario").map_err(to_cli_err)?;
        let solver_file = resolve_source_optional(args, "solver").map_err(to_cli_err)?;

        let mut checked_anything = false;
        let mut any_failed = false;

        let loaded_model = match &model_file {
            Some(file) => {
                checked_anything = true;
                let path = project_dir.join(file);
                let path_str = path_to_str(&path).map_err(to_cli_err)?;
                match load_model(path_str) {
                    Ok(model) => {
                        println!("model    OK    {file}");
                        Some(model)
                    }
                    Err(e) => {
                        any_failed = true;
                        println!("model    FAIL  {file} — {e}");
                        None
                    }
                }
            }
            None => None,
        };

        if let Some(file) = &scenario_file {
            checked_anything = true;
            match loaded_model {
                Some(mut model) => {
                    let path = project_dir.join(file);
                    let path_str = path_to_str(&path).map_err(to_cli_err)?;
                    match load_scenario(path_str, &mut *model) {
                        Ok(_) => println!("scenario OK    {file}"),
                        Err(e) => {
                            any_failed = true;
                            println!("scenario FAIL  {file} — {e}");
                        }
                    }
                }
                None if model_file.is_some() => {
                    // A model was given but failed to load — already
                    // reported above; scenario can't be checked against it.
                    println!("scenario SKIP  {file} — model failed to load, see above");
                }
                None => {
                    println!(
                        "scenario SKIP  {file} — cannot validate without \
                         '--source model file=...' (injections are checked against the model)"
                    );
                }
            }
        }

        if let Some(file) = &solver_file {
            checked_anything = true;
            let path = project_dir.join(file);
            let path_str = path_to_str(&path).map_err(to_cli_err)?;
            match load_solver(path_str) {
                Ok(_) => println!("solver   OK    {file}"),
                Err(e) => {
                    any_failed = true;
                    println!("solver   FAIL  {file} — {e}");
                }
            }
        }

        if !checked_anything {
            println!("No '--source' given — directory inventory only, nothing validated.");
        }

        if any_failed {
            return Err(to_cli_err(anyhow!(
                "one or more configuration files failed validation"
            )));
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use dynamic_cli::parser::cli_parser::{OptionOccurrence, ParsedValue};
    use std::collections::HashMap;
    use std::io::Write;

    /// Builds a `ParsedArgs` with a "project-dir" scalar plus a "source" key
    /// carrying the given (discriminant, file) occurrences.
    fn args_with_source(project_dir: &std::path::Path, occurrences: &[(&str, &str)]) -> ParsedArgs {
        let occs: Vec<OptionOccurrence> = occurrences
            .iter()
            .map(|(discriminant, file)| OptionOccurrence {
                discriminant: discriminant.to_string(),
                params: HashMap::from([("file".to_string(), file.to_string())]),
            })
            .collect();
        let mut map = HashMap::new();
        map.insert(
            "project-dir".to_string(),
            ParsedValue::Scalar(project_dir.to_string_lossy().into_owned()),
        );
        map.insert("source".to_string(), ParsedValue::Repeated(occs));
        ParsedArgs::new(map)
    }

    fn temp_project_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    fn write_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("failed to create file");
        f.write_all(content.as_bytes())
            .expect("failed to write file");
        path
    }

    #[test]
    fn test_check_no_source_lists_directory_only() {
        let dir = temp_project_dir();
        write_file(dir.path(), "model.yml", "LangmuirSingle:\n  lambda: 1.0\n");
        write_file(dir.path(), "notes.txt", "not a config file");

        let mut ctx = ChromContext::new();
        let args = ParsedArgs::from_scalars(HashMap::from([(
            "project-dir".to_string(),
            dir.path().to_string_lossy().into_owned(),
        )]));

        let result = CheckHandler.execute(&mut ctx, &args);
        assert!(result.is_ok(), "check with no --source should never fail");
    }

    #[test]
    fn test_check_scenario_without_model_is_skipped_not_failed() {
        let dir = temp_project_dir();
        write_file(
            dir.path(),
            "scenario.yml",
            "initial_condition: zero\ndefault_injection:\n  type: None\n",
        );

        let mut ctx = ChromContext::new();
        let args = args_with_source(dir.path(), &[("scenario", "scenario.yml")]);

        let result = CheckHandler.execute(&mut ctx, &args);
        assert!(
            result.is_ok(),
            "scenario-without-model should be a skip, not a failure"
        );
    }

    #[test]
    fn test_check_missing_source_file_fails() {
        let dir = temp_project_dir();

        let mut ctx = ChromContext::new();
        let args = args_with_source(dir.path(), &[("model", "does-not-exist.yml")]);

        let result = CheckHandler.execute(&mut ctx, &args);
        assert!(
            result.is_err(),
            "a --source pointing at a nonexistent file must fail check"
        );
    }
}
