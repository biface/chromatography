//! `config` command (alias `build`) — build model/solver/scenario
//! configuration interactively, field by field, without hand-writing YAML.
//!
//! Every `--model`/`--solver`/`--scenario` occurrence is repeatable
//! (DD-016, [#53](https://github.com/biface/chromatography/issues/53)) and
//! merges into [`ChromContext`]'s pending-configuration slots (see
//! [`crate::cli::builders`]) — nothing here is validated against the real
//! model/solver/scenario types; that happens only at `save`/`run`.
//!
//! # Scope of this file today
//!
//! Only `--model single ...` is wired up
//! ([#68](https://github.com/biface/chromatography/issues/68)). `--model
//! multi`/`--model species` ([#69](https://github.com/biface/chromatography/issues/69)),
//! `--solver` ([#70](https://github.com/biface/chromatography/issues/70)),
//! and `--scenario`
//! ([#71](https://github.com/biface/chromatography/issues/71)) land in
//! later, separate commits — this handler grows in place, `commands.yml`'s
//! `model` option gains more `choices` alongside it.

use std::collections::HashMap;

use anyhow::anyhow;
use dynamic_cli::error::ExecutionError;
use dynamic_cli::{CommandHandler, DynamicCliError, ExecutionContext, ParsedArgs};

use super::app::{ChromContext, to_cli_err};
use super::builders::SingleModelBuilder;

/// `config`/`build` command handler.
///
/// See the module-level doc comment for what is and is not wired up yet.
pub struct ConfigHandler;

impl CommandHandler for ConfigHandler {
    fn execute(
        &self,
        ctx: &mut dyn ExecutionContext,
        args: &ParsedArgs,
    ) -> dynamic_cli::Result<()> {
        let chrom_ctx = ctx
            .as_any_mut()
            .downcast_mut::<ChromContext>()
            .ok_or_else(|| {
                DynamicCliError::from(ExecutionError::ContextDowncastFailed {
                    expected_type: "ChromContext".to_string(),
                    suggestion: None,
                })
            })?;

        if let Some(occurrences) = args.get_repeated("model") {
            for occurrence in occurrences {
                match occurrence.discriminant.as_str() {
                    "single" => {
                        let fields = parse_single_fields(&occurrence.params).map_err(to_cli_err)?;
                        if let Some(warning) = chrom_ctx.merge_model_single(fields) {
                            eprintln!("{warning}");
                        }
                    }
                    other => {
                        // Defensive only: commands.yml's `choices: [single]`
                        // means dcli itself rejects anything else before
                        // this handler ever runs.
                        return Err(to_cli_err(anyhow!(
                            "unsupported --model discriminant '{other}'"
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Reads the eight `LangmuirSingle`-shaped sub-parameters of a `--model
/// single ...` occurrence into a [`SingleModelBuilder`].
///
/// Each value has already been type-validated by `dynamic-cli` against
/// `commands.yml`'s `option_parameters` declaration (stored as a string
/// regardless, like every other option/argument value) — the `.parse()`
/// calls here are a defensive second check, not the primary validation.
fn parse_single_fields(params: &HashMap<String, String>) -> anyhow::Result<SingleModelBuilder> {
    Ok(SingleModelBuilder {
        lambda: parse_optional_f64(params, "lambda")?,
        langmuir_k: parse_optional_f64(params, "langmuir-k")?,
        port_number: parse_optional_f64(params, "port-number")?,
        column_length: parse_optional_f64(params, "column-length")?,
        n_points: parse_optional_usize(params, "n-points")?,
        dz: parse_optional_f64(params, "dz")?,
        fe: parse_optional_f64(params, "fe")?,
        ue: parse_optional_f64(params, "ue")?,
    })
}

fn parse_optional_f64(params: &HashMap<String, String>, key: &str) -> anyhow::Result<Option<f64>> {
    params
        .get(key)
        .map(|raw| {
            raw.parse::<f64>()
                .map_err(|e| anyhow!("invalid float for '--model single {key}=...': '{raw}' ({e})"))
        })
        .transpose()
}

fn parse_optional_usize(
    params: &HashMap<String, String>,
    key: &str,
) -> anyhow::Result<Option<usize>> {
    params
        .get(key)
        .map(|raw| {
            raw.parse::<usize>().map_err(|e| {
                anyhow!("invalid integer for '--model single {key}=...': '{raw}' ({e})")
            })
        })
        .transpose()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::builders::ModelBuilder;
    use dynamic_cli::parser::cli_parser::{OptionOccurrence, ParsedValue};

    /// Builds a `ParsedArgs` with a single "model" key carrying the given
    /// occurrences — mirrors the `source_args` test helper in `app.rs`.
    fn model_args(occurrences: Vec<OptionOccurrence>) -> ParsedArgs {
        let mut map = HashMap::new();
        map.insert("model".to_string(), ParsedValue::Repeated(occurrences));
        ParsedArgs::new(map)
    }

    fn single_occurrence(pairs: &[(&str, &str)]) -> OptionOccurrence {
        OptionOccurrence {
            discriminant: "single".to_string(),
            params: pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[test]
    fn test_one_call_sets_all_eight_fields() {
        let mut ctx = ChromContext::new();
        let args = model_args(vec![single_occurrence(&[
            ("lambda", "1.2"),
            ("langmuir-k", "0.4"),
            ("port-number", "2.0"),
            ("column-length", "0.25"),
            ("n-points", "100"),
            ("dz", "0.0025"),
            ("fe", "1.5"),
            ("ue", "0.0025"),
        ])]);

        ConfigHandler
            .execute(&mut ctx, &args)
            .expect("must succeed");

        match ctx.pending_model() {
            Some(ModelBuilder::Single(single)) => {
                assert_eq!(single.lambda, Some(1.2));
                assert_eq!(single.langmuir_k, Some(0.4));
                assert_eq!(single.port_number, Some(2.0));
                assert_eq!(single.column_length, Some(0.25));
                assert_eq!(single.n_points, Some(100));
                assert_eq!(single.dz, Some(0.0025));
                assert_eq!(single.fe, Some(1.5));
                assert_eq!(single.ue, Some(0.0025));
            }
            other => panic!("expected Single, got {other:?}"),
        }
    }

    #[test]
    fn test_eight_chained_calls_produce_the_same_pending_state_as_one() {
        let mut ctx_one_call = ChromContext::new();
        ConfigHandler
            .execute(
                &mut ctx_one_call,
                &model_args(vec![single_occurrence(&[
                    ("lambda", "1.2"),
                    ("langmuir-k", "0.4"),
                    ("port-number", "2.0"),
                    ("column-length", "0.25"),
                    ("n-points", "100"),
                    ("dz", "0.0025"),
                    ("fe", "1.5"),
                    ("ue", "0.0025"),
                ])]),
            )
            .expect("must succeed");

        let mut ctx_chained = ChromContext::new();
        let fields: [(&str, &str); 8] = [
            ("lambda", "1.2"),
            ("langmuir-k", "0.4"),
            ("port-number", "2.0"),
            ("column-length", "0.25"),
            ("n-points", "100"),
            ("dz", "0.0025"),
            ("fe", "1.5"),
            ("ue", "0.0025"),
        ];
        for field in fields {
            ConfigHandler
                .execute(
                    &mut ctx_chained,
                    &model_args(vec![single_occurrence(&[field])]),
                )
                .expect("must succeed");
        }

        match (ctx_one_call.pending_model(), ctx_chained.pending_model()) {
            (Some(ModelBuilder::Single(a)), Some(ModelBuilder::Single(b))) => {
                assert_eq!(a, b);
            }
            other => panic!("expected two Single builders, got {other:?}"),
        }
    }

    #[test]
    fn test_no_model_occurrence_leaves_pending_model_untouched() {
        let mut ctx = ChromContext::new();
        let args = model_args(vec![]);

        ConfigHandler
            .execute(&mut ctx, &args)
            .expect("must succeed");

        assert!(ctx.pending_model().is_none());
    }

    #[test]
    fn test_invalid_float_is_reported_with_the_offending_field() {
        let mut ctx = ChromContext::new();
        let args = model_args(vec![single_occurrence(&[("lambda", "not-a-number")])]);

        let err = ConfigHandler
            .execute(&mut ctx, &args)
            .expect_err("must fail");
        assert!(err.to_string().contains("lambda"));
    }
}
