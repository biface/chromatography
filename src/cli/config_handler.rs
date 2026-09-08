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
//! `--model single`/`multi`/`species` (#68, #69) and `--solver` (#70) are
//! wired up. `--scenario`
//! ([#71](https://github.com/biface/chromatography/issues/71)) lands in a
//! later, separate commit — this handler grows in place, `commands.yml`
//! gains a `scenario` option alongside `model`/`solver`.

use std::collections::HashMap;

use anyhow::anyhow;
use dynamic_cli::error::ExecutionError;
use dynamic_cli::{CommandHandler, DynamicCliError, ExecutionContext, ParsedArgs};

use super::app::{ChromContext, to_cli_err};
use super::builders::{MultiModelBuilder, SingleModelBuilder, SolverBuilder, SpeciesBuilder};

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
                    "multi" => {
                        let fields = parse_multi_fields(&occurrence.params).map_err(to_cli_err)?;
                        if let Some(warning) = chrom_ctx.merge_model_multi(fields) {
                            eprintln!("{warning}");
                        }
                    }
                    "species" => {
                        let species =
                            parse_species_fields(&occurrence.params).map_err(to_cli_err)?;
                        if let Some(warning) = chrom_ctx.add_species(species) {
                            eprintln!("{warning}");
                        }
                    }
                    other => {
                        // Defensive only: commands.yml's `choices` list means
                        // dcli itself rejects anything else before this
                        // handler ever runs.
                        return Err(to_cli_err(anyhow!(
                            "unsupported --model discriminant '{other}'"
                        )));
                    }
                }
            }
        }

        if let Some(occurrences) = args.get_repeated("solver") {
            for occurrence in occurrences {
                // The discriminant itself ("RK4" or "Euler") *is* the
                // solver type here — commands.yml's `choices: [RK4, Euler]`
                // is the only source of truth for valid values, so nothing
                // is re-validated against it in this handler.
                let fields = parse_solver_fields(&occurrence.discriminant, &occurrence.params)
                    .map_err(to_cli_err)?;
                chrom_ctx.merge_solver(fields);
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

/// Reads the eight `LangmuirMulti`-shaped scalar sub-parameters of a
/// `--model multi ...` occurrence into a [`MultiModelBuilder`]. `species` is
/// always empty here — species are added exclusively through
/// [`parse_species_fields`] / [`ChromContext::add_species`], never through
/// this occurrence, so ordering across chained occurrences stays
/// predictable (see [`MultiModelBuilder::merge_scalars`]).
fn parse_multi_fields(params: &HashMap<String, String>) -> anyhow::Result<MultiModelBuilder> {
    Ok(MultiModelBuilder {
        n_points: parse_optional_usize(params, "n-points")?,
        porosity: parse_optional_f64(params, "porosity")?,
        velocity: parse_optional_f64(params, "velocity")?,
        column_length: parse_optional_f64(params, "column-length")?,
        dz: parse_optional_f64(params, "dz")?,
        fe: parse_optional_f64(params, "fe")?,
        ue: parse_optional_f64(params, "ue")?,
        stationary_fraction: parse_optional_f64(params, "stationary-fraction")?,
        species: Vec::new(),
    })
}

/// Reads one `--model species name=... ...` occurrence into a
/// [`SpeciesBuilder`]. `name` is required by `commands.yml` (`required:
/// true` in `option_parameters.species`), so `dynamic-cli` itself rejects
/// an occurrence missing it before this function ever runs.
fn parse_species_fields(params: &HashMap<String, String>) -> anyhow::Result<SpeciesBuilder> {
    Ok(SpeciesBuilder {
        name: params.get("name").cloned(),
        lambda: parse_optional_f64(params, "lambda")?,
        langmuir_k: parse_optional_f64(params, "langmuir-k")?,
        port_number: parse_optional_u32(params, "port-number")?,
    })
}

/// Reads one `--solver <RK4|Euler> total-time=... time-steps=... [step=...]`
/// occurrence into a [`SolverBuilder`]. `discriminant` becomes
/// `solver_type` directly — it already carries the solver name, so there is
/// no separate `type=...` sub-parameter to also read. `step` is genuinely
/// optional here: an occurrence that omits it produces `step: None`,
/// matching `solver.yml`'s own "absent means full trajectory" convention
/// rather than introducing a new one.
fn parse_solver_fields(
    discriminant: &str,
    params: &HashMap<String, String>,
) -> anyhow::Result<SolverBuilder> {
    Ok(SolverBuilder {
        solver_type: Some(discriminant.to_string()),
        total_time: parse_optional_f64(params, "total-time")?,
        time_steps: parse_optional_usize(params, "time-steps")?,
        step: parse_optional_usize(params, "step")?,
    })
}

fn parse_optional_f64(params: &HashMap<String, String>, key: &str) -> anyhow::Result<Option<f64>> {
    params
        .get(key)
        .map(|raw| {
            raw.parse::<f64>()
                .map_err(|e| anyhow!("invalid float for '--model ... {key}=...': '{raw}' ({e})"))
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
            raw.parse::<usize>()
                .map_err(|e| anyhow!("invalid integer for '--model ... {key}=...': '{raw}' ({e})"))
        })
        .transpose()
}

fn parse_optional_u32(params: &HashMap<String, String>, key: &str) -> anyhow::Result<Option<u32>> {
    params
        .get(key)
        .map(|raw| {
            raw.parse::<u32>().map_err(|e| {
                anyhow!("invalid integer for '--model species {key}=...': '{raw}' ({e})")
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
        occurrence("single", pairs)
    }

    fn occurrence(discriminant: &str, pairs: &[(&str, &str)]) -> OptionOccurrence {
        OptionOccurrence {
            discriminant: discriminant.to_string(),
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

    // ── #69: --model multi / --model species ────────────────────────────

    #[test]
    fn test_multi_scalar_fields_are_set() {
        let mut ctx = ChromContext::new();
        let args = model_args(vec![occurrence(
            "multi",
            &[
                ("n-points", "100"),
                ("porosity", "0.4"),
                ("velocity", "0.001"),
                ("column-length", "0.25"),
                ("dz", "0.0025"),
                ("fe", "1.5"),
                ("ue", "0.0025"),
                ("stationary-fraction", "0.6"),
            ],
        )]);

        ConfigHandler
            .execute(&mut ctx, &args)
            .expect("must succeed");

        match ctx.pending_model() {
            Some(ModelBuilder::Multi(multi)) => {
                assert_eq!(multi.n_points, Some(100));
                assert_eq!(multi.porosity, Some(0.4));
                assert_eq!(multi.stationary_fraction, Some(0.6));
                assert!(multi.species.is_empty());
            }
            other => panic!("expected Multi, got {other:?}"),
        }
    }

    #[test]
    fn test_two_chained_species_occurrences_produce_a_two_element_list_in_order() {
        let mut ctx = ChromContext::new();

        ConfigHandler
            .execute(
                &mut ctx,
                &model_args(vec![occurrence(
                    "species",
                    &[
                        ("name", "Ascorbic"),
                        ("lambda", "1.0"),
                        ("langmuir-k", "1.1"),
                        ("port-number", "2"),
                    ],
                )]),
            )
            .expect("must succeed");

        ConfigHandler
            .execute(
                &mut ctx,
                &model_args(vec![occurrence(
                    "species",
                    &[("name", "Erythorbic"), ("lambda", "0.9")],
                )]),
            )
            .expect("must succeed");

        match ctx.pending_model() {
            Some(ModelBuilder::Multi(multi)) => {
                assert_eq!(multi.species.len(), 2);
                assert_eq!(multi.species[0].name.as_deref(), Some("Ascorbic"));
                assert_eq!(multi.species[0].port_number, Some(2));
                assert_eq!(multi.species[1].name.as_deref(), Some("Erythorbic"));
            }
            other => panic!("expected Multi, got {other:?}"),
        }
    }

    #[test]
    fn test_species_occurrence_alone_locks_multi_shape_without_prior_multi_call() {
        let mut ctx = ChromContext::new();
        let args = model_args(vec![occurrence("species", &[("name", "A")])]);

        ConfigHandler
            .execute(&mut ctx, &args)
            .expect("must succeed");

        assert!(matches!(ctx.pending_model(), Some(ModelBuilder::Multi(_))));
    }

    #[test]
    fn test_duplicate_species_name_accepted_at_this_construction_stage() {
        let mut ctx = ChromContext::new();
        let args = model_args(vec![
            occurrence("species", &[("name", "A")]),
            occurrence("species", &[("name", "A")]),
        ]);

        ConfigHandler
            .execute(&mut ctx, &args)
            .expect("must succeed");

        match ctx.pending_model() {
            Some(ModelBuilder::Multi(multi)) => assert_eq!(multi.species.len(), 2),
            other => panic!("expected Multi, got {other:?}"),
        }
    }

    #[test]
    fn test_species_missing_required_name_is_rejected_by_dcli_before_the_handler() {
        // dcli validates `option_parameters.species`'s `required: true` on
        // `name` before ParsedArgs ever reaches a handler — this test only
        // documents that our own parser fills a missing name with `None`
        // rather than panicking, in case that guarantee ever changes.
        let params: HashMap<String, String> = HashMap::new();
        let species = parse_species_fields(&params).expect("parsing itself does not fail");
        assert_eq!(species.name, None);
    }

    // ── #70: --solver ────────────────────────────────────────────────────

    fn solver_args(occurrences: Vec<OptionOccurrence>) -> ParsedArgs {
        let mut map = HashMap::new();
        map.insert("solver".to_string(), ParsedValue::Repeated(occurrences));
        ParsedArgs::new(map)
    }

    #[test]
    fn test_solver_discriminant_becomes_solver_type() {
        let mut ctx = ChromContext::new();
        let args = solver_args(vec![occurrence(
            "RK4",
            &[("total-time", "600"), ("time-steps", "10000")],
        )]);

        ConfigHandler
            .execute(&mut ctx, &args)
            .expect("must succeed");

        let solver = ctx.pending_solver().expect("must be set");
        assert_eq!(solver.solver_type.as_deref(), Some("RK4"));
        assert_eq!(solver.total_time, Some(600.0));
        assert_eq!(solver.time_steps, Some(10_000));
        assert_eq!(solver.step, None);
    }

    #[test]
    fn test_omitting_step_leaves_it_unset() {
        let mut ctx = ChromContext::new();
        let args = solver_args(vec![occurrence(
            "Euler",
            &[("total-time", "600"), ("time-steps", "5000")],
        )]);

        ConfigHandler
            .execute(&mut ctx, &args)
            .expect("must succeed");

        let solver = ctx.pending_solver().expect("must be set");
        assert_eq!(
            solver.step, None,
            "omitted 'step' must stay unset, not default to 0 or 1"
        );
    }

    #[test]
    fn test_solver_step_is_set_when_given() {
        let mut ctx = ChromContext::new();
        let args = solver_args(vec![occurrence(
            "RK4",
            &[
                ("total-time", "600"),
                ("time-steps", "10000"),
                ("step", "100"),
            ],
        )]);

        ConfigHandler
            .execute(&mut ctx, &args)
            .expect("must succeed");

        assert_eq!(ctx.pending_solver().unwrap().step, Some(100));
    }

    #[test]
    fn test_solver_fields_merge_across_chained_calls() {
        let mut ctx = ChromContext::new();
        ConfigHandler
            .execute(
                &mut ctx,
                &solver_args(vec![occurrence("RK4", &[("total-time", "600")])]),
            )
            .expect("must succeed");
        ConfigHandler
            .execute(
                &mut ctx,
                &solver_args(vec![occurrence("RK4", &[("time-steps", "10000")])]),
            )
            .expect("must succeed");

        let solver = ctx.pending_solver().expect("must be set");
        assert_eq!(solver.total_time, Some(600.0));
        assert_eq!(solver.time_steps, Some(10_000));
    }
}
