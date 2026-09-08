//! `save` command — serialise a pending `config` builder slot to a real
//! `model.yml` / `solver.yml` / `scenario.yml` file, in exactly the shape
//! [`crate::config::model::load_model`],
//! [`crate::config::solver::load_solver`], and
//! [`crate::config::scenario::load_scenario`] already expect to read back
//! (DD-016, [#53](https://github.com/biface/chromatography/issues/53),
//! [#72](https://github.com/biface/chromatography/issues/72)).
//!
//! `LangmuirSingle`/`LangmuirMulti`/`SpeciesParams`'s own fields are
//! private, and their public constructors take raw physical inputs
//! (`porosity`, `velocity`) rather than the already-precomputed
//! `dz`/`fe`/`ue` a pending model carries — so this module builds the YAML
//! directly as a [`serde_yaml::Value`] tree instead of instantiating the
//! real structs, matching field names and the `LangmuirSingle:`/
//! `LangmuirMulti:` typetag root key by hand.
//!
//! Every target validates its own required fields before writing anything:
//! a missing field is an explicit, named error, never a YAML file with a
//! `null` in place of a value `load_model`/`load_solver` would then reject
//! with a much less helpful message.

use std::path::PathBuf;

use anyhow::anyhow;
use dynamic_cli::error::ExecutionError;
use dynamic_cli::{CommandHandler, DynamicCliError, ExecutionContext, ParsedArgs};
use serde_yaml::{Mapping, Value};

use super::app::{ChromContext, path_to_str, to_cli_err};
use super::builders::{
    InjectionBuilder, ModelBuilder, MultiModelBuilder, ScenarioBuilder, SingleModelBuilder,
    SolverBuilder,
};

/// `save` command handler.
pub struct SaveHandler;

impl CommandHandler for SaveHandler {
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

        let project_dir_str = args.get_scalar("project-dir").unwrap_or(".");
        chrom_ctx
            .set_project_dir(project_dir_str)
            .map_err(|e| to_cli_err(anyhow!("{e}")))?;
        let project_dir: PathBuf = chrom_ctx.project_dir().to_path_buf();

        let target = args
            .get_scalar("target")
            .ok_or_else(|| to_cli_err(anyhow!("--target is required")))?;
        let file = args
            .get_scalar("file")
            .ok_or_else(|| to_cli_err(anyhow!("--file is required")))?;

        let content = match target {
            "model-single" => match chrom_ctx.pending_model() {
                Some(ModelBuilder::Single(single)) => {
                    build_model_single(single).map_err(to_cli_err)?
                }
                Some(ModelBuilder::Multi(_)) => {
                    return Err(to_cli_err(anyhow!(
                        "cannot save 'model-single': the pending model is currently \
                         multi-species — use --target model-multi, or start over with \
                         config --model single ..."
                    )));
                }
                None => {
                    return Err(to_cli_err(anyhow!(
                        "cannot save 'model-single': no pending model in this session \
                         (config --model single ... first)"
                    )));
                }
            },
            "model-multi" => match chrom_ctx.pending_model() {
                Some(ModelBuilder::Multi(multi)) => build_model_multi(multi).map_err(to_cli_err)?,
                Some(ModelBuilder::Single(_)) => {
                    return Err(to_cli_err(anyhow!(
                        "cannot save 'model-multi': the pending model is currently \
                         single-species — use --target model-single, or start over with \
                         config --model multi/species ..."
                    )));
                }
                None => {
                    return Err(to_cli_err(anyhow!(
                        "cannot save 'model-multi': no pending model in this session \
                         (config --model multi/species ... first)"
                    )));
                }
            },
            "solver" => {
                let solver = chrom_ctx.pending_solver().ok_or_else(|| {
                    to_cli_err(anyhow!(
                        "cannot save 'solver': no pending solver in this session \
                         (config --solver ... first)"
                    ))
                })?;
                build_solver(solver).map_err(to_cli_err)?
            }
            "scenario" => {
                // Unlike model-single/model-multi/solver, an entirely empty
                // scenario is a legitimate save: every scenario.yml field is
                // optional to load_scenario (initial_condition defaults to
                // "zero", default_injection/injections default to none) —
                // there is no required-field error path for this target.
                let scenario = chrom_ctx.pending_scenario().cloned().unwrap_or_default();
                build_scenario(&scenario).map_err(to_cli_err)?
            }
            other => {
                // Defensive only: commands.yml's `choices` list means dcli
                // itself rejects anything else before this handler runs.
                return Err(to_cli_err(anyhow!("unsupported --target '{other}'")));
            }
        };

        let file_path = project_dir.join(file);
        std::fs::write(&file_path, content).map_err(|e| {
            to_cli_err(anyhow!(
                "failed to write '{}': {e}",
                path_to_str(&file_path).unwrap_or(file)
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// YAML builders
// ============================================================================

/// Returns `value`, or a named "required field missing" error mentioning
/// both the field and the target — never lets a `None` slip through to
/// become a YAML `null`.
fn require<T>(value: Option<T>, field: &str, target: &str) -> anyhow::Result<T> {
    value.ok_or_else(|| anyhow!("cannot save '{target}': required field '{field}' is not set"))
}

/// `{ type: None }` — the placeholder every `model.yml` injection field
/// carries; real injections are supplied by `scenario.yml` and applied by
/// `scenario::load_scenario` after the model loads, never by `model.yml`
/// itself.
fn none_injection() -> Value {
    let mut m = Mapping::new();
    m.insert("type".into(), "None".into());
    Value::Mapping(m)
}

/// Builds `model.yml`'s `LangmuirSingle:`-rooted shape from a
/// [`SingleModelBuilder`]. All eight fields are required — see the module
/// doc comment for why this doesn't go through `LangmuirSingle` itself.
fn build_model_single(single: &SingleModelBuilder) -> anyhow::Result<String> {
    const TARGET: &str = "model-single";
    let lambda = require(single.lambda, "lambda", TARGET)?;
    let langmuir_k = require(single.langmuir_k, "langmuir-k", TARGET)?;
    let port_number = require(single.port_number, "port-number", TARGET)?;
    let column_length = require(single.column_length, "column-length", TARGET)?;
    let n_points = require(single.n_points, "n-points", TARGET)?;
    let dz = require(single.dz, "dz", TARGET)?;
    let fe = require(single.fe, "fe", TARGET)?;
    let ue = require(single.ue, "ue", TARGET)?;

    let mut fields = Mapping::new();
    fields.insert("lambda".into(), lambda.into());
    fields.insert("langmuir_k".into(), langmuir_k.into());
    fields.insert("port_number".into(), port_number.into());
    fields.insert("column_length".into(), column_length.into());
    fields.insert("n_points".into(), n_points.into());
    fields.insert("dz".into(), dz.into());
    fields.insert("fe".into(), fe.into());
    fields.insert("ue".into(), ue.into());
    fields.insert("injection".into(), none_injection());

    let mut root = Mapping::new();
    root.insert("LangmuirSingle".into(), Value::Mapping(fields));

    serde_yaml::to_string(&Value::Mapping(root))
        .map_err(|e| anyhow!("failed to serialise {TARGET}: {e}"))
}

/// Builds `model.yml`'s `LangmuirMulti:`-rooted shape from a
/// [`MultiModelBuilder`]. All eight scalar fields are required, at least
/// one species is required, and every species must itself have all four
/// fields set — a species missing `name` should be unreachable in practice
/// (`commands.yml` marks it `required: true`), but is still checked here
/// defensively rather than assumed.
fn build_model_multi(multi: &MultiModelBuilder) -> anyhow::Result<String> {
    const TARGET: &str = "model-multi";
    let n_points = require(multi.n_points, "n-points", TARGET)?;
    let porosity = require(multi.porosity, "porosity", TARGET)?;
    let velocity = require(multi.velocity, "velocity", TARGET)?;
    let column_length = require(multi.column_length, "column-length", TARGET)?;
    let dz = require(multi.dz, "dz", TARGET)?;
    let fe = require(multi.fe, "fe", TARGET)?;
    let ue = require(multi.ue, "ue", TARGET)?;
    let stationary_fraction = require(multi.stationary_fraction, "stationary-fraction", TARGET)?;

    if multi.species.is_empty() {
        return Err(anyhow!(
            "cannot save '{TARGET}': at least one species is required \
             (config --model species name=... ...)"
        ));
    }

    let mut species_seq = Vec::with_capacity(multi.species.len());
    for (index, species) in multi.species.iter().enumerate() {
        let unnamed_context = format!("{TARGET} (species #{index})");
        let name = require(species.name.clone(), "name", &unnamed_context)?;
        let named_context = format!("{TARGET} species '{name}'");
        let lambda = require(species.lambda, "lambda", &named_context)?;
        let langmuir_k = require(species.langmuir_k, "langmuir-k", &named_context)?;
        let port_number = require(species.port_number, "port-number", &named_context)?;

        let mut sp = Mapping::new();
        sp.insert("name".into(), name.into());
        sp.insert("lambda".into(), lambda.into());
        sp.insert("langmuir_k".into(), langmuir_k.into());
        sp.insert("port_number".into(), port_number.into());
        sp.insert("injection".into(), none_injection());
        species_seq.push(Value::Mapping(sp));
    }

    let mut fields = Mapping::new();
    fields.insert("species".into(), Value::Sequence(species_seq));
    fields.insert("n_points".into(), n_points.into());
    fields.insert("porosity".into(), porosity.into());
    fields.insert("velocity".into(), velocity.into());
    fields.insert("column_length".into(), column_length.into());
    fields.insert("dz".into(), dz.into());
    fields.insert("fe".into(), fe.into());
    fields.insert("ue".into(), ue.into());
    fields.insert("stationary_fraction".into(), stationary_fraction.into());

    let mut root = Mapping::new();
    root.insert("LangmuirMulti".into(), Value::Mapping(fields));

    serde_yaml::to_string(&Value::Mapping(root))
        .map_err(|e| anyhow!("failed to serialise {TARGET}: {e}"))
}

/// Builds `solver.yml`'s flat (no root wrapper key) shape from a
/// [`SolverBuilder`]. `type`/`total_time`/`time_steps` are required; `step`
/// is written only when set — an absent key means the same "full
/// trajectory" default as an explicit `null` to `load_solver`, so omitting
/// it is preferred over writing a literal `null`.
fn build_solver(solver: &SolverBuilder) -> anyhow::Result<String> {
    const TARGET: &str = "solver";
    let solver_type = require(solver.solver_type.clone(), "type", TARGET)?;
    let total_time = require(solver.total_time, "total-time", TARGET)?;
    let time_steps = require(solver.time_steps, "time-steps", TARGET)?;

    let mut fields = Mapping::new();
    fields.insert("type".into(), solver_type.into());
    fields.insert("total_time".into(), total_time.into());
    fields.insert("time_steps".into(), time_steps.into());
    if let Some(step) = solver.step {
        fields.insert("step".into(), step.into());
    }

    serde_yaml::to_string(&Value::Mapping(fields))
        .map_err(|e| anyhow!("failed to serialise {TARGET}: {e}"))
}

/// Builds `scenario.yml`'s flat shape from a [`ScenarioBuilder`]. Every
/// field is optional — see the `"scenario"` match arm in
/// [`SaveHandler::execute`] for why there is no required-field error path
/// for this target.
fn build_scenario(scenario: &ScenarioBuilder) -> anyhow::Result<String> {
    let mut fields = Mapping::new();

    if let Some(initial_condition) = &scenario.initial_condition {
        fields.insert("initial_condition".into(), initial_condition.clone().into());
    }

    if let Some(default) = &scenario.default_injection {
        fields.insert(
            "default_injection".into(),
            injection_to_value(default, "scenario default_injection")?,
        );
    }

    if !scenario.species_overrides.is_empty() {
        let mut overrides = Vec::with_capacity(scenario.species_overrides.len());
        for (species, injection) in &scenario.species_overrides {
            let context = format!("scenario injection override for '{species}'");
            let mut entry = match injection_to_value(injection, &context)? {
                Value::Mapping(m) => m,
                _ => unreachable!("injection_to_value always returns a Mapping"),
            };
            entry.insert("species".into(), species.clone().into());
            overrides.push(Value::Mapping(entry));
        }
        fields.insert("injections".into(), Value::Sequence(overrides));
    }

    serde_yaml::to_string(&Value::Mapping(fields))
        .map_err(|e| anyhow!("failed to serialise scenario: {e}"))
}

/// Builds one injection block (`default_injection` or one entry of
/// `injections`). `type` is required and selects which further fields are
/// required — `Gaussian` needs `center`/`width`/`peak-concentration`,
/// `Dirac` needs `time`/`amount`, `None` needs nothing else.
fn injection_to_value(injection: &InjectionBuilder, context: &str) -> anyhow::Result<Value> {
    let injection_type = require(injection.injection_type.clone(), "type", context)?;

    let mut m = Mapping::new();
    m.insert("type".into(), injection_type.clone().into());

    match injection_type.as_str() {
        "Gaussian" => {
            m.insert(
                "center".into(),
                require(injection.center, "center", context)?.into(),
            );
            m.insert(
                "width".into(),
                require(injection.width, "width", context)?.into(),
            );
            m.insert(
                "peak_concentration".into(),
                require(injection.peak_concentration, "peak-concentration", context)?.into(),
            );
        }
        "Dirac" => {
            m.insert(
                "time".into(),
                require(injection.time, "time", context)?.into(),
            );
            m.insert(
                "amount".into(),
                require(injection.amount, "amount", context)?.into(),
            );
        }
        "None" => {}
        other => {
            return Err(anyhow!(
                "{context}: unknown injection type '{other}' (expected Gaussian, Dirac, or None)"
            ));
        }
    }

    Ok(Value::Mapping(m))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::builders::SpeciesBuilder;
    use crate::cli::config_handler::ConfigHandler;
    use crate::config::model::load_model;
    use crate::config::solver::load_solver;
    use dynamic_cli::parser::cli_parser::{OptionOccurrence, ParsedValue};
    use std::collections::HashMap;

    fn temp_project_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    fn model_occurrence(discriminant: &str, pairs: &[(&str, &str)]) -> OptionOccurrence {
        OptionOccurrence {
            discriminant: discriminant.to_string(),
            params: pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn model_args(occurrences: Vec<OptionOccurrence>) -> ParsedArgs {
        let mut map = HashMap::new();
        map.insert("model".to_string(), ParsedValue::Repeated(occurrences));
        ParsedArgs::new(map)
    }

    fn save_args(project_dir: &std::path::Path, target: &str, file: &str) -> ParsedArgs {
        ParsedArgs::from_scalars(HashMap::from([
            (
                "project-dir".to_string(),
                project_dir.to_string_lossy().into_owned(),
            ),
            ("target".to_string(), target.to_string()),
            ("file".to_string(), file.to_string()),
        ]))
    }

    // ── model-single round-trip (acceptance criterion) ──────────────────

    #[test]
    fn test_model_single_round_trips_through_load_model() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();

        ConfigHandler
            .execute(
                &mut ctx,
                &model_args(vec![model_occurrence(
                    "single",
                    &[
                        ("lambda", "1.2"),
                        ("langmuir-k", "0.4"),
                        ("port-number", "2.0"),
                        ("column-length", "0.25"),
                        ("n-points", "100"),
                        ("dz", "0.0025"),
                        ("fe", "1.5"),
                        ("ue", "0.0025"),
                    ],
                )]),
            )
            .expect("config must succeed");

        SaveHandler
            .execute(
                &mut ctx,
                &save_args(dir.path(), "model-single", "model.yml"),
            )
            .expect("save must succeed");

        let model = load_model(dir.path().join("model.yml").to_str().unwrap())
            .expect("the saved file must load back successfully");
        assert_eq!(model.points(), 100);
        assert_eq!(
            model.name(),
            "Langmuir single specie with temporal injection"
        );
    }

    #[test]
    fn test_model_multi_round_trips_through_load_model() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();

        ConfigHandler
            .execute(
                &mut ctx,
                &model_args(vec![
                    model_occurrence(
                        "multi",
                        &[
                            ("n-points", "50"),
                            ("porosity", "0.4"),
                            ("velocity", "0.001"),
                            ("column-length", "0.25"),
                            ("dz", "0.005"),
                            ("fe", "1.5"),
                            ("ue", "0.0025"),
                            ("stationary-fraction", "0.6"),
                        ],
                    ),
                    model_occurrence(
                        "species",
                        &[
                            ("name", "A"),
                            ("lambda", "1.0"),
                            ("langmuir-k", "0.5"),
                            ("port-number", "1"),
                        ],
                    ),
                    model_occurrence(
                        "species",
                        &[
                            ("name", "B"),
                            ("lambda", "1.0"),
                            ("langmuir-k", "2.0"),
                            ("port-number", "1"),
                        ],
                    ),
                ]),
            )
            .expect("config must succeed");

        SaveHandler
            .execute(&mut ctx, &save_args(dir.path(), "model-multi", "model.yml"))
            .expect("save must succeed");

        let model = load_model(dir.path().join("model.yml").to_str().unwrap())
            .expect("the saved file must load back successfully");
        assert_eq!(model.points(), 50);
        assert_eq!(model.name(), "Langmuir Multi-Species");
    }

    #[test]
    fn test_solver_round_trips_through_load_solver() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();
        ctx.merge_solver(SolverBuilder {
            solver_type: Some("RK4".to_string()),
            total_time: Some(600.0),
            time_steps: Some(10_000),
            step: None,
        });

        SaveHandler
            .execute(&mut ctx, &save_args(dir.path(), "solver", "solver.yml"))
            .expect("save must succeed");

        let solver_config = load_solver(dir.path().join("solver.yml").to_str().unwrap())
            .expect("the saved file must load back successfully");
        assert_eq!(solver_config.solver_name, "RK4");
        assert!(solver_config.config.step.is_none());
    }

    // ── missing required field (acceptance criterion) ───────────────────

    #[test]
    fn test_saving_model_single_with_a_missing_field_names_it() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();
        ConfigHandler
            .execute(
                &mut ctx,
                &model_args(vec![model_occurrence("single", &[("lambda", "1.2")])]),
            )
            .expect("config must succeed");

        let err = SaveHandler
            .execute(
                &mut ctx,
                &save_args(dir.path(), "model-single", "model.yml"),
            )
            .expect_err("must fail — only 'lambda' was set");
        assert!(err.to_string().contains("langmuir-k"));
        assert!(
            !dir.path().join("model.yml").exists(),
            "no file should be written on failure"
        );
    }

    #[test]
    fn test_saving_model_multi_with_no_species_is_an_explicit_error() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();
        ConfigHandler
            .execute(
                &mut ctx,
                &model_args(vec![model_occurrence(
                    "multi",
                    &[
                        ("n-points", "50"),
                        ("porosity", "0.4"),
                        ("velocity", "0.001"),
                        ("column-length", "0.25"),
                        ("dz", "0.005"),
                        ("fe", "1.5"),
                        ("ue", "0.0025"),
                        ("stationary-fraction", "0.6"),
                    ],
                )]),
            )
            .expect("config must succeed");

        let err = SaveHandler
            .execute(&mut ctx, &save_args(dir.path(), "model-multi", "model.yml"))
            .expect_err("must fail — no species were added");
        assert!(err.to_string().contains("species"));
    }

    #[test]
    fn test_saving_model_multi_with_a_species_missing_a_field_names_it() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();
        ctx.merge_model_multi(MultiModelBuilder {
            n_points: Some(50),
            porosity: Some(0.4),
            velocity: Some(0.001),
            column_length: Some(0.25),
            dz: Some(0.005),
            fe: Some(1.5),
            ue: Some(0.0025),
            stationary_fraction: Some(0.6),
            species: Vec::new(),
        });
        ctx.add_species(SpeciesBuilder {
            name: Some("A".to_string()),
            lambda: Some(1.0),
            langmuir_k: None, // missing on purpose
            port_number: Some(1),
        });

        let err = SaveHandler
            .execute(&mut ctx, &save_args(dir.path(), "model-multi", "model.yml"))
            .expect_err("must fail — species 'A' is missing langmuir-k");
        assert!(err.to_string().contains("langmuir-k"));
        assert!(err.to_string().contains('A'));
    }

    #[test]
    fn test_saving_solver_with_a_missing_field_names_it() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();
        ctx.merge_solver(SolverBuilder {
            solver_type: Some("RK4".to_string()),
            total_time: None, // missing on purpose
            time_steps: Some(10_000),
            step: None,
        });

        let err = SaveHandler
            .execute(&mut ctx, &save_args(dir.path(), "solver", "solver.yml"))
            .expect_err("must fail — 'total-time' was never set");
        assert!(err.to_string().contains("total-time"));
    }

    // ── other explicit errors ────────────────────────────────────────────

    #[test]
    fn test_saving_without_any_pending_model_is_an_explicit_error() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();

        let err = SaveHandler
            .execute(
                &mut ctx,
                &save_args(dir.path(), "model-single", "model.yml"),
            )
            .expect_err("must fail — no pending model at all");
        assert!(err.to_string().contains("no pending model"));
    }

    #[test]
    fn test_saving_wrong_shape_target_is_an_explicit_error() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();
        ctx.merge_model_single(SingleModelBuilder {
            lambda: Some(1.2),
            ..Default::default()
        });

        let err = SaveHandler
            .execute(&mut ctx, &save_args(dir.path(), "model-multi", "model.yml"))
            .expect_err("must fail — the pending model is single, not multi");
        assert!(err.to_string().contains("single-species"));
    }

    // ── scenario: empty is legitimate ────────────────────────────────────

    #[test]
    fn test_saving_an_empty_scenario_succeeds() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();

        SaveHandler
            .execute(&mut ctx, &save_args(dir.path(), "scenario", "scenario.yml"))
            .expect("an entirely empty scenario is a legitimate save");
        assert!(dir.path().join("scenario.yml").exists());
    }

    #[test]
    fn test_saving_scenario_with_default_injection_and_override() {
        let dir = temp_project_dir();
        let mut ctx = ChromContext::new();
        ctx.set_scenario_initial_condition("zero");
        ctx.merge_scenario_default_injection(InjectionBuilder {
            injection_type: Some("Gaussian".to_string()),
            center: Some(10.0),
            width: Some(3.0),
            peak_concentration: Some(0.1),
            ..Default::default()
        });
        ctx.merge_scenario_species_override(
            "Erythorbic",
            InjectionBuilder {
                injection_type: Some("Dirac".to_string()),
                time: Some(5.0),
                amount: Some(0.05),
                ..Default::default()
            },
        );

        SaveHandler
            .execute(&mut ctx, &save_args(dir.path(), "scenario", "scenario.yml"))
            .expect("save must succeed");

        let content = std::fs::read_to_string(dir.path().join("scenario.yml")).unwrap();
        let value: serde_yaml::Value = serde_yaml::from_str(&content).unwrap();
        assert_eq!(value["initial_condition"].as_str(), Some("zero"));
        assert_eq!(
            value["default_injection"]["type"].as_str(),
            Some("Gaussian")
        );
        assert_eq!(
            value["injections"][0]["species"].as_str(),
            Some("Erythorbic")
        );
    }
}
