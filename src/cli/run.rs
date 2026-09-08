//! `run` command — orchestrates the full simulation pipeline.
//!
//! - [`RunHandler`]: loads model → scenario → solver, dispatches to the
//!   correct [`Solver`], and writes requested outputs (CSV, plot, JSON).
//! - `resolve_species_names`: detects the model type from the config file
//!   before `Box<dyn PhysicalModel>` erases it.
//! - `resolve_export_map`: deserialises the model into a concrete type to
//!   call [`Exportable::to_map`](crate::physics::Exportable) for JSON export.
//!
//! `resolve_source`/`resolve_new_outputs` and the private model-file
//! helpers (`read_model_file`/`peek_root_key`/`deserialise_inner`) are used
//! exclusively by [`RunHandler::execute`] and stay private to this module —
//! see [`crate::cli::support`] for the helpers shared with other commands.

use std::path::{Path, PathBuf};

use anyhow::anyhow;
use dynamic_cli::error::ExecutionError;
use dynamic_cli::{CommandHandler, DynamicCliError, ExecutionContext, ParsedArgs};

use crate::config::{Format, model::load_model, scenario::load_scenario, solver::load_solver};
use crate::models::{LangmuirMulti, LangmuirSingle};
use crate::output::export::{CsvConfig, CsvExporter, Exporter, to_json};
use crate::output::visualization::{plot_chromatogram, plot_chromatogram_multi};
use crate::physics::Exportable;
use crate::solver::{EulerSolver, RK4Solver, Scenario, SimulationResult, Solver};

use super::context::ChromContext;
use super::support::{path_to_str, to_cli_err};

// ============================================================================
// resolve_species_names
// ============================================================================

/// Reads the root key of a model file and, for multi-species models, returns
/// the species names in declaration order.
///
/// Returns an empty `Vec` for single-species models (`LangmuirSingle`).
/// This is the only place in `cli/` that knows about concrete model types —
/// the knowledge is available here because the file has not yet been erased
/// behind `Box<dyn PhysicalModel>`.
///
/// # Errors
///
/// Propagates I/O and parse errors from the config layer.
pub(super) fn resolve_species_names(model_path: &Path) -> anyhow::Result<Vec<String>> {
    let (format, content) = read_model_file(model_path)?;
    let root_key = peek_root_key(format, &content, model_path)?;

    if root_key != "LangmuirMulti" {
        return Ok(vec![]);
    }

    let model = deserialise_inner::<LangmuirMulti>(format, &content, "LangmuirMulti", model_path)?;
    Ok(model
        .species_names()
        .iter()
        .map(|s| s.to_string())
        .collect())
}

// ============================================================================
// resolve_export_map
// ============================================================================

/// Builds the JSON export map for the simulation result.
///
/// `Exportable::to_map` is defined on concrete model types, not on
/// `Box<dyn PhysicalModel>`. This helper re-reads and deserialises the model
/// file into the appropriate concrete type to call `to_map`, keeping all
/// concrete type knowledge confined to `cli/run.rs`.
///
/// # Errors
///
/// Propagates I/O, parse, and deserialisation errors.
pub(super) fn resolve_export_map(
    model_path: &Path,
    result: &SimulationResult,
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let (format, content) = read_model_file(model_path)?;
    let root_key = peek_root_key(format, &content, model_path)?;

    match root_key.as_str() {
        "LangmuirMulti" => {
            let model =
                deserialise_inner::<LangmuirMulti>(format, &content, "LangmuirMulti", model_path)?;
            Ok(model.to_map(
                &result.time_points,
                &result.state_trajectory,
                &result.metadata,
            ))
        }
        _ => {
            let model = deserialise_inner::<LangmuirSingle>(
                format,
                &content,
                "LangmuirSingle",
                model_path,
            )?;
            Ok(model.to_map(
                &result.time_points,
                &result.state_trajectory,
                &result.metadata,
            ))
        }
    }
}

// ============================================================================
// Private model-file helpers
// ============================================================================

/// Reads a model file and returns its detected format and raw content.
fn read_model_file(model_path: &Path) -> anyhow::Result<(Format, String)> {
    use crate::config::format_from_path;

    let path_str = model_path
        .to_str()
        .ok_or_else(|| anyhow!("model path is not valid UTF-8"))?;

    let format =
        format_from_path(path_str).map_err(|e| anyhow!("unsupported model file format: {e}"))?;

    let content = std::fs::read_to_string(model_path)
        .map_err(|e| anyhow!("cannot read '{}': {e}", model_path.display()))?;

    Ok((format, content))
}

/// Peeks at the root key of a YAML or JSON model file.
fn peek_root_key(format: Format, content: &str, model_path: &Path) -> anyhow::Result<String> {
    let key = match format {
        Format::Yaml => {
            let value: serde_yaml::Value = serde_yaml::from_str(content)
                .map_err(|e| anyhow!("YAML parse error in '{}': {e}", model_path.display()))?;
            value
                .as_mapping()
                .and_then(|m| m.keys().next())
                .and_then(|k| k.as_str())
                .unwrap_or("")
                .to_string()
        }
        Format::Json => {
            let value: serde_json::Value = serde_json::from_str(content)
                .map_err(|e| anyhow!("JSON parse error in '{}': {e}", model_path.display()))?;
            value
                .as_object()
                .and_then(|m| m.keys().next())
                .map(|k| k.as_str())
                .unwrap_or("")
                .to_string()
        }
    };
    Ok(key)
}

/// Extracts the inner value under `key` and deserialises it into `T`.
fn deserialise_inner<T>(
    format: Format,
    content: &str,
    key: &str,
    model_path: &Path,
) -> anyhow::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    // Normalise through serde_json::Value so both YAML and JSON share the same
    // deserialisation path into T.
    let root: serde_json::Value = match format {
        Format::Yaml => serde_yaml::from_str(content)
            .map_err(|e| anyhow!("YAML parse error in '{}': {e}", model_path.display()))?,
        Format::Json => serde_json::from_str(content)
            .map_err(|e| anyhow!("JSON parse error in '{}': {e}", model_path.display()))?,
    };

    let inner = root
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow!("missing '{}' key in '{}'", key, model_path.display()))?;

    serde_json::from_value(inner).map_err(|e| {
        anyhow!(
            "deserialisation error for '{}' in '{}': {e}",
            key,
            model_path.display()
        )
    })
}

// ============================================================================
// RunHandler
// ============================================================================

/// Handler for the `run` command.
///
/// Orchestrates the full simulation pipeline:
///
/// 1. Validate and apply `--project-dir` to the context.
/// 2. Resolve all file paths under the project directory.
/// 3. Detect species names from the model file.
/// 4. Load model → scenario → solver.
/// 5. Build [`Scenario`] and dispatch to the correct [`Solver`].
/// 6. Write requested outputs (CSV, plot, JSON export).
pub struct RunHandler;

impl CommandHandler for RunHandler {
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

        // ── 2. Resolve input file paths ──────────────────────────────────────
        let model_path = resolve_source(&project_dir, args, "model").map_err(to_cli_err)?;
        let scenario_path = resolve_source(&project_dir, args, "scenario").map_err(to_cli_err)?;
        let solver_path = resolve_source(&project_dir, args, "solver").map_err(to_cli_err)?;

        // ── 3. Detect species before Box<dyn PhysicalModel> erases the type ──
        let species_names = resolve_species_names(&model_path).map_err(to_cli_err)?;
        let is_multi = !species_names.is_empty();

        // ── 4. Load configuration ────────────────────────────────────────────
        let model_path_str = path_to_str(&model_path).map_err(to_cli_err)?;
        let scenario_path_str = path_to_str(&scenario_path).map_err(to_cli_err)?;
        let solver_path_str = path_to_str(&solver_path).map_err(to_cli_err)?;

        let mut model =
            load_model(model_path_str).map_err(|e| to_cli_err(anyhow!("loading model: {e}")))?;

        let boundaries = load_scenario(scenario_path_str, &mut *model)
            .map_err(|e| to_cli_err(anyhow!("loading scenario: {e}")))?;

        let solver_cfg =
            load_solver(solver_path_str).map_err(|e| to_cli_err(anyhow!("loading solver: {e}")))?;

        // ── 5. Build scenario and solve ──────────────────────────────────────
        // Capture spatial point count before model is moved into Scenario.
        let n_points = model.points();
        let scenario = Scenario::new(model, boundaries);

        let result = match solver_cfg.solver_name.as_str() {
            "RK4" => RK4Solver::new()
                .solve(&scenario, &solver_cfg.config)
                .map_err(|e| to_cli_err(anyhow!("RK4 solver: {e}")))?,
            "Euler" => EulerSolver::new()
                .solve(&scenario, &solver_cfg.config)
                .map_err(|e| to_cli_err(anyhow!("Euler solver: {e}")))?,
            other => {
                return Err(to_cli_err(anyhow!(
                    "unknown solver '{}' — expected 'RK4' or 'Euler'",
                    other
                )));
            }
        };

        println!(
            "Simulation complete — {} time points",
            result.time_points.len()
        );

        // ── 6. Outputs ───────────────────────────────────────────────────────
        // Each artifact type merges its legacy scalar option (0 or 1 path)
        // with any number of `--output <type> file=...` occurrences.
        // Multiple paths for the same type are allowed — write the same
        // artifact to several files, not an error.

        // CSV
        let mut csv_paths = resolve_new_outputs(&project_dir, args, "csv").map_err(to_cli_err)?;
        if let Some(csv_name) = args.get_scalar("output-csv") {
            csv_paths.push(project_dir.join(csv_name));
        }
        for csv_buf in &csv_paths {
            let csv_path = path_to_str(csv_buf).map_err(to_cli_err)?;
            let exporter = CsvExporter::new(CsvConfig::default());
            if is_multi {
                let name_refs: Vec<&str> = species_names.iter().map(|s| s.as_str()).collect();
                exporter
                    .export_multi(&result, None, &name_refs, csv_path)
                    .map_err(|e| to_cli_err(anyhow!("CSV export: {e}")))?;
            } else {
                exporter
                    .export_single(&result, None, csv_path)
                    .map_err(|e| to_cli_err(anyhow!("CSV export: {e}")))?;
            }
            println!("CSV written → {csv_path}");
        }

        // Plot — legacy `--output-plot` keeps its existing extension-sniffed
        // format detection unchanged; the new `svg`/`png` discriminants each
        // enforce their own extension (see `resolve_new_outputs`).
        let mut plot_paths = resolve_new_outputs(&project_dir, args, "svg").map_err(to_cli_err)?;
        plot_paths.extend(resolve_new_outputs(&project_dir, args, "png").map_err(to_cli_err)?);
        if let Some(plot_name) = args.get_scalar("output-plot") {
            plot_paths.push(project_dir.join(plot_name));
        }
        for plot_buf in &plot_paths {
            let plot_path = path_to_str(plot_buf).map_err(to_cli_err)?;
            if is_multi {
                let name_refs: Vec<&str> = species_names.iter().map(|s| s.as_str()).collect();
                plot_chromatogram_multi(&result, n_points, &name_refs, plot_path, None)
                    .map_err(|e| to_cli_err(anyhow!("plot: {e}")))?;
            } else {
                plot_chromatogram(&result, n_points, plot_path, None)
                    .map_err(|e| to_cli_err(anyhow!("plot: {e}")))?;
            }
            println!("Plot written → {plot_path}");
        }

        // JSON export
        let mut json_paths = resolve_new_outputs(&project_dir, args, "json").map_err(to_cli_err)?;
        if let Some(json_name) = args.get_scalar("export-json") {
            json_paths.push(project_dir.join(json_name));
        }
        for json_buf in &json_paths {
            let json_path = path_to_str(json_buf).map_err(to_cli_err)?;
            let map = resolve_export_map(&model_path, &result)
                .map_err(|e| to_cli_err(anyhow!("building export map: {e}")))?;
            to_json(&map, json_path).map_err(|e| to_cli_err(anyhow!("JSON export: {e}")))?;
            println!("JSON written → {json_path}");
        }

        Ok(())
    }
}

// ============================================================================
// Private helpers
// ============================================================================

/// Resolves a required source role (`model`, `scenario`, or `solver`) to a
/// [`PathBuf`] under `project_dir` — from *either* its legacy scalar option
/// (`--model`/`--scenario`/`--solver`) *or* a `--source <role> file=...`
/// occurrence, never both, never neither.
///
/// `role` doubles as the legacy option's long name — the two happen to be
/// spelled identically (`model`, `scenario`, `solver`), which is what lets
/// `args.get_scalar(role)` below stand in for the legacy lookup.
pub(crate) fn resolve_source(
    project_dir: &Path,
    args: &ParsedArgs,
    role: &str,
) -> anyhow::Result<PathBuf> {
    let legacy = args.get_scalar(role);

    let mut via_source: Vec<&str> = Vec::new();
    if let Some(occurrences) = args.get_repeated("source") {
        for occ in occurrences {
            if occ.discriminant == role {
                let file = occ.params.get("file").ok_or_else(|| {
                    anyhow!("'--source {role}' requires a 'file=...' sub-parameter")
                })?;
                via_source.push(file.as_str());
            }
        }
    }

    match via_source.len() {
        0 => match legacy {
            Some(path) => Ok(project_dir.join(path)),
            None => Err(anyhow!(
                "missing required source '{role}' — use '--{role} <file>' or '--source {role} file=<file>'"
            )),
        },
        1 => match legacy {
            Some(_) => Err(anyhow!(
                "'{role}' was given both via '--{role}' and '--source {role}' — use only one"
            )),
            None => Ok(project_dir.join(via_source[0])),
        },
        _ => Err(anyhow!(
            "'--source {role}' was given more than once — expected at most one"
        )),
    }
}

/// Collects every path requested for one `--output <discriminant>
/// file=...` occurrence — zero or more, unlike [`resolve_source`]:
/// writing the same artifact type to several files in one invocation is
/// allowed, not an ambiguity. For `svg`/`png`, also rejects a `file=` value
/// whose extension doesn't match the discriminant, so a typo in the
/// extension fails immediately with a clear message instead of silently
/// writing the "wrong" format for what the filename suggests.
pub(crate) fn resolve_new_outputs(
    project_dir: &Path,
    args: &ParsedArgs,
    discriminant: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if let Some(occurrences) = args.get_repeated("output") {
        for occ in occurrences {
            if occ.discriminant != discriminant {
                continue;
            }
            let file = occ.params.get("file").ok_or_else(|| {
                anyhow!("'--output {discriminant}' requires a 'file=...' sub-parameter")
            })?;
            if (discriminant == "svg" && !file.ends_with(".svg"))
                || (discriminant == "png" && !file.ends_with(".png"))
            {
                return Err(anyhow!(
                    "'--output {discriminant}' requires a file ending in .{discriminant}, got '{file}'"
                ));
            }
            paths.push(project_dir.join(file));
        }
    }
    Ok(paths)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{PhysicalData, PhysicalModel, PhysicalQuantity, PhysicalState};
    use std::collections::HashMap;
    use std::io::Write;

    // ── Fixtures YAML ─────────────────────────────────────────────────────────

    const SINGLE_YAML: &str = "\
LangmuirSingle:
  lambda: 1.2
  langmuir_k: 0.4
  port_number: 2.0
  column_length: 0.25
  n_points: 100
  dz: 0.0025
  fe: 1.5
  ue: 0.0025
  injection:
    type: None
";

    const MULTI_YAML: &str = "\
LangmuirMulti:
  species:
    - name: A
      lambda: 1.0
      langmuir_k: 0.5
      port_number: 1
      injection:
        type: None
    - name: B
      lambda: 1.0
      langmuir_k: 2.0
      port_number: 1
      injection:
        type: None
  n_points: 50
  porosity: 0.4
  velocity: 0.001
  column_length: 0.25
  dz: 0.005
  fe: 1.5
  ue: 0.0025
  stationary_fraction: 0.6
";

    /// Écrit un fichier YAML temporaire et retourne le handle.
    fn tmp_yaml(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".yml").tempfile().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    /// Écrit un fichier JSON temporaire et retourne le handle.
    fn tmp_json(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    /// Construit un `SimulationResult` minimal pour les tests d'export.
    fn minimal_result() -> SimulationResult {
        let state = PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::Vector(nalgebra::DVector::from_vec(vec![0.0; 100])),
        );
        SimulationResult::new(
            vec![0.0, 1.0, 2.0],
            vec![state.clone(), state.clone(), state.clone()],
            state,
        )
    }

    // ── resolve_source / resolve_source_optional ──────────────────────

    /// Builds a `ParsedArgs` with a single "source" key carrying the given
    /// (discriminant, file) occurrences, for testing `resolve_source*`
    /// without going through the full CLI parser.
    fn source_args(occurrences: &[(&str, &str)]) -> ParsedArgs {
        use dynamic_cli::parser::cli_parser::{OptionOccurrence, ParsedValue};

        let occs: Vec<OptionOccurrence> = occurrences
            .iter()
            .map(|(discriminant, file)| OptionOccurrence {
                discriminant: discriminant.to_string(),
                params: HashMap::from([("file".to_string(), file.to_string())]),
            })
            .collect();
        let mut map = HashMap::new();
        map.insert("source".to_string(), ParsedValue::Repeated(occs));
        ParsedArgs::new(map)
    }

    #[test]
    fn test_resolve_source_via_legacy_scalar() {
        let mut map = HashMap::new();
        map.insert("model".to_string(), "model.yml".to_string());
        let args = ParsedArgs::from_scalars(map);
        let result = resolve_source(Path::new("/proj"), &args, "model").unwrap();
        assert_eq!(result, PathBuf::from("/proj/model.yml"));
    }

    #[test]
    fn test_resolve_source_via_new_syntax() {
        let args = source_args(&[("model", "model.yml")]);
        let result = resolve_source(Path::new("/proj"), &args, "model").unwrap();
        assert_eq!(result, PathBuf::from("/proj/model.yml"));
    }

    #[test]
    fn test_resolve_source_missing_errors() {
        let args = ParsedArgs::from_scalars(HashMap::new());
        assert!(resolve_source(Path::new("."), &args, "model").is_err());
    }

    #[test]
    fn test_resolve_source_both_given_is_ambiguous() {
        use dynamic_cli::parser::cli_parser::{OptionOccurrence, ParsedValue};
        let mut combined = HashMap::new();
        combined.insert(
            "model".to_string(),
            ParsedValue::Scalar("legacy.yml".to_string()),
        );
        combined.insert(
            "source".to_string(),
            ParsedValue::Repeated(vec![OptionOccurrence {
                discriminant: "model".to_string(),
                params: HashMap::from([("file".to_string(), "new.yml".to_string())]),
            }]),
        );
        let args = ParsedArgs::new(combined);
        assert!(resolve_source(Path::new("."), &args, "model").is_err());
    }

    #[test]
    fn test_resolve_source_duplicate_new_syntax_errors() {
        let args = source_args(&[("model", "a.yml"), ("model", "b.yml")]);
        assert!(resolve_source(Path::new("."), &args, "model").is_err());
    }

    // ── read_model_file ───────────────────────────────────────────────────────

    #[test]
    fn test_read_model_file_yaml() {
        let f = tmp_yaml(SINGLE_YAML);
        let (format, content) = read_model_file(f.path()).unwrap();
        assert_eq!(format, Format::Yaml);
        assert!(content.contains("LangmuirSingle"));
    }

    #[test]
    fn test_read_model_file_json() {
        let json = r#"{"LangmuirSingle": {"lambda": 1.0}}"#;
        let f = tmp_json(json);
        let (format, _) = read_model_file(f.path()).unwrap();
        assert_eq!(format, Format::Json);
    }

    #[test]
    fn test_read_model_file_missing() {
        let p = PathBuf::from("/tmp/chrom_rs_missing_model.yml");
        assert!(read_model_file(&p).is_err());
    }

    // ── peek_root_key ─────────────────────────────────────────────────────────

    #[test]
    fn test_peek_root_key_yaml_single() {
        let f = tmp_yaml(SINGLE_YAML);
        let (fmt, content) = read_model_file(f.path()).unwrap();
        let key = peek_root_key(fmt, &content, f.path()).unwrap();
        assert_eq!(key, "LangmuirSingle");
    }

    #[test]
    fn test_peek_root_key_yaml_multi() {
        let f = tmp_yaml(MULTI_YAML);
        let (fmt, content) = read_model_file(f.path()).unwrap();
        let key = peek_root_key(fmt, &content, f.path()).unwrap();
        assert_eq!(key, "LangmuirMulti");
    }

    #[test]
    fn test_peek_root_key_json() {
        let json = r#"{"LangmuirSingle": {}}"#;
        let f = tmp_json(json);
        let (fmt, content) = read_model_file(f.path()).unwrap();
        let key = peek_root_key(fmt, &content, f.path()).unwrap();
        assert_eq!(key, "LangmuirSingle");
    }

    // ── resolve_species_names ─────────────────────────────────────────────────

    #[test]
    fn test_resolve_species_names_single_returns_empty() {
        let f = tmp_yaml(SINGLE_YAML);
        let names = resolve_species_names(f.path()).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn test_resolve_species_names_multi_returns_names() {
        let f = tmp_yaml(MULTI_YAML);
        let names = resolve_species_names(f.path()).unwrap();
        assert_eq!(names, vec!["A", "B"]);
    }

    #[test]
    fn test_resolve_species_names_missing_file() {
        let p = PathBuf::from("/tmp/chrom_rs_no_model.yml");
        assert!(resolve_species_names(&p).is_err());
    }

    // ── deserialise_inner ─────────────────────────────────────────────────────

    #[test]
    fn test_deserialise_inner_single_yaml() {
        let f = tmp_yaml(SINGLE_YAML);
        let (fmt, content) = read_model_file(f.path()).unwrap();
        let model: crate::models::LangmuirSingle =
            deserialise_inner(fmt, &content, "LangmuirSingle", f.path()).unwrap();
        assert_eq!(
            model.name(),
            "Langmuir single specie with temporal injection"
        );
    }

    #[test]
    fn test_deserialise_inner_multi_yaml() {
        let f = tmp_yaml(MULTI_YAML);
        let (fmt, content) = read_model_file(f.path()).unwrap();
        let model: crate::models::LangmuirMulti =
            deserialise_inner(fmt, &content, "LangmuirMulti", f.path()).unwrap();
        assert_eq!(model.species_names(), vec!["A", "B"]);
    }

    #[test]
    fn test_deserialise_inner_missing_key_returns_error() {
        let f = tmp_yaml(SINGLE_YAML);
        let (fmt, content) = read_model_file(f.path()).unwrap();
        let result: anyhow::Result<crate::models::LangmuirMulti> =
            deserialise_inner(fmt, &content, "LangmuirMulti", f.path());
        assert!(result.is_err());
    }

    // ── resolve_export_map ────────────────────────────────────────────────────

    #[test]
    fn test_resolve_export_map_single() {
        let f = tmp_yaml(SINGLE_YAML);
        let result = minimal_result();
        let map = resolve_export_map(f.path(), &result).unwrap();
        assert!(!map.is_empty());
    }

    #[test]
    fn test_resolve_export_map_multi() {
        let f = tmp_yaml(MULTI_YAML);
        // Multi needs a Matrix state — build a 50×2 trajectory
        let state = PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::Matrix(nalgebra::DMatrix::zeros(50, 2)),
        );
        let result =
            SimulationResult::new(vec![0.0, 1.0], vec![state.clone(), state.clone()], state);
        let map = resolve_export_map(f.path(), &result).unwrap();
        assert!(!map.is_empty());
    }

    #[test]
    fn test_resolve_export_map_missing_file() {
        let p = PathBuf::from("/tmp/chrom_rs_no_model.yml");
        let result = minimal_result();
        assert!(resolve_export_map(&p, &result).is_err());
    }
}
