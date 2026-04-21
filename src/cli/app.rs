//! Application layer for the `chrom-rs` CLI (DD-001).
//!
//! Groups all runtime concerns in one place:
//!
//! - [`ChromContext`] / [`ContextError`]: execution context and its invariants.
//! - [`RunHandler`]: the `run` command handler that orchestrates the full
//!   simulation pipeline.
//! - [`resolve_species_names`]: detects the model type from the config file
//!   before `Box<dyn PhysicalModel>` erases it.
//! - [`resolve_export_map`]: deserialises the model into a concrete type to
//!   call [`Exportable::to_map`](crate::physics::Exportable) for JSON export.
//!
//! # Path invariants
//!
//! `project_dir` is always a *safe* directory:
//! - No `..` component (prevents escaping the declared root).
//! - The directory exists and is readable by the current process.
//! - The directory is writable by the current process (output files land here).
//!
//! These invariants are enforced exclusively by
//! [`ChromContext::set_project_dir`].

use std::collections::HashMap;
use std::io;
use std::path::{Component, Path, PathBuf};

use anyhow::anyhow;
use dynamic_cli::error::ExecutionError;
use dynamic_cli::{CommandHandler, DynamicCliError, ExecutionContext};

use crate::config::{model::load_model, scenario::load_scenario, solver::load_solver};
use crate::models::{LangmuirMulti, LangmuirSingle};
use crate::output::export::{CsvConfig, CsvExporter, Exporter, to_json};
use crate::output::visualization::{plot_chromatogram, plot_chromatogram_multi};
use crate::physics::Exportable;
use crate::solver::{EulerSolver, RK4Solver, Scenario, SimulationResult, Solver};

// ============================================================================
// ContextError
// ============================================================================

/// Errors that can occur when configuring a [`ChromContext`].
#[derive(Debug)]
pub enum ContextError {
    /// The path contains a `..` component.
    PathTraversal(PathBuf),

    /// The path does not point to an existing directory.
    NotADirectory(PathBuf),

    /// The current process lacks read or write permission on the directory.
    PermissionDenied(PathBuf, io::Error),
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextError::PathTraversal(p) => write!(
                f,
                "project-dir '{}' contains '..': path traversal is not allowed",
                p.display()
            ),
            ContextError::NotADirectory(p) => write!(
                f,
                "project-dir '{}' does not exist or is not a directory",
                p.display()
            ),
            ContextError::PermissionDenied(p, e) => write!(
                f,
                "project-dir '{}': insufficient permissions — {e}",
                p.display()
            ),
        }
    }
}

impl std::error::Error for ContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ContextError::PermissionDenied(_, e) => Some(e),
            _ => None,
        }
    }
}

// ============================================================================
// ChromContext
// ============================================================================

/// Runtime state shared across all `chrom-rs` command handlers.
///
/// In v0.2.0 the only piece of state is `project_dir`: the directory
/// relative to which `--model`, `--scenario`, `--solver`, and all output
/// file names are resolved.
///
/// # Example
///
/// ```rust,no_run
/// use chrom_rs::cli::app::ChromContext;
///
/// let mut ctx = ChromContext::new();           // project_dir = "."
/// ctx.set_project_dir("experiments/run_01").unwrap();
/// assert_eq!(ctx.project_dir(), std::path::Path::new("experiments/run_01"));
/// ```
pub struct ChromContext {
    /// Root directory for all file-name resolution.
    ///
    /// Invariants enforced by [`set_project_dir`](Self::set_project_dir):
    /// no `..` component; existing, readable, writable directory.
    project_dir: PathBuf,
}

impl ChromContext {
    /// Creates a context whose project directory is the current working
    /// directory (`.`).
    pub fn new() -> Self {
        Self {
            project_dir: PathBuf::from("."),
        }
    }

    /// Returns the current project directory.
    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// Sets the project directory after validating the path.
    ///
    /// # Validation
    ///
    /// 1. Rejects any path containing a `..` component.
    /// 2. Checks that the path points to an existing directory.
    /// 3. Verifies read permission by listing the directory.
    /// 4. Verifies write permission by creating and removing a probe file.
    ///
    /// # Errors
    ///
    /// - [`ContextError::PathTraversal`] if `path` contains `..`.
    /// - [`ContextError::NotADirectory`] if `path` does not exist or is a file.
    /// - [`ContextError::PermissionDenied`] if the process cannot read or
    ///   write the directory.
    pub fn set_project_dir(&mut self, path: impl Into<PathBuf>) -> Result<(), ContextError> {
        let path = path.into();

        // 1 — Reject `..` regardless of position.
        if path.components().any(|c| c == Component::ParentDir) {
            return Err(ContextError::PathTraversal(path));
        }

        // 2 — Must be an existing directory.
        if !path.is_dir() {
            return Err(ContextError::NotADirectory(path));
        }

        // 3 — Read permission.
        std::fs::read_dir(&path).map_err(|e| ContextError::PermissionDenied(path.clone(), e))?;

        // 4 — Write permission: probe with a temporary file.
        let probe = path.join(".chrom_rs_write_probe");
        std::fs::File::create(&probe)
            .map_err(|e| ContextError::PermissionDenied(path.clone(), e))?;
        let _ = std::fs::remove_file(&probe);

        self.project_dir = path;
        Ok(())
    }
}

impl Default for ChromContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext for ChromContext {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ============================================================================
// Error conversion helper
// ============================================================================

/// Wraps any displayable error into [`DynamicCliError`] via
/// [`ExecutionError::CommandFailed`].
///
/// This is the single conversion point used throughout [`RunHandler::execute`]
/// to bridge `anyhow::Error` (and other error types) into the error type
/// required by `CommandHandler::execute`.
fn to_cli_err(e: impl Into<anyhow::Error>) -> DynamicCliError {
    ExecutionError::CommandFailed(e.into()).into()
}

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
/// concrete type knowledge confined to `cli/app.rs`.
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

use crate::config::Format;

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
        args: &HashMap<String, String>,
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

        let project_dir_str = args.get("project-dir").map(|s| s.as_str()).unwrap_or(".");

        chrom_ctx
            .set_project_dir(project_dir_str)
            .map_err(|e| to_cli_err(anyhow!("{e}")))?;

        let project_dir: PathBuf = chrom_ctx.project_dir().to_path_buf();

        // ── 2. Resolve input file paths ──────────────────────────────────────
        let model_path = resolve_input_path(&project_dir, args, "model").map_err(to_cli_err)?;
        let scenario_path =
            resolve_input_path(&project_dir, args, "scenario").map_err(to_cli_err)?;
        let solver_path = resolve_input_path(&project_dir, args, "solver").map_err(to_cli_err)?;

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

        // CSV
        if let Some(csv_name) = args.get("output-csv") {
            let csv_buf = project_dir.join(csv_name);
            let csv_path = path_to_str(&csv_buf).map_err(to_cli_err)?;
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

        // Plot
        if let Some(plot_name) = args.get("output-plot") {
            let plot_buf = project_dir.join(plot_name);
            let plot_path = path_to_str(&plot_buf).map_err(to_cli_err)?;
            if is_multi {
                let name_refs: Vec<&str> = species_names.iter().map(|s| s.as_str()).collect();
                plot_chromatogram_multi(
                    &result,
                    result.time_points.len(),
                    &name_refs,
                    plot_path,
                    None,
                )
                .map_err(|e| to_cli_err(anyhow!("plot: {e}")))?;
            } else {
                plot_chromatogram(&result, result.time_points.len(), plot_path, None)
                    .map_err(|e| to_cli_err(anyhow!("plot: {e}")))?;
            }
            println!("Plot written → {plot_path}");
        }

        // JSON export
        if let Some(json_name) = args.get("export-json") {
            let json_buf = project_dir.join(json_name);
            let json_path = path_to_str(&json_buf).map_err(to_cli_err)?;
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

/// Resolves a required option to a [`PathBuf`] under `project_dir`.
fn resolve_input_path(
    project_dir: &Path,
    args: &HashMap<String, String>,
    key: &str,
) -> anyhow::Result<PathBuf> {
    let name = args
        .get(key)
        .ok_or_else(|| anyhow!("missing required option '--{key}'"))?;
    Ok(project_dir.join(name))
}

/// Converts a [`Path`] to `&str`, rejecting non-UTF-8 paths.
fn path_to_str(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow!("path '{}' contains non-UTF-8 characters", path.display()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{PhysicalData, PhysicalModel, PhysicalQuantity, PhysicalState};
    use crate::solver::SimulationResult;
    use dynamic_cli::downcast_ref;
    use std::io::Write;

    // ── Fixtures YAML ─────────────────────────────────────────────────────────

    const SINGLE_YAML: &str = "\
LangmuirSingle:
  lambda: 1.2
  langmuir_k: 0.4
  port_number: 2.0
  length: 0.25
  nz: 100
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

    // ── ChromContext ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_defaults_to_current_dir() {
        let ctx = ChromContext::new();
        assert_eq!(ctx.project_dir(), Path::new("."));
    }

    #[test]
    fn test_default_equals_new() {
        let ctx = ChromContext::default();
        assert_eq!(ctx.project_dir(), Path::new("."));
    }

    #[test]
    fn test_set_project_dir_valid() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = ChromContext::new();
        ctx.set_project_dir(dir.path()).unwrap();
        assert_eq!(ctx.project_dir(), dir.path());
    }

    #[test]
    fn test_set_project_dir_rejects_parent_component() {
        let mut ctx = ChromContext::new();
        assert!(matches!(
            ctx.set_project_dir("some/../other"),
            Err(ContextError::PathTraversal(_))
        ));
    }

    #[test]
    fn test_set_project_dir_rejects_leading_parent() {
        let mut ctx = ChromContext::new();
        assert!(matches!(
            ctx.set_project_dir("../sibling"),
            Err(ContextError::PathTraversal(_))
        ));
    }

    #[test]
    fn test_set_project_dir_rejects_missing_path() {
        let mut ctx = ChromContext::new();
        assert!(matches!(
            ctx.set_project_dir("/tmp/chrom_rs_does_not_exist_xyz"),
            Err(ContextError::NotADirectory(_))
        ));
    }

    #[test]
    fn test_set_project_dir_rejects_file_path() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut ctx = ChromContext::new();
        assert!(matches!(
            ctx.set_project_dir(file.path()),
            Err(ContextError::NotADirectory(_))
        ));
    }

    // ── ExecutionContext downcast ─────────────────────────────────────────────

    #[test]
    fn test_as_any_downcast_ref() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = ChromContext::new();
        ctx.set_project_dir(dir.path()).unwrap();
        let boxed: Box<dyn ExecutionContext> = Box::new(ctx);
        let recovered = downcast_ref::<ChromContext>(boxed.as_ref()).unwrap();
        assert_eq!(recovered.project_dir(), dir.path());
    }

    #[test]
    fn test_as_any_mut_downcast_mut() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        let mut ctx = ChromContext::new();
        ctx.set_project_dir(dir1.path()).unwrap();
        let any_mut = ctx.as_any_mut();
        let recovered = any_mut.downcast_mut::<ChromContext>().unwrap();
        recovered.set_project_dir(dir2.path()).unwrap();
        assert_eq!(ctx.project_dir(), dir2.path());
    }

    // ── ContextError ─────────────────────────────────────────────────────────

    #[test]
    fn test_display_path_traversal() {
        let e = ContextError::PathTraversal(PathBuf::from("a/../b"));
        assert!(e.to_string().contains(".."));
        assert!(e.to_string().contains("path traversal"));
    }

    #[test]
    fn test_display_not_a_directory() {
        let e = ContextError::NotADirectory(PathBuf::from("/no/such/dir"));
        assert!(e.to_string().contains("no/such/dir"));
    }

    #[test]
    fn test_display_permission_denied() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = ContextError::PermissionDenied(PathBuf::from("/locked"), io_err);
        assert!(e.to_string().contains("locked"));
        assert!(e.to_string().contains("permissions"));
    }

    #[test]
    fn test_source_permission_denied_has_source() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = ContextError::PermissionDenied(PathBuf::from("/locked"), io_err);
        assert!(e.source().is_some());
    }

    #[test]
    fn test_source_path_traversal_is_none() {
        use std::error::Error;
        let e = ContextError::PathTraversal(PathBuf::from("a/b"));
        assert!(e.source().is_none());
    }

    // ── to_cli_err ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_cli_err_produces_command_failed() {
        use dynamic_cli::error::DynamicCliError;
        let err = to_cli_err(anyhow!("test error"));
        assert!(matches!(err, DynamicCliError::Execution(_)));
    }

    // ── path_to_str ──────────────────────────────────────────────────────────

    #[test]
    fn test_path_to_str_valid_utf8() {
        let p = PathBuf::from("/tmp/results.csv");
        assert_eq!(path_to_str(&p).unwrap(), "/tmp/results.csv");
    }

    #[test]
    fn test_path_to_str_rejects_non_utf8() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let bad = OsStr::from_bytes(&[0xff, 0xfe]);
        let p = PathBuf::from(bad);
        assert!(path_to_str(&p).is_err());
    }

    // ── resolve_input_path ────────────────────────────────────────────────────

    #[test]
    fn test_resolve_input_path_found() {
        let mut args = HashMap::new();
        args.insert("model".to_string(), "model.yml".to_string());
        let result = resolve_input_path(Path::new("/proj"), &args, "model").unwrap();
        assert_eq!(result, PathBuf::from("/proj/model.yml"));
    }

    #[test]
    fn test_resolve_input_path_missing_key() {
        let args = HashMap::new();
        assert!(resolve_input_path(Path::new("."), &args, "model").is_err());
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
