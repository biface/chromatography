//! Loader for `solver.yml` / `solver.json` (DD-015).
//!
//! `solver.yml` defines **how** to solve the scenario: numerical method,
//! time discretisation, and optional trajectory subsampling.
//!
//! # File format
//!
//! ```yaml
//! type: RK4          # or Euler
//! total_time: 600.0
//! time_steps: 10000
//! step: null         # null = full trajectory; integer N = every N-th step
//! ```
//!
//! The file is deserialised directly into [`SolverConfiguration`] via serde.
//! The `type` field is mapped to [`SolverType::TimeEvolution`].
//!
//! # Supported solver types
//!
//! | YAML `type` | Maps to |
//! |---|---|
//! | `RK4` | `SolverType::TimeEvolution` dispatched to [`RK4Solver`] |
//! | `Euler` | `SolverType::TimeEvolution` dispatched to [`EulerSolver`] |
//!
//! The solver name is stored as metadata in `SimulationResult` after the run.
//! The CLI selects the concrete solver implementation based on this field.

use crate::config::{ConfigError, load_from_file};
use crate::solver::SolverConfiguration;

// ============================================================================
// SolverFile — intermediate deserialisation struct
// ============================================================================

/// Flat representation of `solver.yml` / `solver.json`.
///
/// Deserialised from the file, then converted to [`SolverConfiguration`] +
/// a solver name string used by the CLI to select the concrete solver.
#[derive(Debug, serde::Deserialize)]
struct SolverFile {
    /// Solver algorithm name: `"RK4"` or `"Euler"`.
    #[serde(rename = "type")]
    solver_type: String,

    /// Total simulation time $T$ **\[s\]**.
    total_time: f64,

    /// Number of time steps $N_t$.
    time_steps: usize,

    /// Trajectory subsampling interval (DD-010 / DD-015).
    ///
    /// `null` or absent → full trajectory. `N` → every N-th step kept.
    #[serde(default)]
    step: Option<usize>,
}

// ============================================================================
// Public API
// ============================================================================

/// Output of [`load_solver`]: the solver configuration and the algorithm name.
pub struct SolverConfig {
    /// Numerical integration parameters.
    pub config: SolverConfiguration,

    /// Solver algorithm name as written in `solver.yml` (`"RK4"` or `"Euler"`).
    ///
    /// Used by the CLI to instantiate the correct [`Solver`](crate::solver::Solver)
    /// implementation.
    pub solver_name: String,
}

/// Loads a solver configuration from a file.
///
/// # Errors
///
/// - [`ConfigError::Io`] if the file cannot be read.
/// - [`ConfigError::UnsupportedFormat`] if the extension is not recognised.
/// - [`ConfigError::Parse`] if the content is not valid YAML/JSON.
/// - [`ConfigError::Validation`] if:
///   - `total_time` ≤ 0.
///   - `time_steps` = 0.
///   - `type` is not `"RK4"` or `"Euler"`.
///
/// # Example
///
/// ```rust,no_run
/// use chrom_rs::config::solver::load_solver;
///
/// let sc = load_solver("solver_rk4.yml").unwrap();
/// println!("solver: {}, step: {:?}", sc.solver_name, sc.config.step);
/// ```
pub fn load_solver(path: &str) -> Result<SolverConfig, ConfigError> {
    let raw: SolverFile = load_from_file(path)?;

    // Validate solver type
    if raw.solver_type != "RK4" && raw.solver_type != "Euler" {
        return Err(ConfigError::Validation(format!(
            "unknown solver type '{}' (expected 'RK4' or 'Euler')",
            raw.solver_type
        )));
    }

    // Validate numerical parameters
    if raw.total_time <= 0.0 {
        return Err(ConfigError::Validation(format!(
            "total_time must be > 0, got {}",
            raw.total_time
        )));
    }

    if raw.time_steps == 0 {
        return Err(ConfigError::Validation("time_steps must be > 0".into()));
    }

    let mut config = SolverConfiguration::time_evolution(raw.total_time, raw.time_steps);
    if let Some(n) = raw.step {
        config = config.with_step(n);
    }

    Ok(SolverConfig {
        config,
        solver_name: raw.solver_type,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_yaml(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".yml").tempfile().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    fn tmp_json(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    const RK4_YAML: &str = "\
type: RK4
total_time: 600.0
time_steps: 10000
step: null
";

    const EULER_YAML: &str = "\
type: Euler
total_time: 300.0
time_steps: 5000
";

    // ── Success paths ─────────────────────────────────────────────────────────

    #[test]
    fn test_load_rk4_yaml() {
        let f = tmp_yaml(RK4_YAML);
        let sc = load_solver(f.path().to_str().unwrap()).unwrap();

        assert_eq!(sc.solver_name, "RK4");
        assert!(sc.config.validate().is_ok());
        assert!(sc.config.step.is_none());
    }

    #[test]
    fn test_load_euler_yaml() {
        let f = tmp_yaml(EULER_YAML);
        let sc = load_solver(f.path().to_str().unwrap()).unwrap();

        assert_eq!(sc.solver_name, "Euler");
        assert!(sc.config.validate().is_ok());
    }

    #[test]
    fn test_load_with_step() {
        let yaml = "type: RK4\ntotal_time: 600.0\ntime_steps: 10000\nstep: 50\n";
        let f = tmp_yaml(yaml);
        let sc = load_solver(f.path().to_str().unwrap()).unwrap();

        assert_eq!(sc.config.step, Some(50));
    }

    #[test]
    fn test_load_json_format() {
        let json = r#"{"type":"RK4","total_time":600.0,"time_steps":10000}"#;
        let f = tmp_json(json);
        let sc = load_solver(f.path().to_str().unwrap()).unwrap();

        assert_eq!(sc.solver_name, "RK4");
        assert!(sc.config.step.is_none());
    }

    #[test]
    fn test_load_step_absent_defaults_to_none() {
        // No step field → None
        let yaml = "type: Euler\ntotal_time: 100.0\ntime_steps: 1000\n";
        let f = tmp_yaml(yaml);
        let sc = load_solver(f.path().to_str().unwrap()).unwrap();
        assert!(sc.config.step.is_none());
    }

    // ── Validation errors ─────────────────────────────────────────────────────

    #[test]
    fn test_unknown_solver_type() {
        let yaml = "type: Adams\ntotal_time: 600.0\ntime_steps: 10000\n";
        let f = tmp_yaml(yaml);
        let result = load_solver(f.path().to_str().unwrap());
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    #[test]
    fn test_negative_total_time() {
        let yaml = "type: RK4\ntotal_time: -1.0\ntime_steps: 1000\n";
        let f = tmp_yaml(yaml);
        let result = load_solver(f.path().to_str().unwrap());
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    #[test]
    fn test_zero_time_steps() {
        let yaml = "type: RK4\ntotal_time: 600.0\ntime_steps: 0\n";
        let f = tmp_yaml(yaml);
        let result = load_solver(f.path().to_str().unwrap());
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    // ── I/O and format errors ─────────────────────────────────────────────────

    #[test]
    fn test_file_not_found() {
        let result = load_solver("/tmp/does_not_exist_chrom_rs_solver.yml");
        assert!(matches!(result, Err(ConfigError::Io(_))));
    }

    #[test]
    fn test_unsupported_extension() {
        let result = load_solver("solver.toml");
        assert!(matches!(result, Err(ConfigError::UnsupportedFormat(_))));
    }

    #[test]
    fn test_invalid_yaml_syntax() {
        let f = tmp_yaml("not: valid: yaml: [");
        let result = load_solver(f.path().to_str().unwrap());
        assert!(matches!(result, Err(ConfigError::Parse(_))));
    }
}
