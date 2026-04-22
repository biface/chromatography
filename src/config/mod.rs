//! Configuration file loading for `chrom-rs`.
//!
//! Provides three independent loaders — one per configuration file — that
//! reflect the architecture separation between what and how at the user level :
//!
//! | Loader | File | Returns |
//! |---|---|---|
//! | [`load_model`](crate::config::model::load_model) | `model.yml` | `Box<dyn PhysicalModel>` |
//! | [`load_scenario`](crate::config::scenario::load_scenario) | `scenario.yml` | `DomainBoundaries` |
//! | [`load_solver`](crate::config::solver::load_solver) | `solver.yml` | `SolverConfiguration` |//!
//! # Supported formats
//!
//! Both JSON and YAML are accepted. The format is inferred from the file
//! extension: `.yml` / `.yaml` → serde_yaml, `.json` → serde_json.
//!
//! # Example
//!
//! ```rust,no_run
//! use chrom_rs::config::{model, scenario, solver};
//! use chrom_rs::solver::Scenario;
//!
//! let mut phys_model = model::load_model("model_tfa.yml").unwrap();
//! let boundaries = scenario::load_scenario("scenario_step.yml", &mut *phys_model).unwrap();
//! let solver_config = solver::load_solver("solver_rk4.yml").unwrap();
//!
//! let scenario = Scenario::new(phys_model, boundaries);
//! ```

use std::fmt;
use std::fs;
use std::path::Path;

/// Loader for `model.yml` — returns a boxed [`PhysicalModel`](crate::physics::PhysicalModel).
pub mod model;
/// Loader for `scenario.yml` — applies injections and returns [`DomainBoundaries`](crate::solver::DomainBoundaries).
pub mod scenario;
/// Loader for `solver.yml` — returns a [`SolverConfiguration`](crate::solver::SolverConfiguration).
pub mod solver;

// ============================================================================
// ConfigError
// ============================================================================

/// Errors that can occur while loading a configuration file.
#[derive(Debug)]
pub enum ConfigError {
    /// The file could not be read (not found, permission denied, …).
    Io(std::io::Error),

    /// The file extension is not `.yml`, `.yaml`, or `.json`.
    UnsupportedFormat(String),

    /// The file content is syntactically invalid (YAML / JSON parse error).
    Parse(String),

    /// The file parsed successfully but violates a domain constraint
    /// (e.g. a species named in `injections` does not exist in the model).
    Validation(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "config I/O error: {e}"),
            ConfigError::UnsupportedFormat(ext) => {
                write!(
                    f,
                    "unsupported config format '{ext}' (use .yml, .yaml or .json)"
                )
            }
            ConfigError::Parse(msg) => write!(f, "config parse error: {msg}"),
            ConfigError::Validation(msg) => write!(f, "config validation error: {msg}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::Io(e)
    }
}

// ============================================================================
// Format detection
// ============================================================================

/// Recognised configuration file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    Yaml,
    Json,
}

/// Infers the file format from the path extension.
///
/// # Errors
///
/// Returns [`ConfigError::UnsupportedFormat`] if the extension is absent or
/// not one of `.yml`, `.yaml`, `.json`.
pub(crate) fn format_from_path(path: &str) -> Result<Format, ConfigError> {
    match Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("yml") | Some("yaml") => Ok(Format::Yaml),
        Some("json") => Ok(Format::Json),
        Some(other) => Err(ConfigError::UnsupportedFormat(other.to_string())),
        None => Err(ConfigError::UnsupportedFormat(String::new())),
    }
}

// ============================================================================
// Generic loader helper
// ============================================================================

/// Reads a file and deserialises it into `T`, dispatching on the extension.
///
/// Used by [`model::load_model`] and [`solver::load_solver`] where the target
/// type implements `serde::de::DeserializeOwned` directly. The scenario loader
/// uses [`format_from_path`] + manual field extraction instead, because
/// `scenario.yml` feeds multiple existing types rather than a single struct.
pub(crate) fn load_from_file<T>(path: &str) -> Result<T, ConfigError>
where
    T: serde::de::DeserializeOwned,
{
    // Check format first — returns UnsupportedFormat before any I/O attempt.
    let format = format_from_path(path)?;
    let content = fs::read_to_string(path)?;
    match format {
        Format::Yaml => {
            serde_yaml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))
        }
        Format::Json => {
            serde_json::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── format_from_path ──────────────────────────────────────────────────────

    #[test]
    fn test_format_yml() {
        assert_eq!(format_from_path("model.yml").unwrap(), Format::Yaml);
    }

    #[test]
    fn test_format_yaml() {
        assert_eq!(
            format_from_path("config/solver.yaml").unwrap(),
            Format::Yaml
        );
    }

    #[test]
    fn test_format_json() {
        assert_eq!(
            format_from_path("results/model.json").unwrap(),
            Format::Json
        );
    }

    #[test]
    fn test_format_case_insensitive() {
        assert_eq!(format_from_path("model.YML").unwrap(), Format::Yaml);
        assert_eq!(format_from_path("model.JSON").unwrap(), Format::Json);
    }

    #[test]
    fn test_format_unsupported_extension() {
        let err = format_from_path("model.toml").unwrap_err();
        assert!(matches!(err, ConfigError::UnsupportedFormat(ref s) if s == "toml"));
    }

    #[test]
    fn test_format_no_extension() {
        let err = format_from_path("model").unwrap_err();
        assert!(matches!(err, ConfigError::UnsupportedFormat(ref s) if s.is_empty()));
    }

    // ── ConfigError display ───────────────────────────────────────────────────

    #[test]
    fn test_display_unsupported_format() {
        let e = ConfigError::UnsupportedFormat("toml".to_string());
        assert!(e.to_string().contains("toml"));
    }

    #[test]
    fn test_display_parse() {
        let e = ConfigError::Parse("unexpected field".to_string());
        assert!(e.to_string().contains("unexpected field"));
    }

    #[test]
    fn test_display_validation() {
        let e = ConfigError::Validation("species 'X' not found".to_string());
        assert!(e.to_string().contains("species 'X'"));
    }

    #[test]
    fn test_display_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let e = ConfigError::from(io_err);
        assert!(e.to_string().contains("I/O"));
    }

    // ── load_from_file ────────────────────────────────────────────────────────

    #[test]
    fn test_load_from_file_yaml() {
        use std::io::Write;
        let mut f = tempfile::Builder::new().suffix(".yml").tempfile().unwrap();
        writeln!(
            f,
            "solver_type: !TimeEvolution\n  total_time: 10.0\n  time_steps: 100"
        )
        .unwrap();

        use crate::solver::SolverConfiguration;
        let config: SolverConfiguration = load_from_file(f.path().to_str().unwrap()).unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_load_from_file_json() {
        use std::io::Write;
        let mut f = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        writeln!(
            f,
            r#"{{"solver_type":{{"TimeEvolution":{{"total_time":10.0,"time_steps":100}}}}}}"#
        )
        .unwrap();

        use crate::solver::SolverConfiguration;
        let config: SolverConfiguration = load_from_file(f.path().to_str().unwrap()).unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_load_from_file_missing_file() {
        use crate::solver::SolverConfiguration;
        let result: Result<SolverConfiguration, _> =
            load_from_file("/tmp/does_not_exist_chrom_rs_config.yml");
        assert!(matches!(result, Err(ConfigError::Io(_))));
    }

    #[test]
    fn test_load_from_file_invalid_yaml() {
        use std::io::Write;
        let mut f = tempfile::Builder::new().suffix(".yml").tempfile().unwrap();
        writeln!(f, "not: valid: yaml: at: all: [").unwrap();

        use crate::solver::SolverConfiguration;
        let result: Result<SolverConfiguration, _> = load_from_file(f.path().to_str().unwrap());
        assert!(matches!(result, Err(ConfigError::Parse(_))));
    }
}
