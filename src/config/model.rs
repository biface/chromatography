//! Loader for `model.yml` / `model.json`.
//!
//! `model.yml` describes the physical column and its parameters.
//! The injection profile is intentionally absent (or set to `None`) here —
//! it is supplied by `scenario.yml` and applied by [`scenario::load_scenario`].
//!
//! # File format
//!
//! The file is deserialised via `typetag`, which uses the type name as the
//! root key. Supported formats: `.yml` / `.yaml` (serde_yaml) and `.json`
//! (serde_json). The extension determines the parser.
//!
//! ## `LangmuirSingle` example
//!
//! ```yaml
//! LangmuirSingle:
//!   lambda: 1.2
//!   langmuir_k: 0.4
//!   port_number: 2.0
//!   porosity: 0.4
//!   velocity: 0.001
//!   column_length: 0.25
//!   nz: 100
//!   injection:
//!     type: None
//! ```
//!
//! ## `LangmuirMulti` example
//!
//! ```yaml
//! LangmuirMulti:
//!   species:
//!     - name: Ascorbic
//!       lambda: 1.0
//!       langmuir_k: 1.1
//!       port_number: 2
//!       injection:
//!         type: None
//!     - name: Erythorbic
//!       lambda: 1.0
//!       langmuir_k: 1.7
//!       port_number: 2
//!       injection:
//!         type: None
//!   n_points: 100
//!   porosity: 0.4
//!   velocity: 0.001
//!   column_length: 0.25
//! ```
//!
//! The `injection: type: None` placeholder is replaced by
//! [`scenario::load_scenario`] after the model is loaded.

use crate::config::ConfigError;
use crate::physics::PhysicalModel;

/// Loads a physical model from a configuration file.
///
/// The file is deserialised via `typetag` — the root key must match the
/// registered type name (e.g. `LangmuirSingle`, `LangmuirMulti`).
///
/// The returned model has its injection profile set to `None` by convention.
/// Call [`scenario::load_scenario`](super::scenario::load_scenario) to apply
/// the injection defined in `scenario.yml`.
///
/// # Errors
///
/// - [`ConfigError::Io`] if the file cannot be read.
/// - [`ConfigError::UnsupportedFormat`] if the extension is not `.yml`,
///   `.yaml`, or `.json`.
/// - [`ConfigError::Parse`] if the file content is not valid YAML/JSON or
///   does not match a known `PhysicalModel` type.
///
/// # Example
///
/// ```rust,no_run
/// use chrom_rs::config::model::load_model;
///
/// let model = load_model("model_tfa.yml").unwrap();
/// println!("Loaded: {}", model.name());
/// ```
pub fn load_model(path: &str) -> Result<Box<dyn PhysicalModel>, ConfigError> {
    crate::config::load_from_file::<Box<dyn PhysicalModel>>(path)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Writes a temporary YAML file and returns it (kept alive for the test).
    fn tmp_yaml(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".yml").tempfile().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    /// Writes a temporary JSON file and returns it (kept alive for the test).
    fn tmp_json(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    const LANGMUIR_SINGLE_YAML: &str = "\
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

    const LANGMUIR_MULTI_YAML: &str = "\
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

    // ── Success paths ─────────────────────────────────────────────────────────

    #[test]
    fn test_load_langmuir_single_yaml() {
        let f = tmp_yaml(LANGMUIR_SINGLE_YAML);
        let model = load_model(f.path().to_str().unwrap())
            .expect("load_model must succeed for LangmuirSingle YAML");

        assert_eq!(
            model.name(),
            "Langmuir single specie with temporal injection"
        );
        assert_eq!(model.points(), 100);
    }

    #[test]
    fn test_load_langmuir_multi_yaml() {
        let f = tmp_yaml(LANGMUIR_MULTI_YAML);
        let model = load_model(f.path().to_str().unwrap())
            .expect("load_model must succeed for LangmuirMulti YAML");

        assert_eq!(model.name(), "Langmuir Multi-Species");
        assert_eq!(model.points(), 50);
    }

    #[test]
    fn test_load_langmuir_single_json() {
        let json = r#"{
  "LangmuirSingle": {
    "lambda": 1.2,
    "langmuir_k": 0.4,
    "port_number": 2.0,
    "length": 0.25,
    "nz": 100,
    "dz": 0.0025,
    "fe": 1.5,
    "ue": 0.0025,
    "injection": { "type": "None" }
  }
}"#;
        let f = tmp_json(json);
        let model = load_model(f.path().to_str().unwrap())
            .expect("load_model must succeed for LangmuirSingle JSON");

        assert_eq!(model.points(), 100);
    }

    // ── Error paths ───────────────────────────────────────────────────────────

    #[test]
    fn test_load_model_file_not_found() {
        let result = load_model("/tmp/does_not_exist_chrom_rs_model.yml");
        assert!(
            matches!(result, Err(ConfigError::Io(_))),
            "Missing file must yield ConfigError::Io"
        );
    }

    #[test]
    fn test_load_model_unsupported_extension() {
        let result = load_model("model.toml");
        assert!(
            matches!(result, Err(ConfigError::UnsupportedFormat(_))),
            "Unknown extension must yield ConfigError::UnsupportedFormat"
        );
    }

    #[test]
    fn test_load_model_invalid_yaml_syntax() {
        let f = tmp_yaml("not: valid: yaml: [");
        let result = load_model(f.path().to_str().unwrap());
        assert!(
            matches!(result, Err(ConfigError::Parse(_))),
            "Malformed YAML must yield ConfigError::Parse"
        );
    }

    #[test]
    fn test_load_model_unknown_type() {
        let f = tmp_yaml("UnknownModel:\n  field: 42\n");
        let result = load_model(f.path().to_str().unwrap());
        assert!(
            matches!(result, Err(ConfigError::Parse(_))),
            "Unknown typetag type must yield ConfigError::Parse"
        );
    }
}
