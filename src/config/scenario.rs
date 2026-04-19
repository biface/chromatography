//! Loader for `scenario.yml` / `scenario.json` (DD-015).
//!
//! `scenario.yml` defines the **boundary conditions** for a simulation run:
//! the initial column state and the temporal injection profile(s). It is
//! intentionally decoupled from `model.yml` so that the same physical model
//! can be driven by different injection protocols without modification.
//!
//! # File format
//!
//! ```yaml
//! # Initial column state
//! initial_condition: zero   # only supported value in v0.2.0
//!
//! # Default injection applied to all species not listed in `injections`
//! default_injection:
//!   type: Gaussian
//!   center: 10.0
//!   width: 3.0
//!   peak_concentration: 0.1
//!
//! # Optional per-species overrides (LangmuirMulti only)
//! injections:
//!   - species: Erythorbic
//!     type: Dirac
//!     time: 5.0
//!     amount: 0.05
//! ```
//!
//! If `default_injection` is absent, all species keep the `None` profile
//! they received from `load_model`. If `injections` is absent or empty,
//! `default_injection` applies to all species.
//!
//! # Design note
//!
//! This loader does **not** deserialise into an intermediate struct.
//! It reads the YAML/JSON fields, constructs [`TemporalInjection`] values
//! directly, and applies them to the model via [`PhysicalModel::set_default_injection`]
//! and [`PhysicalModel::set_injection_for_species`]. The result is a fully
//! initialised [`DomainBoundaries`] built from `initial_condition`.

use crate::config::{ConfigError, Format, format_from_path};
use crate::models::TemporalInjection;
use crate::physics::PhysicalModel;
use crate::solver::DomainBoundaries;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

// ============================================================================
// Public API
// ============================================================================

/// Loads boundary conditions from a scenario file and applies injections to
/// the model.
///
/// # Arguments
///
/// * `path`  — Path to `scenario.yml` or `scenario.json`.
/// * `model` — Mutable reference to the model loaded by
///   [`model::load_model`](super::model::load_model). Injections defined in
///   the scenario are applied in-place via the [`PhysicalModel`] trait.
///
/// # Returns
///
/// A [`DomainBoundaries`] built from `initial_condition`. Pass it to
/// [`Scenario::new`](crate::solver::Scenario::new) along with the model.
///
/// # Errors
///
/// - [`ConfigError::Io`] if the file cannot be read.
/// - [`ConfigError::UnsupportedFormat`] if the extension is not recognised.
/// - [`ConfigError::Parse`] if the content is not valid YAML/JSON.
/// - [`ConfigError::Validation`] if:
///   - `initial_condition` has an unrecognised value.
///   - An injection type is unrecognised or missing required fields.
///   - A named species in `injections` is not found in the model.
///
/// # Example
///
/// ```rust,no_run
/// use chrom_rs::config::{model, scenario, solver};
/// use chrom_rs::solver::Scenario;
///
/// let mut phys_model = model::load_model("model_tfa.yml").unwrap();
/// let boundaries = scenario::load_scenario("scenario_step.yml", &mut *phys_model).unwrap();
/// let solver_cfg = solver::load_solver("solver_rk4.yml").unwrap();
///
/// let sc = Scenario::new(phys_model, boundaries);
/// ```
pub fn load_scenario(
    path: &str,
    model: &mut dyn PhysicalModel,
) -> Result<DomainBoundaries, ConfigError> {
    let content = fs::read_to_string(path)?;

    // Normalise to a serde_json::Value regardless of source format.
    let root: Value = match format_from_path(path)? {
        Format::Yaml => {
            serde_yaml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?
        }
        Format::Json => {
            serde_json::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?
        }
    };

    // ── initial_condition ─────────────────────────────────────────────────────
    let _initial_cond = root
        .get("initial_condition")
        .and_then(|v| v.as_str())
        .unwrap_or("zero");

    // v0.2.0: only "zero" is supported.
    if _initial_cond != "zero" {
        return Err(ConfigError::Validation(format!(
            "unsupported initial_condition '{_initial_cond}' (only 'zero' is supported in v0.2.0)"
        )));
    }

    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);

    // ── Build injection map ───────────────────────────────────────────────────
    //
    // The map is constructed in full before calling set_injections once.
    // Key None   → default injection applied to all unlisted species.
    // Key Some   → per-species override.
    let mut injection_map: HashMap<Option<String>, crate::models::TemporalInjection> =
        HashMap::new();

    if let Some(inj_val) = root.get("default_injection") {
        let inj = parse_injection(inj_val)?;
        injection_map.insert(None, inj);
    }

    if let Some(Value::Array(overrides)) = root.get("injections") {
        for entry in overrides {
            let species = entry
                .get("species")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ConfigError::Validation(
                        "each entry in 'injections' must have a 'species' field".into(),
                    )
                })?;

            let inj = parse_injection(entry)?;
            injection_map.insert(Some(species.to_string()), inj);
        }
    }

    if !injection_map.is_empty() {
        model
            .set_injections(&injection_map)
            .map_err(ConfigError::Validation)?;
    }

    Ok(boundaries)
}

// ============================================================================
// Injection parser
// ============================================================================

/// Parses a `TemporalInjection` from a JSON `Value` node.
///
/// Accepts the same field names as `TemporalInjectionSnapshot` (the internal
/// serde representation). The `"type"` field selects the variant; remaining
/// fields supply the parameters.
///
/// | `type`      | Required fields                              |
/// |-------------|----------------------------------------------|
/// | `Dirac`     | `time`, `amount`                             |
/// | `Gaussian`  | `center`, `width`, `peak_concentration`      |
/// | `Rectangle` | `start`, `end`, `concentration`              |
/// | `None`      | *(none)*                                     |
fn parse_injection(v: &Value) -> Result<TemporalInjection, ConfigError> {
    let kind = v
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| ConfigError::Validation("injection block missing 'type' field".into()))?;

    match kind {
        "Dirac" => {
            let time = req_f64(v, "time")?;
            let amount = req_f64(v, "amount")?;
            Ok(TemporalInjection::dirac(time, amount))
        }
        "Gaussian" => {
            let center = req_f64(v, "center")?;
            let width = req_f64(v, "width")?;
            let peak = req_f64(v, "peak_concentration")?;
            Ok(TemporalInjection::gaussian(center, width, peak))
        }
        "Rectangle" => {
            let start = req_f64(v, "start")?;
            let end = req_f64(v, "end")?;
            let concentration = req_f64(v, "concentration")?;
            Ok(TemporalInjection::rectangle(start, end, concentration))
        }
        "None" => Ok(TemporalInjection::none()),
        other => Err(ConfigError::Validation(format!(
            "unknown injection type '{other}' (expected Dirac, Gaussian, Rectangle, or None)"
        ))),
    }
}

/// Extracts a required `f64` field from a JSON object.
fn req_f64(v: &Value, key: &str) -> Result<f64, ConfigError> {
    v.get(key)
        .and_then(|f| f.as_f64())
        .ok_or_else(|| ConfigError::Validation(format!("injection missing required field '{key}'")))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{LangmuirSingle, TemporalInjection};
    use std::io::Write;

    // ── Helpers ───────────────────────────────────────────────────────────────

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

    fn single_model() -> LangmuirSingle {
        LangmuirSingle::new(
            1.2,
            0.4,
            2.0,
            0.4,
            0.001,
            0.25,
            100,
            TemporalInjection::none(),
        )
    }

    // ── initial_condition ─────────────────────────────────────────────────────

    #[test]
    fn test_load_scenario_zero_ic() {
        let f = tmp_yaml("initial_condition: zero\n");
        let mut model = single_model();
        let boundaries = load_scenario(f.path().to_str().unwrap(), &mut model).unwrap();
        assert!(boundaries.is_time_dependent());
        assert!(boundaries.initial_condition().is_some());
    }

    #[test]
    fn test_load_scenario_missing_ic_defaults_to_zero() {
        // No initial_condition key → defaults to "zero"
        let f = tmp_yaml("default_injection:\n  type: None\n");
        let mut model = single_model();
        let result = load_scenario(f.path().to_str().unwrap(), &mut model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_scenario_unsupported_ic() {
        let f = tmp_yaml("initial_condition: uniform\n");
        let mut model = single_model();
        let result = load_scenario(f.path().to_str().unwrap(), &mut model);
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    // ── default_injection ─────────────────────────────────────────────────────

    #[test]
    fn test_load_scenario_default_dirac() {
        let yaml = "initial_condition: zero\ndefault_injection:\n  type: Dirac\n  time: 5.0\n  amount: 0.1\n";
        let f = tmp_yaml(yaml);
        let mut model = single_model();
        load_scenario(f.path().to_str().unwrap(), &mut model).unwrap();

        // Dirac at t=5 → non-zero at peak
        assert!(model.injection().evaluate(5.0) > 0.0);
    }

    #[test]
    fn test_load_scenario_default_gaussian() {
        let yaml = "initial_condition: zero\ndefault_injection:\n  type: Gaussian\n  center: 10.0\n  width: 3.0\n  peak_concentration: 0.1\n";
        let f = tmp_yaml(yaml);
        let mut model = single_model();
        load_scenario(f.path().to_str().unwrap(), &mut model).unwrap();
        assert!(model.injection().evaluate(10.0) > 0.0);
    }

    #[test]
    fn test_load_scenario_default_rectangle() {
        let yaml = "initial_condition: zero\ndefault_injection:\n  type: Rectangle\n  start: 0.0\n  end: 10.0\n  concentration: 0.5\n";
        let f = tmp_yaml(yaml);
        let mut model = single_model();
        load_scenario(f.path().to_str().unwrap(), &mut model).unwrap();
        assert!(model.injection().evaluate(5.0) > 0.0);
        assert_eq!(model.injection().evaluate(15.0), 0.0);
    }

    #[test]
    fn test_load_scenario_default_none() {
        let yaml = "initial_condition: zero\ndefault_injection:\n  type: None\n";
        let f = tmp_yaml(yaml);
        let mut model = single_model();
        load_scenario(f.path().to_str().unwrap(), &mut model).unwrap();
        assert_eq!(model.injection().evaluate(0.0), 0.0);
    }

    // ── per-species injections ────────────────────────────────────────────────

    #[test]
    fn test_load_scenario_per_species_override() {
        use crate::models::{LangmuirMulti, SpeciesParams};

        let sp_a = SpeciesParams::new("A", 1.0, 0.5, 1, TemporalInjection::none());
        let sp_b = SpeciesParams::new("B", 1.0, 2.0, 1, TemporalInjection::none());
        let mut model = LangmuirMulti::new(vec![sp_a, sp_b], 50, 0.4, 0.001, 0.25).unwrap();

        let yaml = "initial_condition: zero\ndefault_injection:\n  type: None\ninjections:\n  - species: B\n    type: Dirac\n    time: 5.0\n    amount: 0.1\n";
        let f = tmp_yaml(yaml);
        load_scenario(f.path().to_str().unwrap(), &mut model).unwrap();

        let params = model.species_params();
        // A: still None
        assert_eq!(params[0].injection.evaluate(5.0), 0.0);
        // B: Dirac at t=5
        assert!(params[1].injection.evaluate(5.0) > 0.0);
    }

    #[test]
    fn test_load_scenario_per_species_unknown_name() {
        use crate::models::{LangmuirMulti, SpeciesParams};

        let sp = SpeciesParams::new("A", 1.0, 0.5, 1, TemporalInjection::none());
        let mut model = LangmuirMulti::new(vec![sp], 50, 0.4, 0.001, 0.25).unwrap();

        let yaml = "initial_condition: zero\ninjections:\n  - species: Unknown\n    type: None\n";
        let f = tmp_yaml(yaml);
        let result = load_scenario(f.path().to_str().unwrap(), &mut model);
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    // ── injection parse errors ────────────────────────────────────────────────

    #[test]
    fn test_load_scenario_unknown_injection_type() {
        let yaml = "initial_condition: zero\ndefault_injection:\n  type: Step\n";
        let f = tmp_yaml(yaml);
        let mut model = single_model();
        let result = load_scenario(f.path().to_str().unwrap(), &mut model);
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    #[test]
    fn test_load_scenario_dirac_missing_field() {
        let yaml = "initial_condition: zero\ndefault_injection:\n  type: Dirac\n  time: 5.0\n";
        // "amount" is missing
        let f = tmp_yaml(yaml);
        let mut model = single_model();
        let result = load_scenario(f.path().to_str().unwrap(), &mut model);
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    #[test]
    fn test_load_scenario_injection_missing_type() {
        let yaml = "initial_condition: zero\ndefault_injection:\n  center: 5.0\n";
        let f = tmp_yaml(yaml);
        let mut model = single_model();
        let result = load_scenario(f.path().to_str().unwrap(), &mut model);
        assert!(matches!(result, Err(ConfigError::Validation(_))));
    }

    // ── format / I/O errors ───────────────────────────────────────────────────

    #[test]
    fn test_load_scenario_json_format() {
        let json = r#"{"initial_condition":"zero","default_injection":{"type":"None"}}"#;
        let f = tmp_json(json);
        let mut model = single_model();
        let result = load_scenario(f.path().to_str().unwrap(), &mut model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_scenario_file_not_found() {
        let mut model = single_model();
        let result = load_scenario("/tmp/does_not_exist_chrom_rs_scenario.yml", &mut model);
        assert!(matches!(result, Err(ConfigError::Io(_))));
    }

    #[test]
    fn test_load_scenario_invalid_yaml() {
        let f = tmp_yaml("not: valid: yaml: [");
        let mut model = single_model();
        let result = load_scenario(f.path().to_str().unwrap(), &mut model);
        assert!(matches!(result, Err(ConfigError::Parse(_))));
    }
}
