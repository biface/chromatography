//! JSON export and import for simulation results.
//!
//! # Design
//!
//! This module provides two **universal** functions that operate on a
//! `serde_json::Map` — the exchange format produced by
//! [`crate::physics::Exportable::to_map`].
//!
//! The functions have no knowledge of physical models: they read and write
//! JSON files from/to a generic map. All domain-specific structure lives in
//! the [`Exportable`](crate::physics::Exportable) implementations on the
//! model types.
//!
//! # Responsibilities
//!
//! This module is **I/O only**. It does not know about physical models,
//! `SimulationResult`, or any domain type. The full pipeline is:
//!
//! ```text
//! CLI layer
//!   model.to_map(&time_points, &trajectory, &metadata)  →  Map<String, Value>
//!   to_json(&map, path)                                  →  file
//!   from_json(path)                                      →  Map<String, Value>
//!   Model::from_map(map)                                 →  model
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use chrom_rs::output::export::{to_json, from_json};
//! use serde_json::{Map, Value, json};
//!
//! // Produced by model.to_map(...) in the CLI layer
//! let map: Map<String, Value> = json!({
//!     "metadata": { "solver": "RK4", "model": "LangmuirSingle" },
//!     "data": {
//!         "time_points": [0.0, 0.06, 0.12],
//!         "profiles": {
//!             "species_0": { "Concentration": [0.0, 1.2e-4, 2.1e-4] }
//!         }
//!     }
//! })
//! .as_object()
//! .cloned()
//! .unwrap();
//!
//! to_json(&map, "/tmp/result.json").unwrap();
//! let reloaded = from_json("/tmp/result.json").unwrap();
//! ```

use serde_json::{Map, Value};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter};

// ============================================================================
// JsonError
// ============================================================================

/// Errors that can occur during JSON file I/O.
#[derive(Debug)]
pub enum JsonError {
    /// System error: unable to open, read, or write the file.
    Io(std::io::Error),

    /// The file content is not valid JSON, or the JSON structure is unexpected.
    Serialization(serde_json::Error),
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonError::Io(e) => write!(f, "JSON I/O error: {e}"),
            JsonError::Serialization(e) => write!(f, "JSON serialization error: {e}"),
        }
    }
}

impl std::error::Error for JsonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            JsonError::Io(e) => Some(e),
            JsonError::Serialization(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for JsonError {
    fn from(e: std::io::Error) -> Self {
        JsonError::Io(e)
    }
}

impl From<serde_json::Error> for JsonError {
    fn from(e: serde_json::Error) -> Self {
        JsonError::Serialization(e)
    }
}

// ============================================================================
// Universal I/O functions
// ============================================================================

/// Writes a JSON map to a file (pretty-printed).
///
/// The map is typically produced by
/// [`Exportable::to_map`](crate::physics::Exportable::to_map).
///
/// # Errors
///
/// - [`JsonError::Io`] if the file cannot be created or written.
/// - [`JsonError::Serialization`] if the map cannot be serialized (unlikely
///   for maps produced by `to_map`).
///
/// # Example
///
/// ```rust,no_run
/// use chrom_rs::output::export::to_json;
/// use serde_json::{Map, Value};
///
/// let mut map = Map::new();
/// map.insert("key".into(), Value::from(42.0));
/// let _ = to_json(&map, "/tmp/test.json");
/// ```
pub fn to_json(map: &Map<String, Value>, path: &str) -> Result<(), JsonError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, map)?;
    Ok(())
}

/// Reads a JSON file and returns its content as a generic map.
///
/// The map can then be passed to
/// [`Exportable::from_map`](crate::physics::Exportable::from_map)
/// to reconstruct a physical model.
///
/// # Errors
///
/// - [`JsonError::Io`] if the file cannot be opened or read.
/// - [`JsonError::Serialization`] if the file content is not a valid JSON
///   object (arrays and primitives at the top level are rejected).
///
/// # Example
///
/// ```rust,no_run
/// use chrom_rs::output::export::from_json;
///
/// let map = from_json("/tmp/result.json");
/// ```
pub fn from_json(path: &str) -> Result<Map<String, Value>, JsonError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let value: Value = serde_json::from_reader(reader)?;
    value.as_object().cloned().ok_or_else(|| {
        JsonError::Serialization(serde_json::from_str::<Value>("not_an_object").unwrap_err())
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_map() -> Map<String, Value> {
        let v = json!({
            "metadata": { "solver": "RK4", "model": "TestModel" },
            "data": {
                "time_points": [0.0, 1.0, 2.0],
                "profiles": {
                    "species_0": { "Concentration": [0.0, 0.5, 1.0] }
                }
            }
        });
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn test_to_json_creates_file() {
        let map = make_map();
        let path = "/tmp/chrom_rs_test_json.json";
        to_json(&map, path).expect("to_json should succeed");
        assert!(std::path::Path::new(path).exists());
    }

    #[test]
    fn test_round_trip() {
        let original = make_map();
        let path = "/tmp/chrom_rs_test_roundtrip.json";

        to_json(&original, path).expect("write");
        let loaded = from_json(path).expect("read");

        let solver = loaded["metadata"]["solver"].as_str().unwrap();
        assert_eq!(solver, "RK4");

        let tp = loaded["data"]["time_points"].as_array().unwrap();
        assert_eq!(tp.len(), 3);
        assert!((tp[1].as_f64().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_from_json_missing_file() {
        let result = from_json("/tmp/does_not_exist_chrom_rs.json");
        assert!(matches!(result, Err(JsonError::Io(_))));
    }

    #[test]
    fn test_from_json_not_object() {
        let path = "/tmp/chrom_rs_test_array.json";
        std::fs::write(path, "[1, 2, 3]").unwrap();
        let result = from_json(path);
        assert!(matches!(result, Err(JsonError::Serialization(_))));
    }
}
