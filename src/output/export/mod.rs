//! Export module for simulation results.
//!
//! # Architecture
//!
//! This module defines the [`Exporter`] trait that abstracts the export format.
//! Each format is an independent implementation in its own sub-module.
//! This design follows the **Open/Closed principle**: adding a new format
//! means adding a file, without ever modifying existing code.
//!
//! # Available formats
//!
//! | Format  | Module          | Version |
//! |---------|-----------------|---------|
//! | CSV     | [`csv`]         | v0.1.0  |
//! | ODS     | `ods` (future)  | v0.2.0  |
//! | Scilab  | `scilab` (future)| v0.2.0 |
//! | Matlab  | `matlab` (future)| v0.2.0 |
//!
//! # Usage example
//!
//! ```rust,ignore
//! use chrom_rs::output::export::csv::CsvExporter;
//! use chrom_rs::output::export::Exporter;
//!
//! let exporter = CsvExporter::default();
//!
//! // Full export (all time steps)
//! exporter.export_single(&result, None, "tfa.csv")?;
//!
//! // Downsampled export to 500 points
//! exporter.export_single(&result, Some(500), "tfa_light.csv")?;
//!
//! // Multi-species export with labels
//! exporter.export_multi(&result, None, &["Ascorbic", "Erythorbic"], "acids.csv")?;
//! ```

pub mod csv;

// Re-export the most commonly used types at the module level so users can write:
//   use chrom_rs::output::export::{CsvExporter, CsvConfig, CsvError};
// instead of the full sub-module path.
pub use csv::{CsvConfig, CsvError, CsvExporter};

use crate::solver::SimulationResult;

/// Abstraction trait for all export formats.
///
/// # Associated type `Error`
///
/// Each format manages its own errors via the associated type.
/// This avoids systematic boxing (`Box<dyn Error>`) and allows
/// the caller to react precisely based on the error type.
///
/// # Parameter `n_points`
///
/// - `None`: exports all time steps (default behaviour)
/// - `Some(n)`: uniformly downsamples to `n` points,
///   always guaranteeing that the **first and last** points
///   are included (important to capture the end of the chromatographic peak)
///
/// # Implementing this trait
///
/// A new format must implement [`export_single`] and [`export_multi`].
/// Formats that do not distinguish between the two cases can delegate one to the other.
pub trait Exporter {
    /// Error type specific to this export format.
    type Error: std::error::Error;

    /// Exports a single-species result.
    ///
    /// The file contains two columns: `time` and `c_outlet`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - the path is invalid or the directory does not exist
    /// - `result` contains no data
    fn export_single(
        &self,
        result: &SimulationResult,
        n_points: Option<usize>,
        path: &str,
    ) -> Result<(), Self::Error>;

    /// Exports a multi-species result with an envelope column (`c_total`).
    ///
    /// The file contains: `time`, `c_total`, then one column per species.
    /// The `c_total` column is the sum of all species at each time step
    /// (represents the raw detector signal, which does not distinguish species).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - the number of names in `species_names` does not match the data
    /// - the path is invalid
    /// - `result` contains no data
    fn export_multi(
        &self,
        result: &SimulationResult,
        n_points: Option<usize>,
        species_names: &[&str],
        path: &str,
    ) -> Result<(), Self::Error>;
}