//! CSV export for simulation results.
//!
//! # Produced format
//!
//! The default separator is `;` (European convention), directly readable
//! without configuration in Excel and LibreOffice Calc.
//! Concentrations are written in scientific notation to avoid any
//! ambiguity on orders of magnitude (typically 1e-4 to 1e-2 mol/L).
//!
//! ## Single-species
//!
//! ```text
//! time (s);c_outlet (mol/L)
//! 0.000000e0;0.000000e0
//! 6.000000e-2;1.234567e-4
//! ...
//! ```
//!
//! ## Multi-species
//!
//! ```text
//! time (s);c_total (mol/L);Ascorbic (mol/L);Erythorbic (mol/L)
//! 0.000000e0;0.000000e0;0.000000e0;0.000000e0
//! 6.000000e-2;1.500000e-4;9.000000e-5;6.000000e-5
//! ...
//! ```
//!
//! # Downsampling
//!
//! When `n_points = Some(n)` with `n < total_steps`, indices are selected
//! uniformly with stride = `total / (n - 1)`.
//! The first (t=0) and last points are **always included**:
//! the trailing edge of a chromatographic peak must never be truncated.
//!
//! # Example
//!
//! ```rust,ignore
//! use chrom_rs::output::export::csv::CsvExporter;
//! use chrom_rs::output::export::Exporter;
//!
//! let exporter = CsvExporter::default();
//!
//! // All simulation points
//! exporter.export_single(&result, None, "tfa.csv")?;
//!
//! // Reduced to 1000 points for a lighter file
//! exporter.export_multi(&result, Some(1000), &["Ascorbic", "Erythorbic"], "acids.csv")?;
//! ```

use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::physics::{PhysicalData, PhysicalQuantity};
use crate::solver::SimulationResult;

use super::Exporter;

// =============================================================================
// CsvError
// =============================================================================

/// Possible errors during a CSV export.
///
/// This custom type allows the caller to distinguish failure causes
/// and react precisely (e.g. suggest another path on `Io`,
/// or correct species names on `SpeciesCountMismatch`).
#[derive(Debug)]
pub enum CsvError {
    /// System error: unable to open or write the file.
    ///
    /// Common causes: directory does not exist, insufficient permissions,
    /// disk space exhausted.
    Io(std::io::Error),

    /// The `SimulationResult` contains no time steps.
    ///
    /// This indicates an upstream problem (solver not run, empty result).
    EmptyResult,

    /// The number of names provided does not match the number of species
    /// in the simulation result.
    ///
    /// # Fields
    /// - `expected`: number of species in the concentration vector
    /// - `got`: number of names provided by the caller
    SpeciesCountMismatch {
        expected: usize,
        got: usize,
    },
}

// Display implementation for human-readable error messages.
// Required by std::error::Error.
impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CsvError::Io(e) => write!(f, "CSV I/O error: {e}"),
            CsvError::EmptyResult => {
                write!(f, "CSV export failed: SimulationResult contains no time points")
            }
            CsvError::SpeciesCountMismatch { expected, got } => write!(
                f,
                "CSV export failed: expected {expected} species names, got {got}"
            ),
        }
    }
}

// std::error::Error is required by the `type Error: std::error::Error` constraint
// on the Exporter trait. The source is the underlying io error when available.
impl std::error::Error for CsvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CsvError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// Automatic conversion from io::Error → CsvError to allow `?` in file-writing functions.
impl From<std::io::Error> for CsvError {
    fn from(e: std::io::Error) -> Self {
        CsvError::Io(e)
    }
}

// =============================================================================
// CsvConfig
// =============================================================================

/// Configuration for the produced CSV format.
///
/// Controls the column separator and numeric precision.
/// For scientific use, scientific notation is always used
/// (regardless of the `precision` value).
///
/// # Example
///
/// ```rust,ignore
/// // Comma separator (Anglo-Saxon style) with 4 decimal places
/// let config = CsvConfig {
///     separator: ',',
///     precision: 4,
/// };
/// let exporter = CsvExporter::new(config);
/// ```
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Column separator.
    ///
    /// - `;` (default): European convention, compatible with Excel/LibreOffice without import wizard
    /// - `,`: Anglo-Saxon convention, compatible with Python pandas out of the box
    pub separator: char,

    /// Number of significant digits in scientific notation.
    ///
    /// Example: `precision = 6` → `1.234567e-4`
    /// Recommended value: 6 (sufficient physical precision, compact file)
    pub precision: usize,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            separator: ';',  // European convention: readable without config in LibreOffice
            precision: 6,    // 6 significant digits: reasonable physical precision
        }
    }
}

// =============================================================================
// CsvExporter
// =============================================================================

/// CSV format exporter.
///
/// Built with a [`CsvConfig`] to control the separator and precision.
/// Uses a [`BufWriter`] to minimise system calls when writing
/// (important for simulations with many time steps).
///
/// # Creation
///
/// ```rust,ignore
/// // Default configuration (separator ';', 6 decimal places)
/// let exporter = CsvExporter::default();
///
/// // Custom configuration
/// let exporter = CsvExporter::new(CsvConfig { separator: ',', precision: 4 });
/// ```
#[derive(Debug, Clone, Default)]
pub struct CsvExporter {
    pub config: CsvConfig,
}

impl CsvExporter {
    /// Creates an exporter with a custom configuration.
    pub fn new(config: CsvConfig) -> Self {
        Self { config }
    }
}

// =============================================================================
// Exporter trait implementation
// =============================================================================

impl Exporter for CsvExporter {
    type Error = CsvError;

    /// Exports a single-species result to a CSV file.
    ///
    /// # Produced format
    ///
    /// ```text
    /// time (s);c_outlet (mol/L)
    /// 0.000000e0;0.000000e0
    /// 6.000000e-2;1.234567e-4
    /// ```
    ///
    /// # Errors
    ///
    /// - [`CsvError::EmptyResult`] if `result.time_points` is empty
    /// - [`CsvError::Io`] if the file cannot be created or written
    fn export_single(
        &self,
        result: &SimulationResult,
        n_points: Option<usize>,
        path: &str,
    ) -> Result<(), CsvError> {
        // Guard: reject empty results before touching the filesystem
        if result.time_points.is_empty() {
            return Err(CsvError::EmptyResult);
        }

        // Compute the indices to export (all or a uniform subset)
        let indices = compute_sample_indices(result.time_points.len(), n_points);

        // Open file with a buffer — BufWriter groups small writes to reduce syscalls,
        // which matters when exporting tens of thousands of rows
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let sep = self.config.separator;
        let prec = self.config.precision;

        // Header: column names include units to be self-describing
        writeln!(writer, "time (s){sep}c_outlet (mol/L)")?;

        // Data rows
        for idx in indices {
            let t = result.time_points[idx];

            // Extract outlet concentration (last spatial point = column exit).
            // For single-species the state holds a scalar or a 1-element vector.
            let c = extract_scalar_concentration(&result.state_trajectory[idx])?;

            writeln!(writer, "{t:.prec$e}{sep}{c:.prec$e}")?;
        }

        // Explicit flush to surface any deferred write error as CsvError::Io
        // (BufWriter would flush implicitly on Drop, but silently discards errors there)
        writer.flush()?;

        Ok(())
    }

    /// Exports a multi-species result to a CSV file.
    ///
    /// # Produced format
    ///
    /// ```text
    /// time (s);c_total (mol/L);Ascorbic (mol/L);Erythorbic (mol/L)
    /// 0.000000e0;0.000000e0;0.000000e0;0.000000e0
    /// ```
    ///
    /// The `c_total` column is the **sum** of all species at each time step.
    /// It represents the raw detector signal (which does not distinguish species).
    ///
    /// # Errors
    ///
    /// - [`CsvError::EmptyResult`] if `result.time_points` is empty
    /// - [`CsvError::SpeciesCountMismatch`] if the number of names does not match
    /// - [`CsvError::Io`] if the file cannot be created or written
    fn export_multi(
        &self,
        result: &SimulationResult,
        n_points: Option<usize>,
        species_names: &[&str],
        path: &str,
    ) -> Result<(), CsvError> {
        // Guard: reject empty results before touching the filesystem
        if result.time_points.is_empty() {
            return Err(CsvError::EmptyResult);
        }

        // Check species count consistency against the first state in the trajectory
        let n_species = extract_vector_concentrations(&result.state_trajectory[0])?.len();
        if species_names.len() != n_species {
            return Err(CsvError::SpeciesCountMismatch {
                expected: n_species,
                got: species_names.len(),
            });
        }

        let indices = compute_sample_indices(result.time_points.len(), n_points);

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let sep = self.config.separator;
        let prec = self.config.precision;

        // Header: time + envelope + one column per species
        write!(writer, "time (s){sep}c_total (mol/L)")?;
        for name in species_names {
            write!(writer, "{sep}{name} (mol/L)")?;
        }
        writeln!(writer)?;

        // Data rows
        for idx in indices {
            let t = result.time_points[idx];
            let concentrations = extract_vector_concentrations(&result.state_trajectory[idx])?;

            // c_total = sum of all species (global detector signal)
            let c_total: f64 = concentrations.iter().sum();

            write!(writer, "{t:.prec$e}{sep}{c_total:.prec$e}")?;
            for c in &concentrations {
                write!(writer, "{sep}{c:.prec$e}")?;
            }
            writeln!(writer)?;
        }

        writer.flush()?;

        Ok(())
    }
}

// =============================================================================
// Private helper functions
// =============================================================================

/// Computes the indices to export according to the downsampling strategy.
///
/// # Behaviour
///
/// - `None` → all indices `0..total`
/// - `Some(n)` with `n >= total` → all indices (no reduction)
/// - `Some(1)` → only the first index `[0]`
/// - `Some(n)` → `n` uniformly spaced indices, with the first (`0`) and
///   last (`total - 1`) **always included**
///
/// # Why always include the last point?
///
/// In chromatography, the peak ends with a descending tail.
/// Losing the last point would visually truncate the chromatogram
/// and skew the peak integral (area = quantity of matter).
///
/// # Example
///
/// ```text
/// total = 100, n = 5
/// stride = 100 / 4 = 25
/// indices = [0, 25, 50, 75, 99]  ← 99 always present
/// ```
fn compute_sample_indices(total: usize, n_points: Option<usize>) -> Vec<usize> {
    match n_points {
        // No downsampling: return every index
        None => (0..total).collect(),

        Some(n) if n == 0 || n >= total => {
            // n=0 or n larger than total: export everything
            // (defensive: do not reject, just adapt)
            (0..total).collect()
        }

        Some(1) => {
            // Degenerate case: single point → first one (t=0)
            vec![0]
        }

        Some(n) => {
            // General case: n uniformly spaced points
            // Stride computed over n-1 intervals to include exactly n points
            let mut indices = Vec::with_capacity(n);

            for i in 0..n {
                // Formula: index = round(i * (total - 1) / (n - 1))
                // Guarantees i=0 → 0 and i=n-1 → total-1
                let idx = (i * (total - 1)) / (n - 1);
                indices.push(idx);
            }

            // Explicit guarantee for the last point (guard against integer rounding)
            // The formula above already guarantees it, but defensive code is safer.
            if let Some(last) = indices.last_mut() {
                *last = total - 1;
            }

            indices
        }
    }
}

/// Extracts the scalar concentration from a `PhysicalState`.
///
/// Handles both possible representations of a single-species state:
/// - `PhysicalData::Scalar(c)` → returns `c` directly
/// - `PhysicalData::Vector(v)` → returns `v[last]` (last spatial point = column outlet)
///
/// # Errors
///
/// Returns [`CsvError::EmptyResult`] if the `Concentration` quantity is absent.
fn extract_scalar_concentration(
    state: &crate::physics::PhysicalState,
) -> Result<f64, CsvError> {
    let data = state
        .get(PhysicalQuantity::Concentration)
        .ok_or(CsvError::EmptyResult)?;

    let value = match data {
        PhysicalData::Scalar(c) => *c,
        PhysicalData::Vector(v) => {
            // For a spatial profile, the detector sits at the column outlet
            // = last spatial point (index n-1)
            *v.iter().last().unwrap_or(&0.0)
        }
        // Other variants (Matrix, etc.) are not used in 1D chromatography
        _ => 0.0,
    };

    Ok(value)
}

/// Extracts the multi-species concentration vector from a `PhysicalState`.
///
/// Handles two storage layouts used by multi-species models:
///
/// - `PhysicalData::Vector(v)` — concentrations already collapsed to a 1D vector
///   (e.g. ODE-only models without spatial discretisation)
/// - `PhysicalData::Matrix(m)` — `[n_points × n_species]` spatial layout used by
///   [`LangmuirMulti`]; the **last row** (`n_points - 1`) is the column outlet and
///   each column is one species
///
/// # Return
///
/// A `Vec<f64>` where index `i` corresponds to species `i` at the column outlet.
///
/// # Errors
///
/// Returns [`CsvError::EmptyResult`] if the `Concentration` quantity is absent
/// or if the matrix has no rows.
fn extract_vector_concentrations(
    state: &crate::physics::PhysicalState,
) -> Result<Vec<f64>, CsvError> {
    let data = state
        .get(PhysicalQuantity::Concentration)
        .ok_or(CsvError::EmptyResult)?;

    let values = match data {
        // 1D vector: concentrations are already species-indexed
        PhysicalData::Vector(v) => v.iter().copied().collect(),

        // 2D matrix [n_points × n_species]: outlet = last row, one column per species
        PhysicalData::Matrix(m) => {
            let last_row = m.nrows().checked_sub(1).ok_or(CsvError::EmptyResult)?;
            (0..m.ncols()).map(|s| m[(last_row, s)]).collect()
        }

        // Scalar fallback: single-species without spatial layout
        PhysicalData::Scalar(c) => vec![*c],

        _ => vec![],
    };

    Ok(values)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Helpers — building minimal SimulationResult for tests
    // -------------------------------------------------------------------------
    //
    // We build SimulationResult values directly in memory, without going
    // through a real solver. This isolates the csv module tests from the
    // rest of the codebase (true unit tests).

    use crate::physics::{PhysicalData, PhysicalQuantity, PhysicalState};
    use crate::solver::SimulationResult;

    /// Builds a single-species `SimulationResult` with controlled values.
    ///
    /// Concentrations: c[i] = i as f64 * 0.001 (linear increase)
    fn make_single_result(n_steps: usize) -> SimulationResult {
        let time_points: Vec<f64> = (0..n_steps).map(|i| i as f64 * 0.1).collect();

        let state_trajectory: Vec<PhysicalState> = (0..n_steps)
            .map(|i| {
                let c = i as f64 * 0.001;
                PhysicalState::new(
                    PhysicalQuantity::Concentration,
                    PhysicalData::Scalar(c),
                )
            })
            .collect();

        let final_state = state_trajectory.last().unwrap().clone();
        SimulationResult::new(time_points, state_trajectory, final_state)
    }

    /// Builds a multi-species `SimulationResult` with `n_species` species.
    ///
    /// Concentrations: c[i][s] = i * 0.001 + s * 0.0001
    fn make_multi_result(n_steps: usize, n_species: usize) -> SimulationResult {
        let time_points: Vec<f64> = (0..n_steps).map(|i| i as f64 * 0.1).collect();

        let state_trajectory: Vec<PhysicalState> = (0..n_steps)
            .map(|i| {
                use nalgebra::DVector;
                let concs: Vec<f64> = (0..n_species)
                    .map(|s| i as f64 * 0.001 + s as f64 * 0.0001)
                    .collect();
                PhysicalState::new(
                    PhysicalQuantity::Concentration,
                    PhysicalData::Vector(DVector::from_vec(concs)),
                )
            })
            .collect();

        let final_state = state_trajectory.last().unwrap().clone();
        SimulationResult::new(time_points, state_trajectory, final_state)
    }

    // -------------------------------------------------------------------------
    // compute_sample_indices tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_indices_none_returns_all() {
        // None → every index from 0 to total-1
        let indices = compute_sample_indices(10, None);
        assert_eq!(indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_sample_indices_n_equals_total() {
        // n = total → identical behaviour to None
        let indices = compute_sample_indices(5, Some(5));
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
        assert_eq!(*indices.last().unwrap(), 4);
    }

    #[test]
    fn test_sample_indices_n_greater_than_total() {
        // n > total → export everything (no interpolation, no error)
        let indices = compute_sample_indices(5, Some(100));
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_sample_indices_n_one() {
        // n = 1 → only the first point
        let indices = compute_sample_indices(100, Some(1));
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_sample_indices_uniform_and_last_included() {
        // n = 5 over 100 points → 5 uniform indices, last = 99
        let indices = compute_sample_indices(100, Some(5));
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
        assert_eq!(*indices.last().unwrap(), 99); // ← always the last point
    }

    #[test]
    fn test_sample_indices_stride_correctness() {
        // total=10, n=3 → indices [0, 4, 9]
        // stride = (10-1) / (3-1) = 4
        let indices = compute_sample_indices(10, Some(3));
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 4); // (1 * 9) / 2 = 4
        assert_eq!(indices[2], 9); // always the last point
    }

    // -------------------------------------------------------------------------
    // export_single tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_export_single_creates_file() {
        let result = make_single_result(10);
        let exporter = CsvExporter::default();
        let path = "/tmp/test_single_creates.csv";

        exporter.export_single(&result, None, path).unwrap();

        assert!(std::path::Path::new(path).exists());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_single_header() {
        let result = make_single_result(5);
        let exporter = CsvExporter::default();
        let path = "/tmp/test_single_header.csv";

        exporter.export_single(&result, None, path).unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        let first_line = content.lines().next().unwrap();

        assert_eq!(first_line, "time (s);c_outlet (mol/L)");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_single_row_count_all_points() {
        // No downsampling: 1 header line + n_steps data lines
        let n_steps = 20;
        let result = make_single_result(n_steps);
        let exporter = CsvExporter::default();
        let path = "/tmp/test_single_rows_all.csv";

        exporter.export_single(&result, None, path).unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        let line_count = content.lines().count();
        assert_eq!(line_count, n_steps + 1); // +1 for the header

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_single_row_count_subsampled() {
        // With n_points=5: 1 header + 5 data lines
        let result = make_single_result(100);
        let exporter = CsvExporter::default();
        let path = "/tmp/test_single_rows_sub.csv";

        exporter.export_single(&result, Some(5), path).unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        let line_count = content.lines().count();
        assert_eq!(line_count, 6); // 1 header + 5 data

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_single_values_correctness() {
        // Check that numeric values match the SimulationResult
        // time_points = [0.0, 0.1, 0.2], concentrations = [0.000, 0.001, 0.002]
        let result = make_single_result(3);
        let exporter = CsvExporter::default();
        let path = "/tmp/test_single_values.csv";

        exporter.export_single(&result, None, path).unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Row 1 (index 0, t=0, c=0)
        assert!(lines[1].starts_with("0.000000e0;0.000000e0"));

        // Row 2 (index 1, t=0.1, c=0.001)
        assert!(lines[2].contains("1.000000e-1")); // t
        assert!(lines[2].contains("1.000000e-3")); // c

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_single_empty_result_error() {
        // An empty result must return CsvError::EmptyResult
        let result = SimulationResult::new(vec![], vec![], PhysicalState::empty());
        let exporter = CsvExporter::default();

        let err = exporter
            .export_single(&result, None, "/tmp/unused.csv")
            .unwrap_err();

        assert!(matches!(err, CsvError::EmptyResult));
    }

    #[test]
    fn test_export_single_invalid_path_error() {
        // Non-existent directory → CsvError::Io
        let result = make_single_result(5);
        let exporter = CsvExporter::default();

        let err = exporter
            .export_single(&result, None, "/nonexistent_dir/file.csv")
            .unwrap_err();

        assert!(matches!(err, CsvError::Io(_)));
    }

    // -------------------------------------------------------------------------
    // export_multi tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_export_multi_header() {
        let result = make_multi_result(5, 2);
        let exporter = CsvExporter::default();
        let path = "/tmp/test_multi_header.csv";

        exporter
            .export_multi(&result, None, &["Ascorbic", "Erythorbic"], path)
            .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        let first_line = content.lines().next().unwrap();

        assert_eq!(
            first_line,
            "time (s);c_total (mol/L);Ascorbic (mol/L);Erythorbic (mol/L)"
        );

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_multi_envelope_equals_sum() {
        // c_total must equal the exact sum of all species concentrations at each row
        let n_species = 3;
        let result = make_multi_result(10, n_species);
        let exporter = CsvExporter::default();
        let path = "/tmp/test_multi_envelope.csv";
        let names = vec!["A", "B", "C"];

        exporter.export_multi(&result, None, &names, path).unwrap();

        let content = std::fs::read_to_string(path).unwrap();

        // Check every data row (skip the header)
        for line in content.lines().skip(1) {
            let cols: Vec<f64> = line
                .split(';')
                .skip(1) // skip 'time'
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect();

            // cols[0] = c_total, cols[1..] = individual species
            let c_total = cols[0];
            let sum: f64 = cols[1..].iter().sum();

            assert!(
                (c_total - sum).abs() < 1e-12,
                "c_total {c_total} != sum {sum}"
            );
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_multi_species_count_mismatch() {
        // 2 species in result, 3 names provided → SpeciesCountMismatch
        let result = make_multi_result(5, 2);
        let exporter = CsvExporter::default();

        let err = exporter
            .export_multi(&result, None, &["A", "B", "C"], "/tmp/unused.csv")
            .unwrap_err();

        assert!(matches!(
            err,
            CsvError::SpeciesCountMismatch { expected: 2, got: 3 }
        ));
    }

    #[test]
    fn test_export_multi_subsampled() {
        let result = make_multi_result(50, 2);
        let exporter = CsvExporter::default();
        let path = "/tmp/test_multi_sub.csv";

        exporter
            .export_multi(&result, Some(10), &["A", "B"], path)
            .unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content.lines().count(), 11); // 1 header + 10 data rows

        std::fs::remove_file(path).ok();
    }

    // -------------------------------------------------------------------------
    // CsvConfig tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_custom_separator() {
        // Comma separator (Anglo-Saxon style)
        let result = make_single_result(3);
        let config = CsvConfig {
            separator: ',',
            precision: 6,
        };
        let exporter = CsvExporter::new(config);
        let path = "/tmp/test_custom_sep.csv";

        exporter.export_single(&result, None, path).unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        let first_line = content.lines().next().unwrap();
        assert_eq!(first_line, "time (s),c_outlet (mol/L)");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_csvconfig_default() {
        let config = CsvConfig::default();
        assert_eq!(config.separator, ';');
        assert_eq!(config.precision, 6);
    }

    // -------------------------------------------------------------------------
    // CsvError tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_csv_error_display_empty() {
        let err = CsvError::EmptyResult;
        let msg = err.to_string();
        assert!(msg.contains("no time points"));
    }

    #[test]
    fn test_csv_error_display_mismatch() {
        let err = CsvError::SpeciesCountMismatch {
            expected: 2,
            got: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("expected 2") && msg.contains("got 5"));
    }

    #[test]
    fn test_csv_error_source_io() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = CsvError::Io(io_err);
        // Source must be the underlying io error
        assert!(err.source().is_some());
    }

    #[test]
    fn test_csv_error_source_empty() {
        use std::error::Error;
        let err = CsvError::EmptyResult;
        // EmptyResult has no underlying source
        assert!(err.source().is_none());
    }
}
