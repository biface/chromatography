//! CSV export functionality for chromatography simulation results
//!
//! This module provides tools to export simulation data to CSV (Comma-Separated Values)
//! format, which is compatible with Excel, Python pandas, MATLAB, and most data analysis tools.
//!
//! # Features
//!
//! - **Simple interface**: Export with `&[f64]` slices
//! - **Metadata support**: Optional headers with simulation parameters
//! - **Customizable**: Delimiter, precision, format options
//! - **Multi-species**: Export multiple concentration columns
//! - **Validation**: Checks for NaN, empty data, mismatched lengths
//!
//! # Quick Examples
//!
//! ## Minimal Export
//!
//! ```rust,ignore
//! use chrom_rs::output::export::export_chromatogram_csv;
//!
//! let time = vec![0.0, 1.0, 2.0, 3.0];
//! let conc = vec![0.0, 0.5, 1.0, 0.5];
//!
//! export_chromatogram_csv(&time, &conc, "data.csv", None)?;
//! ```
//!
//! **Output** (`data.csv`):
//! ```csv
//! Time (s),Concentration (mol/L)
//! 0.0,0.0
//! 1.0,0.5
//! 2.0,1.0
//! 3.0,0.5
//! ```
//!
//! ## With Metadata
//!
//! ```rust,ignore
//! use chrom_rs::output::export::{export_chromatogram_csv, CsvConfig, CsvMetadata};
//!
//! let metadata = CsvMetadata {
//!     model_name: Some("LangmuirSingleSimple".to_string()),
//!     solver_name: Some("Forward Euler".to_string()),
//!     total_time: Some(200.0),
//!     time_steps: Some(2000),
//!     ..Default::default()
//! };
//!
//! let mut config = CsvConfig::default();
//! config.include_metadata = true;
//! config.metadata = Some(metadata);
//!
//! export_chromatogram_csv(&time, &conc, "data.csv", Some(&config))?;
//! ```
//!
//! **Output** (`data.csv`):
//! ```csv
//! # Chromatography Simulation Data
//! # Generated: 2026-02-11T15:30:00Z
//! # Model: LangmuirSingleSimple
//! # Solver: Forward Euler
//! # Total Time: 200 s
//! # Time Steps: 2000
//! #
//! Time (s),Concentration (mol/L)
//! 0.0,0.0
//! 1.0,0.5
//! ...
//! ```
//!
//! ## Multi-Species
//!
//! ```rust,ignore
//! use chrom_rs::output::export::export_chromatogram_multi_csv;
//!
//! let time = vec![0.0, 50.0, 100.0, 150.0];
//! let acid_a = vec![0.0, 1.0, 0.5, 0.0];
//! let acid_b = vec![0.0, 0.0, 0.5, 1.0];
//!
//! export_chromatogram_multi_csv(
//!     &time,
//!     &[acid_a, acid_b],
//!     &["Ascorbic", "Erythorbic"],
//!     "acids.csv",
//!     None,
//! )?;
//! ```
//!
//! **Output** (`acids.csv`):
//! ```csv
//! Time (s),Ascorbic (mol/L),Erythorbic (mol/L)
//! 0.0,0.0,0.0
//! 50.0,1.0,0.0
//! 100.0,0.5,0.5
//! 150.0,0.0,1.0
//! ```

use std::error::Error;
use std::fs::File;
use std::io::Write;
// =============================================================================
// Configuration Structures
// =============================================================================

/// Configuration for CSV export
///
/// # Fields
///
/// - `delimiter`: Column separator (default: ',')
/// - `decimal_separator`: Decimal point character (default: '.')
/// - `precision`: Number of decimal places (default: 6)
/// - `include_metadata`: Add header comments with simulation info
/// - `metadata`: Simulation metadata to include
/// - `time_header`: Custom header for time column
/// - `concentration_header`: Custom header for concentration column
///
/// # Example
///
/// ```rust,ignore
/// let config = CsvConfig {
///     delimiter: ';',        // European CSV
///     precision: 10,         // High precision
///     include_metadata: true,
///     ..Default::default()
/// };
/// ```
#[derive(Clone)]
pub struct CsvConfig {
    /// Column delimiter (default: ',')
    pub delimiter: char,

    /// Decimal separator (default: '.')
    pub decimal_separator: char,

    /// Number of decimal places for floating-point values (default: 6)
    pub precision: usize,

    /// Include metadata header comments (default: false)
    pub include_metadata: bool,

    /// Metadata to include in header
    pub metadata: Option<CsvMetadata>,

    /// Custom header for time column (default: "Time (s)")
    pub time_header: String,

    /// Custom header for concentration column (default: "Concentration (mol/L)")
    pub concentration_header: String,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            decimal_separator: '.',
            precision: 6,
            include_metadata: false,
            metadata: None,
            time_header: "Time (s)".to_string(),
            concentration_header: "Concentration (mol/L)".to_string(),
        }
    }
}

impl CsvConfig {
    /// Create config with European CSV format (semicolon, comma for decimal)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = CsvConfig::european();
    /// // delimiter: ';'
    /// // decimal_separator: ','
    /// ```
    pub fn european() -> Self {
        Self {
            delimiter: ';',
            decimal_separator: ',',
            ..Default::default()
        }
    }

    /// Create config with high precision (12 decimal places)
    pub fn high_precision() -> Self {
        Self {
            precision: 12,
            ..Default::default()
        }
    }

    /// Builder pattern: set delimiter
    pub fn delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Builder pattern: set precision
    pub fn precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    /// Builder pattern: enable metadata
    pub fn with_metadata(mut self, metadata: CsvMetadata) -> Self {
        self.include_metadata = true;
        self.metadata = Some(metadata);
        self
    }
}

/// Metadata for CSV header comments
///
/// All fields are optional. Only non-None fields will be included in the CSV header.
///
/// # Example
///
/// ```rust,ignore
/// let metadata = CsvMetadata {
///     model_name: Some("LangmuirSingleSimple".to_string()),
///     solver_name: Some("Forward Euler".to_string()),
///     total_time: Some(200.0),
///     time_steps: Some(2000),
///     lambda: Some(1.2),
///     langmuir_k: Some(0.4),
///     porosity: Some(0.4),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Default)]
pub struct CsvMetadata {
    /// Model name (e.g., "LangmuirSingleSimple")
    pub model_name: Option<String>,

    /// Solver name (e.g., "Forward Euler", "RK4")
    pub solver_name: Option<String>,

    /// Total simulation time (seconds)
    pub total_time: Option<f64>,

    /// Number of time steps
    pub time_steps: Option<usize>,

    /// Langmuir lambda parameter
    pub lambda: Option<f64>,

    /// Langmuir K̃ parameter (L/mol)
    pub langmuir_k: Option<f64>,

    /// Porosity εₑ
    pub porosity: Option<f64>,

    /// Velocity u (m/s)
    pub velocity: Option<f64>,

    /// Additional custom parameters
    pub custom: Vec<(String, String)>,
}

impl CsvMetadata {
    /// Create metadata from simulation result
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let metadata = CsvMetadata::from_simulation(
    ///     "LangmuirSingleSimple",
    ///     "Forward Euler",
    ///     200.0,
    ///     2000,
    /// );
    /// ```
    pub fn from_simulation(
        model: &str,
        solver: &str,
        total_time: f64,
        time_steps: usize,
    ) -> Self {
        Self {
            model_name: Some(model.to_string()),
            solver_name: Some(solver.to_string()),
            total_time: Some(total_time),
            time_steps: Some(time_steps),
            ..Default::default()
        }
    }

    /// Add custom parameter
    pub fn add_custom(&mut self, key: String, value: String) {
        self.custom.push((key, value));
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Write metadata header comments to file
fn write_metadata_header(
    file: &mut File,
    metadata: &CsvMetadata,
) -> Result<(), Box<dyn Error>> {
    writeln!(file, "# Chromatography Simulation Data")?;

    // Timestamp (current time)
    let now = chrono::Utc::now();
    writeln!(file, "# Generated: {}", now.to_rfc3339())?;

    // Model and solver
    if let Some(model) = &metadata.model_name {
        writeln!(file, "# Model: {}", model)?;
    }
    if let Some(solver) = &metadata.solver_name {
        writeln!(file, "# Solver: {}", solver)?;
    }

    // Simulation parameters
    if let Some(total_time) = metadata.total_time {
        writeln!(file, "# Total Time: {} s", total_time)?;
    }
    if let Some(time_steps) = metadata.time_steps {
        writeln!(file, "# Time Steps: {}", time_steps)?;
    }

    // Model parameters
    if let Some(lambda) = metadata.lambda {
        writeln!(file, "# Lambda: {}", lambda)?;
    }
    if let Some(k) = metadata.langmuir_k {
        writeln!(file, "# Langmuir K: {} L/mol", k)?;
    }
    if let Some(eps) = metadata.porosity {
        writeln!(file, "# Porosity: {}", eps)?;
    }
    if let Some(u) = metadata.velocity {
        writeln!(file, "# Velocity: {} m/s", u)?;
    }

    // Custom parameters
    for (key, value) in &metadata.custom {
        writeln!(file, "# {}: {}", key, value)?;
    }

    // Separator
    writeln!(file, "#")?;

    Ok(())
}

/// Format number with configured precision and decimal separator
fn format_number(value: f64, config: &CsvConfig) -> String {
    let formatted = format!("{:.prec$}", value, prec = config.precision);

    // Replace decimal separator if needed
    if config.decimal_separator != '.' {
        formatted.replace('.', &config.decimal_separator.to_string())
    } else {
        formatted
    }
}

// =============================================================================
// Export Functions
// =============================================================================

/// Export single-species chromatogram to CSV
///
/// Writes time and concentration data to a CSV file with optional metadata header.
///
/// # Arguments
///
/// * `time_points` - Time values (seconds)
/// * `concentrations` - Concentration values (mol/L)
/// * `output_path` - Output file path
/// * `config` - Optional CSV configuration (uses default if None)
///
/// # Returns
///
/// `Ok(())` if successful, `Err` with detailed message otherwise
///
/// # Errors
///
/// - Empty data
/// - Mismatched lengths
/// - NaN or Inf values
/// - File creation errors
///
/// # Example
///
/// ```rust,ignore
/// export_chromatogram_csv(&time, &conc, "tfa.csv", None)?;
/// ```
pub fn export_chromatogram_csv(
    time_serie: &[f64],
    concentration_serie: &[f64],
    output_path: &str,
    configuration: Option<&CsvConfig>,
) -> Result<(), Box<dyn Error>> {

    // ============================= Validation =============================

    if time_serie.is_empty() || concentration_serie.is_empty() {
        return Err("Empty data: time and concentration series must not be empty".into());
    }

    if time_serie.len() != concentration_serie.len() {
        return Err(format!(
            "Data length mismatch: {} points versus {} concentrations",
            time_serie.len(),
            concentration_serie.len()).into()
        )
    }

    if time_serie.iter().any(|t| !t.is_finite()) {
        return Err("Invalid data: NaN of Inf detected in time series".into());
    }

    if concentration_serie.iter().any(|t| !t.is_finite()) {
        return Err("Invalid data: NaN of Inf detected in concentration series".into());
    }

    // ============================= Configuration ==========================

    let binding = CsvConfig::default();
    let configuration = configuration.unwrap_or(&binding);

    // ============================= Open File ==============================

    let mut file = File::create(output_path)?;

    // ============================= Write Metadata =========================

    if configuration.include_metadata {
        if let Some(metadata) = &configuration.metadata {
            write_metadata_header(&mut file, metadata)?;
        }
    }

    // ============================= Write Header ===========================

    writeln!(
        file,
        "{}{}{}",
        configuration.time_header,
        configuration.delimiter,
        configuration.concentration_header
    )?;

    // ============================= Write Data =============================

    for (time, concentration) in time_serie.iter().zip(concentration_serie.iter()) {
        let time_str = format_number(*time, configuration);
        let concentration_str = format_number(*concentration, configuration);

        writeln!(
            file,
            "{}{}{}",
            time_str,
            configuration.delimiter,
            concentration_str
        )?;
    }

    Ok(())
}

/// Export multi-species chromatogram to CSV
///
/// Writes time and multiple concentration columns to CSV.
///
/// # Arguments
///
/// * `time_points` - Time values (seconds)
/// * `concentrations_matrix` - Vector of concentration vectors (one per species)
/// * `species_names` - Names for each species (for column headers)
/// * `output_path` - Output file path
/// * `config` - Optional CSV configuration
///
/// # Returns
///
/// `Ok(())` if successful, `Err` otherwise
///
/// # Example
///
/// ```rust,ignore
/// export_chromatogram_multi_csv(
///     &time,
///     &[acid_a, acid_b],
///     &["Ascorbic", "Erythorbic"],
///     "acids.csv",
///     None,
/// )?;
/// ```
pub fn export_chromatogram_multi_csv(
    time_serie: &[f64],
    concentration_matrix: &[Vec<f64>],
    specie_serie: &[&str],
    output_path: &str,
    configuration: Option<&CsvConfig>,
) -> Result<(), Box<dyn Error>> {

    // ============================= Validation =============================

    if time_serie.is_empty() || concentration_matrix.is_empty() {
        return Err("Empty data: time and concentration matrix must not be empty".into());
    }

    if time_serie.iter().any(|t| !t.is_finite()) {
        return Err("Invalid data: NaN of Inf detected in time series".into());
    }

    if concentration_matrix.len() != specie_serie.len() {
        return Err(format!(
           "Data length mismatch: {} number of concentration series versus {} number of species",
            concentration_matrix.len(),
            specie_serie.len()
        ).into());
    }

    for (i, series) in concentration_matrix.iter().enumerate() {
        if series.len() != time_serie.len() {
            return Err(format!(
                "Species [{}] length mismatch: {} concentration vs {} time series",
                specie_serie[i],
                series.len(),
                time_serie.len()
            ).into());
        }

        if series.iter().any(|t| !t.is_finite()) {
            return Err(format!(
                "Invalid data: NaN or Inf detected in specie {}",
                specie_serie[i]
            ).into())
        }
    }

    // ============================= Configuration ==========================

    let binding = CsvConfig::default();
    let configuration = configuration.unwrap_or(&binding);

    // ============================= Open File ==============================

    let mut file = File::create(output_path)?;

    // ============================= Write Metadata =========================

    if configuration.include_metadata {
        if let Some(metadata) = &configuration.metadata {
            write_metadata_header(&mut file, metadata)?;
        }
    }

    // ============================= Write Metadata =========================

    if configuration.include_metadata {
        if let Some(metadata) = &configuration.metadata {
            write_metadata_header(&mut file, metadata)?;
        }
    }

    // ============================= Write Data =============================

    for i in 0..time_serie.len() {
        // Time
        write!(file, "{}", format_number(time_serie[i], configuration))?;

        // Each species concentration
        for concs in concentration_matrix {
            write!(
                file,
                "{}{}",
                configuration.delimiter,
                format_number(concs[i], configuration)
            )?;
        }
        writeln!(file)?;
    }

    Ok(())
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
    use crate::solver::{Scenario, SolverConfiguration, DomainBoundaries};
    use crate::solver::{Solver, EulerSolver};
    use nalgebra::DVector;
    use std::fs;

    // ====== Mock models for CSV file testing ======

    struct TestModel {
        amplitude: f64,
    }

    impl PhysicalModel for TestModel {
        fn points(&self) -> usize { 1 }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            let conc = state.get(PhysicalQuantity::Concentration).unwrap();
            let c = match conc {
                PhysicalData::Scalar(v) => *v,
                _ => panic!("Expected scalar"),
            };

            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::Scalar(-0.1 * c),
            )
        }

        fn setup_initial_state(&self) -> PhysicalState {
            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::Scalar(self.amplitude),
            )
        }

        fn name(&self) -> &str { "TestModel" }
    }

    struct TestModelMulti {
        n_species: usize,
    }

    impl PhysicalModel for TestModelMulti {
        fn points(&self) -> usize { 1 }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            let conc = state.get(PhysicalQuantity::Concentration).unwrap();
            let c_vec = match conc {
                PhysicalData::Vector(v) => v.clone(),
                _ => panic!("Expected vector"),
            };

            let rates: Vec<f64> = (0..self.n_species)
                .map(|i| -0.1 * (i + 1) as f64 * c_vec[i])
                .collect();

            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::Vector(DVector::from_vec(rates)),
            )
        }

        fn setup_initial_state(&self) -> PhysicalState {
            let initial: Vec<f64> = (0..self.n_species)
                .map(|i| 10.0 * (i + 1) as f64)
                .collect();

            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::Vector(DVector::from_vec(initial)),
            )
        }

        fn name(&self) -> &str { "TestModelMulti" }
    }

    // Basic CSV export tests


}