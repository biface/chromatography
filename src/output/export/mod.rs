//! Export sub-module for saving simulation data
//!
//! This module provides tools to export simulation results in various formats
//! for external analysis and archival.
//!
//! # Supported Formats
//!
//! - **CSV**: Comma-separated values (human-readable, Excel-compatible)
//! - **JSON**: (Future v0.2.0+) Structured data with metadata
//! - **HDF5**: (Future v0.2.0+) Large-scale scientific data
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use chrom_rs::output::export::{export_chromatogram_csv, CsvConfig};
//!
//! // Simple export with defaults
//! export_chromatogram_csv(&time, &conc, "data.csv", None)?;
//!
//! // Custom configuration
//! let config = CsvConfig {
//!     delimiter: ';',
//!     include_metadata: true,
//!     ..Default::default()
//! };
//! export_chromatogram_csv(&time, &conc, "data.csv", Some(&config))?;
//! ```
//!
//! # Use Cases
//!
//! - **Python analysis**: Import CSV in pandas/numpy
//! - **Excel**: Direct import for plotting/analysis
//! - **MATLAB**: Load with `readtable()`
//! - **Archival**: Long-term storage in standard format
//! - **Sharing**: Simple text format for collaboration
//!
//! # Design Philosophy
//!
//! CSV export prioritizes:
//! 1. **Simplicity**: Plain text, human-readable
//! 2. **Compatibility**: Works with all analysis tools
//! 3. **Metadata**: Optional headers with simulation parameters
//! 4. **Precision**: Configurable decimal places
//!
//! # Future Extensions (v0.2.0+)
//!
//! - `json`: Structured export with full metadata
//! - `hdf5`: High-performance binary format
//! - `parquet`: Columnar format for big data

pub mod csv;

// Re-export main functions
pub use csv::{
    export_chromatogram_csv,
    export_chromatogram_multi_csv,
    CsvConfig,
    CsvMetadata,
};