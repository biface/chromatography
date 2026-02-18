//! Output module for simulation results
//!
//! This module provides tools to output simulation results in various formats:
//! - **Visualization**: PNG/SVG plots using plotters
//! - **Export**: CSV/JSON data export for external analysis
//!
//! # Architecture
//!
//! ```text
//! output/
//! ├── mod.rs              ← This file
//! ├── visualization/      ← Plots and graphics
//! │   ├── mod.rs
//! │   └── static_plots.rs
//! └── export/             ← Data export
//!     ├── mod.rs
//!     └── csv.rs
//! ```
//!
//! # Quick Start
//!
//! ## Visualization
//!
//! ```rust,ignore
//! use chrom_rs::output::visualization::{plot_chromatogram, PlotConfig};
//!
//! // Generate PNG plot
//! plot_chromatogram(&time, &conc, "output.png", None)?;
//! ```
//!
//! ## CSV Export
//!
//! ```rust,ignore
//! use chrom_rs::output::export::{export_chromatogram_csv, CsvConfig};
//!
//! // Export to CSV
//! export_chromatogram_csv(&time, &conc, "data.csv", None)?;
//! ```
//!
//! # Design Philosophy
//!
//! The output module separates concerns:
//! - **Visualization**: For human interpretation (plots, graphs)
//! - **Export**: For programmatic analysis (CSV, JSON, HDF5)
//!
//! Both sub-modules accept simple `&[f64]` slices for maximum flexibility.
//!
//! # Version History
//!
//! - **v0.1.0**: Static plots (PNG/SVG) and CSV export
//! - **v0.2.0+**: Animations, HDF5, JSON export (planned)
//!
//! # Examples
//!
//! See `examples/` directory for complete workflows:
//! - `examples/output_demo.rs` - Complete output pipeline
//! - `examples/tfa_export.rs` - TFA chromatogram export

pub mod visualization;
pub mod export;

// Re-export commonly used items for convenience
pub use visualization::{
    plot_chromatogram,
    plot_chromatogram_multi,
    plot_result,
    plot_result_multi,
    PlotConfig,
};

pub use export::{
    export_chromatogram_csv,
    export_chromatogram_multi_csv,
    CsvConfig,
};