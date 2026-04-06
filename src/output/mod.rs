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
//! │   └── steady.rs
//! └── export/             ← Data export
//!     ├── mod.rs
//!     └── csv.rs
//! ```
//!
//! # Quick Start
//!
//! ## Visualization
//!
//! ```rust
//! use chrom_rs::output::visualization::{plot_chromatogram, PlotConfig};
//! use chrom_rs::solver::SimulationResult;
//! use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
//!
//! # let state = PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Scalar(0.0));
//! # let result = SimulationResult::new(vec![0.0, 1.0], vec![state.clone(), state.clone()], state);
//! // Generate PNG plot
//! let _ = plot_chromatogram(&result, 100, "/tmp/output.png", None);
//! ```
//!
//! ## CSV Export
//!
//! ```rust
//! use chrom_rs::output::export::{CsvExporter, CsvConfig, Exporter};
//! use chrom_rs::solver::SimulationResult;
//! use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
//!
//! # let state = PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Scalar(0.0));
//! # let result = SimulationResult::new(vec![0.0, 1.0], vec![state.clone(), state.clone()], state);
//! let exporter = CsvExporter::default();
//!
//! // Export to CSV
//! let _ = exporter.export_single(&result, None, "/tmp/data.csv");
//! ```
//!
//! # Design Philosophy
//!
//! The output module separates concerns:
//! - **Visualization**: For human interpretation (plots, graphs)
//! - **Export**: For programmatic analysis (CSV, JSON, HDF5)
//!
//! Both submodules accept simple `&[f64]` slices for maximum flexibility.
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

pub mod export;
pub mod visualization;

// Re-export commonly used items for convenience
pub use visualization::{
    PlotConfig, plot_chromatogram, plot_chromatogram_multi, plot_chromatograms_comparison,
    plot_profile_evolution, plot_steady_state, plot_steady_state_comparison,
};

pub use export::{CsvConfig, CsvError, CsvExporter};
