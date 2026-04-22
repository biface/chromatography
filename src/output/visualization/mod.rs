//! Visualization module for chromatography simulation results
//!
//! This module provides tools to visualize simulation results using the `plotters` library.
//!
//! # Organization
//!
//! - **config**: Shared plot configuration (`PlotConfig`)
//! - **chromatogram**: Temporal plots (outlet concentration vs time)
//! - **steady**: Spatial plots (concentration profile vs position)
//!
//! # Quick Start
//!
//! ## Chromatogram (Temporal - Column Outlet vs Time)
//!
//! ```rust
//! use chrom_rs::output::visualization::{plot_chromatogram, PlotConfig};
//! use chrom_rs::solver::SimulationResult;
//! use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
//!
//! # let state = PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Scalar(0.0));
//! # let result = SimulationResult::new(vec![0.0, 1.0], vec![state.clone(), state.clone()], state);
//! // Plot chromatogram with default config
//! let _ = plot_chromatogram(&result, 100, "/tmp/chromato.png", None);
//!
//! // Or with custom config
//! let mut config = PlotConfig::chromatogram("TFA Elution");
//! let _ = plot_chromatogram(&result, 100, "/tmp/tfa.png", Some(&config));
//! ```
//!
//! ## Steady-State (Spatial Profile at Final Time)
//!
//! ```rust
//! use chrom_rs::output::visualization::plot_steady_state;
//! use chrom_rs::solver::SimulationResult;
//! use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
//!
//! # let state = PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Scalar(0.0));
//! # let result = SimulationResult::new(vec![0.0, 1.0], vec![state.clone(), state.clone()], state);
//! let _ = plot_steady_state(&result, 0.25, "/tmp/profile.png", None);
//! ```
//!
//! # When to Use Which Module
//!
//! | Use Case | Module | Function |
//! |----------|--------|----------|
//! | Chromatogram (outlet vs time) | `chromatogram` | `plot_chromatogram` |
//! | Multi-species chromatogram | `chromatogram` | `plot_chromatogram_multi` |
//! | Compare chromatograms | `chromatogram` | `plot_chromatograms_comparison` |
//! | Steady-state spatial profile | `steady` | `plot_steady_state` |
//! | Compare spatial profiles | `steady` | `plot_steady_state_comparison` |
//! | Profile evolution over time | `steady` | `plot_profile_evolution` |

pub mod chromatogram;
pub mod config;
/// Spatial concentration profile plots (steady-state and evolution over time).
pub mod steady;

pub use config::PlotConfig;

pub use steady::{
    plot_profile_evolution, plot_steady_state, plot_steady_state_comparison,
    plot_steady_state_multi,
};

pub use chromatogram::{
    plot_chromatogram, plot_chromatogram_envelope, plot_chromatogram_multi,
    plot_chromatograms_comparison,
};
