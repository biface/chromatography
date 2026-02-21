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
//! ```rust,ignore
//! use chrom_rs::output::visualization::{plot_chromatogram, PlotConfig};
//!
//! let result = solver.solve(&scenario, &config)?;
//!
//! // Plot chromatogram with default config
//! plot_chromatogram(&result, 100, "chromato.png", None)?;
//!
//! // Or with custom config
//! let mut config = PlotConfig::chromatogram();
//! config.title = "TFA Elution".to_string();
//! plot_chromatogram(&result, 100, "tfa.png", Some(&config))?;
//! ```
//!
//! ## Steady-State (Spatial Profile at Final Time)
//!
//! ```rust,ignore
//! use chrom_rs::output::visualization::plot_steady_state;
//!
//! plot_steady_state(&result, 0.25, "profile.png", None)?;
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

pub mod config;
pub mod chromatogram;
pub mod steady;

pub use config::PlotConfig;

pub use steady::{plot_steady_state, plot_steady_state_comparison, plot_profile_evolution};

pub use chromatogram::plot_chromatogram;