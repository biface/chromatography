//! Visualization module for chromatography simulation results
//!
//! This module provides tools to visualize simulation results using the `plotters` library.
//! It focuses on generating high-quality static plots (PNG/SVG) in v0.1.0.
//!
//! # Modules
//!
//! - `static_plots`: Static image generation (PNG, SVG)
//! - `dynamic_plots`: (Future v0.2.0+) Animations and interactive visualizations
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use chrom_rs::visualization::{plot_chromatogram, PlotConfig};
//!
//! // After running simulation
//! let result = solver.solve(&scenario, &config)?;
//!
//! // Generate chromatogram
//! plot_chromatogram(&result, "output.png", None)?;
//! ```

pub mod static_plots;

pub use static_plots::{
    plot_chromatogram,
    plot_chromatogram_multi,
    plot_result,
    plot_result_multi,
    PlotConfig,
};