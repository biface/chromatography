//! Chromatogram plotting for temporal simulations
//!
//! This module provides plotting functions for time-series data,
//! specifically chromatograms showing concentration at column outlet vs time.
//!
//! # Key Difference from static_plots
//!
//! This module correctly plots **chromatograms** (C_outlet vs time), not spatial profiles.
//! It extracts the outlet concentration (last spatial point) and plots it over time.
//!
//! # Usage
//!
//! ```rust,ignore
//! use chrom_rs::output::visualization::plot_chromatogram;
//!
//! // After temporal simulation
//! let result = solver.solve(&scenario, &config)?;
//!
//! // Plot chromatogram (outlet concentration vs time)
//! plot_chromatogram(&result, 100, "chromato.png", None)?;
//! ```

use plotters::prelude::*;
use std::error::Error;

use crate::solver::SimulationResult;
use crate::physics::{PhysicalQuantity, PhysicalData};
use super::config::{PlotConfig, NO_TITLE};

// =================================================================================================
// Helper Functions - Extract Outlet Concentration
// =================================================================================================

/// Extract outlet concentration over time from SimulationResult
///
/// For chromatography, we want C_outlet(t) = concentration at column exit vs time.
/// This extracts the last spatial point from each time step.
///
/// # Arguments
///
/// * `result` - Simulation result with state trajectory
/// * `n_points` - Number of spatial points (to extract last point)
///
/// # Returns
///
/// Vector of outlet concentrations, one per time point
fn extract_single_species_outlet(result: &SimulationResult, n_points:usize) -> Vec<f64> {
    result.state_trajectory
        .iter()
        .map(|state| {
            match state.get(PhysicalQuantity::Concentration) {
                Some(physical_data) => {
                    match physical_data {
                        PhysicalData::Scalar(concentration) => *concentration,
                        PhysicalData::Vector(profile) => {
                            // Extract outlet (last spatial point)
                            profile[n_points - 1]
                        }
                        // Fallback: we assume there is no data
                        _ => 0.0,
                    }
                }
                None => 0.0,
            }
        })
        .collect()
}

// =================================================================================================
// Core Plotting Functions
// =================================================================================================

/// Plot a single-species chromatogram
///
/// Creates a plot showing outlet concentration vs time.
///
/// # Arguments
///
/// * `result` - Simulation result containing trajectory
/// * `n_points` - Number of spatial points (to extract outlet = last point)
/// * `output_path` - Path to save the plot (PNG or SVG)
/// * `config` - Optional plot configuration
///
/// # Example
///
/// ```rust,ignore
/// let result = solver.solve(&scenario, &config)?;
/// plot_chromatogram(&result, 100, "tfa.png", None)?;
/// ```

pub fn plot_chromatogram(
    result: &SimulationResult,
    n_points: usize,
    output_path: &str,
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    let outlet = extract_single_species_outlet(result, n_points);
    let time_points = &result.time_points;

    // Create default config if needed (avoid temporary value)
    let default_config = PlotConfig::chromatogram(NO_TITLE);
    let config = config.unwrap_or(&default_config);

    // Determine plot range
    let max_time = time_points.last().copied().unwrap_or(1.0);
    let max_conc = outlet
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-10);

    // Determine backend based on file extension and plot
    let ext = std::path::Path::new(output_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");

    match ext {
        "svg" => {
            let backend = SVGBackend::new(output_path, (config.width, config.height));
            plot_chromatogram_impl(backend, time_points, &outlet, config, max_time, max_conc)
        }
        _ => {
            let backend = BitMapBackend::new(output_path, (config.width, config.height));
            plot_chromatogram_impl(backend, time_points, &outlet, config, max_time, max_conc)
        }
    }
}

/// Implementation for chromatogram plotting with concrete backend
fn plot_chromatogram_impl<DB: DrawingBackend>(
    backend: DB,
    time_points: &[f64],
    outlet: &[f64],
    config: &PlotConfig,
    max_time: f64,
    max_conc: f64,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let root = backend.into_drawing_area();
    root.fill(&config.background)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 40).into_font())
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_time, 0.0..(max_conc * 1.1))?;

    if config.show_grid {
        chart.configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .x_label_formatter(&|x| format!("{:.0}", x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    }

    chart.draw_series(LineSeries::new(
        time_points.iter().zip(outlet.iter()).map(|(t, c)| (*t, *c)),
        ShapeStyle::from(&config.line_color).stroke_width(config.line_width),
    ))?
        .label("Outlet Concentration")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &config.line_color)
        });

    chart
        .configure_series_labels()
        .background_style(&config.background.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

/// Plot multiple chromatograms for comparison
///
/// Overlays multiple chromatograms on the same axes.
/// Useful for comparing different injections, solvers, or conditions.
///
/// # Arguments
///
/// * `datasets` - Vec of (label, SimulationResult, n_points)
/// * `output_path` - Path to save the plot
/// * `config` - Optional plot configuration
///
/// # Example
///
/// ```rust,ignore
/// let datasets = vec![
///     ("Euler", &result_euler, 100),
///     ("RK4", &result_rk4, 100),
/// ];
/// plot_chromatograms_comparison(datasets, "comparison.png", None)?;
/// ```
pub fn plot_chromatograms_comparison(
    datasets: Vec<(&str, &SimulationResult, usize)>,
    output_path: &str,
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    if datasets.is_empty() {
        return Err("No datasets provided".into());
    }

    // Create default config if needed (avoid temporary value)
    let default_config = PlotConfig::chromatogram(NO_TITLE);
    let config = config.unwrap_or(&default_config);

    // Extract all outlets
    let mut all_data: Vec<(&str, &[f64], Vec<f64>)> = Vec::new();
    for (label, result, n_points) in &datasets {
        let outlet = extract_single_species_outlet(result, *n_points);
        all_data.push((*label, &result.time_points, outlet));
    }

    // Determine plot range
    let max_time = all_data
        .iter()
        .map(|(_, times, _)| times.last().copied().unwrap_or(0.0))
        .fold(0.0, f64::max);

    let max_conc = all_data
        .iter()
        .flat_map(|(_, _, outlet)| outlet.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-10);

    // Determine backend
    // Determine backend and plot
    let ext = std::path::Path::new(output_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");

    match ext {
        "svg" => {
            let backend = SVGBackend::new(output_path, (config.width, config.height));
            plot_comparison_impl(backend, &all_data, config, max_time, max_conc)
        }
        _ => {
            let backend = BitMapBackend::new(output_path, (config.width, config.height));
            plot_comparison_impl(backend, &all_data, config, max_time, max_conc)
        }
    }
}

/// Implementation for comparison plotting with concrete backend
fn plot_comparison_impl<DB: DrawingBackend>(
    backend: DB,
    datasets: &[(&str, &[f64], Vec<f64>)],
    config: &PlotConfig,
    max_time: f64,
    max_conc: f64,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let root = backend.into_drawing_area();
    root.fill(&config.background)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 40).into_font())
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_time, 0.0..(max_conc * 1.1))?;

    if config.show_grid {
        chart.configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .x_label_formatter(&|x| format!("{:.0}", x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    }

    // Colors for different series
    let colors = vec![RED, BLUE, GREEN, MAGENTA, CYAN];

    for (idx, (label, times, outlet)) in datasets.iter().enumerate() {
        let color = colors[idx % colors.len()];  // Copy instead of reference

        chart
            .draw_series(LineSeries::new(
                times.iter().zip(outlet.iter()).map(|(t, c)| (*t, *c)),
                &color,
            ))?
            .label(*label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
    }

    chart
        .configure_series_labels()
        .background_style(&config.background.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity};
    use crate::solver::{Scenario, SolverConfiguration, DomainBoundaries, EulerSolver, Solver};
    use nalgebra::DVector;

    // Simple test model
    struct TestModel {
        n_points: usize,
    }

    impl PhysicalModel for TestModel {
        fn points(&self) -> usize {
            self.n_points
        }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            // Dummy physics
            let c = state.get(PhysicalQuantity::Concentration).unwrap().as_vector();
            let dc_dt = DVector::from_element(c.len(), -0.01);
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(dc_dt))
        }

        fn setup_initial_state(&self) -> PhysicalState {
            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::Vector(DVector::from_element(self.n_points, 1.0)),
            )
        }

        fn name(&self) -> &str {
            "TestModel"
        }
    }

    #[test]
    fn test_extract_outlet() {
        let model = Box::new(TestModel { n_points: 10 });
        let initial = model.setup_initial_state();
        let scenario = Scenario::new(model, DomainBoundaries::temporal(initial));

        let result = EulerSolver
            .solve(&scenario, &SolverConfiguration::time_evolution(10.0, 100))
            .unwrap();

        let outlet = extract_single_species_outlet(&result, 10);
        assert_eq!(outlet.len(), 101); // 100 steps + initial
        assert!(outlet[0] > 0.9); // Initial value ~1.0
    }

    #[test]
    fn test_plot_chromatogram_png() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");

        let model = Box::new(TestModel { n_points: 10 });
        let initial = model.setup_initial_state();
        let scenario = Scenario::new(model, DomainBoundaries::temporal(initial));

        let result = EulerSolver
            .solve(&scenario, &SolverConfiguration::time_evolution(10.0, 50))
            .unwrap();

        plot_chromatogram(&result, 10, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }
}