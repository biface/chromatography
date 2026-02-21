//! Steady-state spatial profile plotting
//!
//! This module provides plotting functions for spatial profiles,
//! typically for steady-state problems where the final spatial distribution
//! is of interest.
//!
//! # Usage
//!
//! ```rust,ignore
//! use chrom_rs::output::visualization::plot_steady_state;
//!
//! let result = solver.solve(&scenario, &config)?;
//! plot_steady_state(&result, 0.25, "profile.png", None)?;
//! ```

use plotters::prelude::*;
use std::error::Error;

use crate::solver::SimulationResult;
use crate::physics::PhysicalQuantity;
use super::config::{PlotConfig, NO_TITLE};

// =================================================================================================
// Core Plotting Functions
// =================================================================================================

/// Plot the final spatial profile (steady-state)
///
/// Plots the concentration profile C(z) at the final time step.
/// This is useful for steady-state problems or to visualize the
/// final spatial distribution after a transient.
///
/// # Arguments
///
/// * `result` - Simulation result containing state trajectory
/// * `column_length` - Physical length of the column [m]
/// * `output_path` - Path to save the plot (PNG or SVG)
/// * `config` - Optional plot configuration
///
/// # Example
///
/// ```rust,ignore
/// plot_steady_state(&result, 0.25, "profile.png", None)?;
/// ```
pub fn plot_steady_state(
    result: &SimulationResult,
    column_length: f64,
    output_path: &str,
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    // Extract final state
    let final_state = result
        .state_trajectory
        .last()
        .ok_or("Empty trajectory")?;

    let concentration = final_state
        .get(PhysicalQuantity::Concentration)
        .ok_or("Concentration not found")?
        .as_vector();

    let n_points = concentration.len();

    // Create spatial grid
    let z_values: Vec<f64> = (0..n_points)
        .map(|i| (i as f64 / (n_points - 1) as f64) * column_length)
        .collect();

    // Convert DVector to Vec for plotting
    let conc_vec: Vec<f64> = concentration.iter().cloned().collect();

    // Create default config if needed (avoid temporary value)
    let default_config = PlotConfig::steady_state(NO_TITLE);
    let config = config.unwrap_or(&default_config);

    // Determine plot range
    let max_conc = conc_vec
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-10);

    // Determine backend and plot
    let ext = std::path::Path::new(output_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");

    match ext {
        "svg" => {
            let backend = SVGBackend::new(output_path, (config.width, config.height));
            plot_steady_impl(backend, &z_values, &conc_vec, config, column_length, max_conc)
        }
        _ => {
            let backend = BitMapBackend::new(output_path, (config.width, config.height));
            plot_steady_impl(backend, &z_values, &conc_vec, config, column_length, max_conc)
        }
    }
}

/// Implementation for steady-state plotting with concrete backend
fn plot_steady_impl<DB: DrawingBackend>(
    backend: DB,
    z_values: &[f64],
    concentration: &[f64],
    config: &PlotConfig,
    max_z: f64,
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
        .build_cartesian_2d(0.0..max_z, 0.0..(max_conc * 1.1))?;

    if config.show_grid {
        chart.configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .x_label_formatter(&|x| format!("{:.3}", x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    }

    chart.draw_series(LineSeries::new(
        z_values.iter().zip(concentration.iter()).map(|(z, c)| (*z, *c)),
        ShapeStyle::from(&config.line_color).stroke_width(config.line_width),
    ))?
        .label("Concentration Profile")
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

/// Plot multiple spatial profiles for comparison
///
/// Overlays multiple concentration profiles on the same axes.
/// Useful for comparing different conditions, time steps, or models.
///
/// # Arguments
///
/// * `profiles` - Vec of (label, z_values, concentration)
/// * `output_path` - Path to save the plot
/// * `config` - Optional plot configuration
///
/// # Example
///
/// ```rust,ignore
/// let profiles = vec![
///     ("Initial", &z_grid, &c_initial),
///     ("Final", &z_grid, &c_final),
/// ];
/// plot_steady_state_comparison(profiles, "comparison.png", None)?;
/// ```
pub fn plot_steady_state_comparison(
    profiles: Vec<(&str, &[f64], &[f64])>,
    output_path: &str,
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    if profiles.is_empty() {
        return Err("No profiles provided".into());
    }

    // Create default config if needed (avoid temporary value)
    let default_config = PlotConfig::steady_state(NO_TITLE);
    let config = config.unwrap_or(&default_config);

    // Determine plot range
    let max_z = profiles
        .iter()
        .map(|(_, z, _)| z.last().copied().unwrap_or(0.0))
        .fold(0.0, f64::max);

    let max_conc = profiles
        .iter()
        .flat_map(|(_, _, c)| c.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-10);

    // Determine backend and plot
    let ext = std::path::Path::new(output_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");

    match ext {
        "svg" => {
            let backend = SVGBackend::new(output_path, (config.width, config.height));
            plot_comparison_impl(backend, &profiles, config, max_z, max_conc)
        }
        _ => {
            let backend = BitMapBackend::new(output_path, (config.width, config.height));
            plot_comparison_impl(backend, &profiles, config, max_z, max_conc)
        }
    }
}

/// Implementation for comparison plotting with concrete backend
fn plot_comparison_impl<DB: DrawingBackend>(
    backend: DB,
    profiles: &[(&str, &[f64], &[f64])],
    config: &PlotConfig,
    max_z: f64,
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
        .build_cartesian_2d(0.0..max_z, 0.0..(max_conc * 1.1))?;

    if config.show_grid {
        chart.configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .x_label_formatter(&|x| format!("{:.3}", x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    }

    // Colors for different profiles
    let colors = vec![BLUE, RED, GREEN, MAGENTA, CYAN];

    // Draw each profile
    for (idx, (label, z_values, concentration)) in profiles.iter().enumerate() {
        let color = colors[idx % colors.len()];  // Copy instead of reference

        chart
            .draw_series(LineSeries::new(
                z_values.iter().zip(concentration.iter()).map(|(z, c)| (*z, *c)),
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

/// Plot spatial profile evolution (multiple time snapshots)
///
/// Shows how the spatial profile evolves over time by plotting
/// snapshots at regular intervals.
///
/// # Arguments
///
/// * `result` - Simulation result
/// * `column_length` - Column length [m]
/// * `n_snapshots` - Number of time snapshots to show
/// * `output_path` - Path to save the plot
/// * `config` - Optional plot configuration
///
/// # Example
///
/// ```rust,ignore
/// // Show 5 profiles at different times
/// plot_profile_evolution(&result, 0.25, 5, "evolution.png", None)?;
/// ```
pub fn plot_profile_evolution(
    result: &SimulationResult,
    column_length: f64,
    n_snapshots: usize,
    output_path: &str,
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    if result.state_trajectory.is_empty() {
        return Err("Empty trajectory".into());
    }

    let total_steps = result.state_trajectory.len();
    let step_interval = total_steps / n_snapshots.min(total_steps);

    // Extract snapshots
    let mut profiles = Vec::new();
    for i in 0..n_snapshots {
        let idx = (i * step_interval).min(total_steps - 1);
        let state = &result.state_trajectory[idx];
        let time = result.time_points[idx];

        let concentration = state
            .get(PhysicalQuantity::Concentration)
            .ok_or("Concentration not found")?
            .as_vector();

        let n_points = concentration.len();
        let z_values: Vec<f64> = (0..n_points)
            .map(|j| (j as f64 / (n_points - 1) as f64) * column_length)
            .collect();

        let c_vec: Vec<f64> = concentration.iter().cloned().collect();

        profiles.push((format!("t={:.1}s", time), z_values, c_vec));
    }

    // Plot using comparison function
    let profile_refs: Vec<(&str, &[f64], &[f64])> = profiles
        .iter()
        .map(|(label, z, c)| (label.as_str(), z.as_slice(), c.as_slice()))
        .collect();

    plot_steady_state_comparison(profile_refs, output_path, config)
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
    use crate::solver::{Scenario, SolverConfiguration, DomainBoundaries, EulerSolver, Solver};
    use nalgebra::DVector;

    struct TestModel {
        n_points: usize,
    }

    impl PhysicalModel for TestModel {
        fn points(&self) -> usize {
            self.n_points
        }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            let c = state.get(PhysicalQuantity::Concentration).unwrap().as_vector();
            let dc_dt = DVector::from_element(c.len(), -0.01);
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(dc_dt))
        }

        fn setup_initial_state(&self) -> PhysicalState {
            // Gaussian profile
            let mut c = DVector::zeros(self.n_points);
            for i in 0..self.n_points {
                let z = i as f64 / (self.n_points - 1) as f64;
                let center = 0.5;
                let width = 0.1;
                c[i] = (-(z - center).powi(2) / (2.0 * width * width)).exp();
            }
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(c))
        }

        fn name(&self) -> &str {
            "TestModel"
        }
    }

    #[test]
    fn test_plot_steady_state() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");

        let model = Box::new(TestModel { n_points: 50 });
        let initial = model.setup_initial_state();
        let scenario = Scenario::new(model, DomainBoundaries::temporal(initial));

        let result = EulerSolver
            .solve(&scenario, &SolverConfiguration::time_evolution(1.0, 10))
            .unwrap();

        plot_steady_state(&result, 0.25, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_profile_evolution() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");

        let model = Box::new(TestModel { n_points: 50 });
        let initial = model.setup_initial_state();
        let scenario = Scenario::new(model, DomainBoundaries::temporal(initial));

        let result = EulerSolver
            .solve(&scenario, &SolverConfiguration::time_evolution(10.0, 50))
            .unwrap();

        plot_profile_evolution(&result, 0.25, 5, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }
}