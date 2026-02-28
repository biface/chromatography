//! Steady-state spatial profile plotting
//!
//! This module provides plotting functions for spatial profiles,
//! typically for steady-state problems where the final spatial distribution
//! is of interest.
//!
//! # Available functions
//!
//! - [`plot_steady_state`]            — Single-species profile from the final state
//! - [`plot_steady_state_multi`]      — Multi-species profiles from the final state (LangmuirMulti)
//! - [`plot_steady_state_comparison`] — Overlay arbitrary profiles from external data
//! - [`plot_profile_evolution`]       — N regularly-spaced time snapshots
//!
//! # Usage
//!
//! ```rust,ignore
//! use chrom_rs::output::visualization::{
//!     plot_steady_state,
//!     plot_steady_state_multi,
//! };
//!
//! // Single species (LangmuirSingle — Vector state)
//! let result = solver.solve(&scenario, &config)?;
//! plot_steady_state(&result, 0.25, "profile.png", None)?;
//!
//! // Multi-species (LangmuirMulti — Matrix [n_points × n_species])
//! plot_steady_state_multi(
//!     &result,
//!     0.25,
//!     &["Ascorbic", "Erythorbic", "Citric"],
//!     "multi_profile.png",
//!     None,
//! )?;
//! ```

use plotters::prelude::*;
use std::error::Error;

use crate::solver::SimulationResult;
use crate::physics::{PhysicalData, PhysicalQuantity};
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
/// * `column_length` - Physical length of the column \[m\]
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

    let physical_data = final_state
        .get(PhysicalQuantity::Concentration)
        .ok_or("Concentration not found")?;

    // Handle both single-species (Vector) and multi-species (Matrix) states.
    // For a Matrix state we extract species 0 as a best-effort single profile;
    // use plot_steady_state_multi for proper multi-species output.
    let conc_vec: Vec<f64> = match physical_data {
        PhysicalData::Vector(v) => v.iter().cloned().collect(),
        PhysicalData::Matrix(m) => (0..m.nrows()).map(|i| m[(i, 0)]).collect(),
        PhysicalData::Scalar(s) => vec![*s],
        // Array is not yet used in transport calculations — reject early
        // rather than silently returning a meaningless profile.
        PhysicalData::Array(_) => {
            return Err("Array physical data is not yet supported for spatial profile plotting".into());
        }
    };

    let n_points = conc_vec.len();

    // Create spatial grid: nodes from z=0 to z=L (inclusive)
    let z_values: Vec<f64> = (0..n_points)
        .map(|i| (i as f64 / (n_points - 1).max(1) as f64) * column_length)
        .collect();

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

    for (idx, (label, z_values, concentration)) in profiles.iter().enumerate() {
        // Use the shared palette from PlotConfig (falls back to built-in 10-colour default)
        let color = config.get_species_color(idx);

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
/// * `column_length` - Column length \[m\]
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
// Multi-Species Spatial Profile
// =================================================================================================

/// Plot multi-species spatial concentration profiles at the final simulation time
///
/// Designed for `LangmuirMulti` results where the state is a
/// `[n_points × n_species]` concentration matrix. Overlays one curve per species
/// on the same axes, coloured with the palette from `config.species_colors` or
/// the built-in default defined in [`PlotConfig`].
///
/// # Arguments
///
/// * `result`        — Simulation result with matrix-valued trajectory
/// * `column_length` — Column length $L$ **\[m\]**
/// * `species_names` — Legend labels, one per species
/// * `output_path`   — Output file path (`.png` or `.svg`)
/// * `config`        — Optional plot configuration
///
/// # Errors
///
/// Returns `Err` if no multi-species data is found in the final state or
/// the backend fails.
///
/// # Example
///
/// ```rust,ignore
/// use chrom_rs::output::visualization::plot_steady_state_multi;
///
/// plot_steady_state_multi(
///     &result,
///     0.25,
///     &["Ascorbic Acid", "Erythorbic Acid", "Citric Acid"],
///     "acids_profile.png",
///     None,
/// )?;
/// ```
pub fn plot_steady_state_multi(
    result: &SimulationResult,
    column_length: f64,
    species_names: &[&str],
    output_path: &str,
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    let final_state = result
        .state_trajectory
        .last()
        .ok_or("Empty trajectory")?;

    let physical_data = final_state
        .get(PhysicalQuantity::Concentration)
        .ok_or("Concentration not found")?;

    // Transpose [n_points × n_species] matrix into [n_species][n_points] vecs
    let profiles: Vec<Vec<f64>> = match physical_data {
        PhysicalData::Matrix(m) => {
            (0..species_names.len().min(m.ncols()))
                .map(|k| (0..m.nrows()).map(|i| m[(i, k)]).collect())
                .collect()
        }
        // Single-species fallback: wrap the vector in an outer Vec
        PhysicalData::Vector(v) => vec![v.iter().cloned().collect()],
        _ => return Err("No multi-species concentration data found in final state".into()),
    };

    if profiles.is_empty() {
        return Err("No multi-species concentration data found in final state".into());
    }

    let n_points = profiles[0].len();
    let z_values: Vec<f64> = (0..n_points)
        .map(|i| (i as f64 / (n_points - 1).max(1) as f64) * column_length)
        .collect();

    let default_config = PlotConfig::steady_state(NO_TITLE);
    let config = config.unwrap_or(&default_config);

    // Global y-axis maximum across all species
    let max_conc = profiles
        .iter()
        .flat_map(|p| p.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-10);

    let ext = std::path::Path::new(output_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");

    match ext {
        "svg" => {
            let backend = SVGBackend::new(output_path, (config.width, config.height));
            plot_multi_impl(backend, &z_values, &profiles, species_names, config, column_length, max_conc)
        }
        _ => {
            let backend = BitMapBackend::new(output_path, (config.width, config.height));
            plot_multi_impl(backend, &z_values, &profiles, species_names, config, column_length, max_conc)
        }
    }
}

/// Implementation for multi-species spatial profile with concrete backend
fn plot_multi_impl<DB: DrawingBackend>(
    backend: DB,
    z_values: &[f64],
    profiles: &[Vec<f64>],  // shape: [n_species][n_points]
    species_names: &[&str],
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
        chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .x_label_formatter(&|x| format!("{:.3}", x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    }

    for (k, profile) in profiles.iter().enumerate() {
        let color = config.get_species_color(k);
        let label = species_names.get(k).copied().unwrap_or("?");

        chart
            .draw_series(LineSeries::new(
                z_values.iter().zip(profile.iter()).map(|(z, c)| (*z, *c)),
                ShapeStyle::from(&color).stroke_width(config.line_width),
            ))?
            .label(label)
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
    use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
    use crate::solver::{Scenario, SolverConfiguration, DomainBoundaries, EulerSolver, Solver};
    use nalgebra::{DMatrix, DVector};

    // ─────────────────────────────────────────────────────────────────────────
    // Test models
    // ─────────────────────────────────────────────────────────────────────────

    /// Single-species model: Gaussian initial profile with uniform decay
    struct SingleModel {
        n_points: usize,
    }

    impl PhysicalModel for SingleModel {
        fn points(&self) -> usize { self.n_points }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            let c = state.get(PhysicalQuantity::Concentration).unwrap().as_vector();
            let dc_dt = DVector::from_element(c.len(), -0.01);
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(dc_dt))
        }

        fn setup_initial_state(&self) -> PhysicalState {
            // Gaussian profile centred at z = 0.5
            let mut c = DVector::zeros(self.n_points);
            for i in 0..self.n_points {
                let z = i as f64 / (self.n_points - 1).max(1) as f64;
                c[i] = (-(z - 0.5).powi(2) / (2.0 * 0.1 * 0.1)).exp();
            }
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(c))
        }

        fn name(&self) -> &str { "SingleModel" }
    }

    /// Multi-species model: n_species columns with species-dependent amplitudes
    struct MultiModel {
        n_points: usize,
        n_species: usize,
    }

    impl PhysicalModel for MultiModel {
        fn points(&self) -> usize { self.n_points }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            let m = state.get(PhysicalQuantity::Concentration).unwrap().as_matrix();
            let mut dc_dt = DMatrix::zeros(self.n_points, self.n_species);
            for k in 0..self.n_species {
                for i in 0..self.n_points {
                    dc_dt[(i, k)] = m[(i, k)] * -0.01 * (k + 1) as f64;
                }
            }
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Matrix(dc_dt))
        }

        fn setup_initial_state(&self) -> PhysicalState {
            let mut c = DMatrix::zeros(self.n_points, self.n_species);
            for k in 0..self.n_species {
                for i in 0..self.n_points {
                    let z = i as f64 / (self.n_points - 1).max(1) as f64;
                    // Each species has a Gaussian shifted by k * 0.1
                    let center = 0.3 + k as f64 * 0.1;
                    c[(i, k)] = (-(z - center).powi(2) / (2.0 * 0.08 * 0.08)).exp();
                }
            }
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Matrix(c))
        }

        fn name(&self) -> &str { "MultiModel" }
    }

    // Convenience runners
    fn run_single(n: usize) -> SimulationResult {
        let model = Box::new(SingleModel { n_points: n });
        let init = model.setup_initial_state();
        let scenario = Scenario::new(model, DomainBoundaries::temporal(init));
        EulerSolver.solve(&scenario, &SolverConfiguration::time_evolution(1.0, 10)).unwrap()
    }

    fn run_multi(n: usize, k: usize) -> SimulationResult {
        let model = Box::new(MultiModel { n_points: n, n_species: k });
        let init = model.setup_initial_state();
        let scenario = Scenario::new(model, DomainBoundaries::temporal(init));
        EulerSolver.solve(&scenario, &SolverConfiguration::time_evolution(1.0, 10)).unwrap()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // plot_steady_state — single-species
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_plot_steady_state_png() {
        let result = run_single(50);
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        plot_steady_state(&result, 0.25, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_steady_state_svg() {
        let result = run_single(50);
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("svg");
        plot_steady_state(&result, 0.25, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_steady_state_custom_config() {
        let result = run_single(50);
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        let mut config = PlotConfig::steady_state("My Profile");
        config.line_color = BLUE;
        config.width = 800;
        config.height = 600;
        plot_steady_state(&result, 0.25, path.to_str().unwrap(), Some(&config)).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_steady_state_matrix_fallback() {
        // A Matrix state must not panic — it falls back to species 0
        let result = run_multi(20, 2);
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        plot_steady_state(&result, 0.25, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // plot_steady_state_multi
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_plot_steady_state_multi_two_species() {
        let result = run_multi(50, 2);
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        plot_steady_state_multi(
            &result, 0.25, &["A", "B"], path.to_str().unwrap(), None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_steady_state_multi_three_species() {
        let result = run_multi(50, 3);
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        plot_steady_state_multi(
            &result, 0.25,
            &["Ascorbic", "Erythorbic", "Citric"],
            path.to_str().unwrap(), None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_steady_state_multi_svg() {
        let result = run_multi(30, 2);
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("svg");
        plot_steady_state_multi(
            &result, 0.25, &["X", "Y"], path.to_str().unwrap(), None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_steady_state_multi_custom_colors() {
        let result = run_multi(30, 2);
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        let config = PlotConfig::multi_species_colors(vec![RED, BLUE]);
        plot_steady_state_multi(
            &result, 0.25, &["X", "Y"], path.to_str().unwrap(), Some(&config),
        ).unwrap();
        assert!(path.exists());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // plot_steady_state_comparison
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_plot_steady_state_comparison_png() {
        let z: Vec<f64> = (0..50).map(|i| i as f64 / 49.0 * 0.25).collect();
        let c1: Vec<f64> = z.iter().map(|&zi| (-(zi - 0.1).powi(2) / 0.002).exp()).collect();
        let c2: Vec<f64> = z.iter().map(|&zi| (-(zi - 0.2).powi(2) / 0.002).exp()).collect();

        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");

        plot_steady_state_comparison(
            vec![("Initial", z.as_slice(), c1.as_slice()),
                 ("Final",   z.as_slice(), c2.as_slice())],
            path.to_str().unwrap(),
            None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_steady_state_comparison_empty_returns_error() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        let err = plot_steady_state_comparison(vec![], path.to_str().unwrap(), None);
        assert!(err.is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // plot_profile_evolution  (conservé de la version originale)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_plot_profile_evolution() {
        let model = Box::new(SingleModel { n_points: 50 });
        let init = model.setup_initial_state();
        let scenario = Scenario::new(model, DomainBoundaries::temporal(init));
        let result = EulerSolver
            .solve(&scenario, &SolverConfiguration::time_evolution(10.0, 50))
            .unwrap();

        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        plot_profile_evolution(&result, 0.25, 5, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_profile_evolution_empty_returns_error() {
        // Build a result with an empty trajectory by using 0 steps
        // (the solver always stores at least the initial state, so we
        // test the guard via an artificial empty SimulationResult instead)
        use crate::solver::SimulationResult;
        let empty = SimulationResult {
            time_points: vec![],
            state_trajectory: vec![],
            // final_state doit être un PhysicalState concret — on fournit
            // un état scalaire nul qui ne sera jamais lu (la trajectoire est vide).
            final_state: PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::Scalar(0.0),
            ),
            metadata: Default::default(),
        };
        let temp = tempfile::NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        let err = plot_profile_evolution(&empty, 0.25, 3, path.to_str().unwrap(), None);
        assert!(err.is_err());
    }
}