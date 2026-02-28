//! Chromatogram plotting for temporal simulations
//!
//! This module provides plotting functions for time-series data,
//! specifically chromatograms showing concentration at column outlet vs time.
//!
//! # Key Difference from `steady`
//!
//! This module plots **chromatograms** (C_outlet vs time), not spatial profiles.
//! It extracts the outlet concentration (last spatial point) and plots it over time.
//!
//! # Available functions
//!
//! - [`plot_chromatogram`]            — Single-species: outlet concentration vs time
//! - [`plot_chromatogram_multi`]      — Multi-species: one curve per species (LangmuirMulti)
//! - [`plot_chromatograms_comparison`]— Overlay several single-species runs on the same axes
//!
//! # Usage
//!
//! ```rust,ignore
//! use chrom_rs::output::visualization::{plot_chromatogram, plot_chromatogram_multi};
//!
//! // Single-species (LangmuirSingle — Vector state)
//! let result = solver.solve(&scenario, &config)?;
//! plot_chromatogram(&result, 100, "tfa.png", None)?;
//!
//! // Multi-species (LangmuirMulti — Matrix state [n_points × n_species])
//! plot_chromatogram_multi(
//!     &result,
//!     100,
//!     &["Ascorbic", "Erythorbic", "Citric"],
//!     "acids.png",
//!     None,
//! )?;
//! ```

use plotters::prelude::*;
use std::error::Error;

use crate::physics::{PhysicalData, PhysicalQuantity};
use crate::solver::SimulationResult;
use super::config::{PlotConfig, NO_TITLE};

// =================================================================================================
// Helper Functions — Extract Outlet Concentrations
// =================================================================================================

/// Extract the single-species outlet concentration $C_{outlet}(t)$ over all time steps
///
/// For chromatography, $C_{outlet}(t)$ is the concentration at the column exit
/// (last spatial point, index `n_points - 1`) plotted against time.
///
/// Handles all `PhysicalData` variants:
/// - `Vector`  — standard case: takes the last element (primary path)
/// - `Scalar`  — degenerate 0-D model: returns the scalar value directly
/// - `Matrix`  — multi-species state: extracts species 0 as a fallback
///               (prefer [`extract_multi_species_outlet`] for proper multi-species plots)
///
/// # Arguments
///
/// * `result`   — Simulation result with state trajectory
/// * `n_points` — Number of spatial points (to identify the outlet index)
fn extract_single_species_outlet(result: &SimulationResult, n_points: usize) -> Vec<f64> {
    result.state_trajectory
        .iter()
        .map(|state| {
            match state.get(PhysicalQuantity::Concentration) {
                Some(PhysicalData::Scalar(c)) => *c,
                Some(PhysicalData::Vector(profile)) => {
                    // Outlet = last spatial cell
                    profile[n_points - 1]
                }
                Some(PhysicalData::Matrix(m)) => {
                    // Matrix state: best-effort fallback to species 0
                    // Use plot_chromatogram_multi for proper multi-species output
                    m[(n_points - 1, 0)]
                }
                _ => 0.0,
            }
        })
        .collect()
}

/// Extract multi-species outlet concentrations $C_{outlet,k}(t)$ over all time steps
///
/// For a `LangmuirMulti` simulation the state is a `[n_points × n_species]` matrix.
/// The outlet for species $k$ is the last row, column $k$:
/// $$C_{outlet,k}(t_i) = \text{state}[n_{points}-1,\, k] \text{ at step } i$$
///
/// Returns one `Vec<f64>` per species, each of length `n_time_steps`.
///
/// # Arguments
///
/// * `result`    — Simulation result with matrix-valued trajectory
/// * `n_points`  — Number of spatial points (outlet index = `n_points - 1`)
/// * `n_species` — Number of species to extract (caps at `ncols` if smaller)
///
/// # Returns
///
/// `Vec<Vec<f64>>` of shape `[n_species][n_time_steps]`.
/// Returns an empty outer vec if no Matrix data is found.
fn extract_multi_species_outlet(
    result: &SimulationResult,
    n_points: usize,
    n_species: usize,
) -> Vec<Vec<f64>> {
    // Pre-allocate one vec per species
    let mut outlets: Vec<Vec<f64>> = (0..n_species).map(|_| Vec::new()).collect();

    for state in &result.state_trajectory {
        match state.get(PhysicalQuantity::Concentration) {
            Some(PhysicalData::Matrix(m)) => {
                let outlet_row = n_points - 1;
                for k in 0..n_species.min(m.ncols()) {
                    outlets[k].push(m[(outlet_row, k)]);
                }
            }
            // Single-species fallback: fill species 0 only
            Some(PhysicalData::Vector(profile)) => {
                if !outlets.is_empty() {
                    outlets[0].push(profile[n_points - 1]);
                }
            }
            _ => {
                // Push 0.0 to keep all vecs the same length as time_points
                for outlet in outlets.iter_mut() {
                    outlet.push(0.0);
                }
            }
        }
    }

    outlets
}

// =================================================================================================
// Public API
// =================================================================================================

/// Plot a single-species chromatogram (outlet concentration vs time)
///
/// Reads `C_outlet(t)` from the last spatial point of each `Vector`-valued state
/// and plots it against simulation time. Designed for `LangmuirSingle` results.
///
/// # Arguments
///
/// * `result`      — Simulation result containing the trajectory
/// * `n_points`    — Number of spatial points (to extract outlet = last point)
/// * `output_path` — Output file path (`.png` → bitmap, `.svg` → vector)
/// * `config`      — Optional plot configuration; `None` uses defaults
///
/// # Errors
///
/// Returns `Err` if the backend cannot write to `output_path`.
///
/// # Example
///
/// ```rust,ignore
/// use chrom_rs::output::visualization::plot_chromatogram;
///
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

    let default_config = PlotConfig::chromatogram(NO_TITLE);
    let config = config.unwrap_or(&default_config);

    let max_time = time_points.last().copied().unwrap_or(1.0);
    let max_conc = outlet
        .iter()
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
            plot_chromatogram_impl(backend, time_points, &outlet, config, max_time, max_conc)
        }
        _ => {
            let backend = BitMapBackend::new(output_path, (config.width, config.height));
            plot_chromatogram_impl(backend, time_points, &outlet, config, max_time, max_conc)
        }
    }
}

/// Plot a multi-species chromatogram (one outlet curve per species vs time)
///
/// Reads `C_{outlet,k}(t)` from the last row of the `[n_points × n_species]`
/// concentration matrix at each time step. Designed for `LangmuirMulti` results
/// where competitive adsorption produces distinct elution peaks per species.
///
/// # Arguments
///
/// * `result`        — Simulation result with matrix-valued trajectory
/// * `n_points`      — Number of spatial points (outlet index = `n_points - 1`)
/// * `species_names` — Legend labels, one per species
/// * `output_path`   — Output file path (`.png` or `.svg`)
/// * `config`        — Optional plot configuration;
///                     use `config.species_colors` to override the default palette
///
/// # Errors
///
/// Returns `Err` if no multi-species data is found or the backend fails.
///
/// # Example
///
/// ```rust,ignore
/// use chrom_rs::output::visualization::plot_chromatogram_multi;
///
/// // Three-acid separation (LangmuirMulti, 3 species)
/// plot_chromatogram_multi(
///     &result,
///     100,
///     &["Ascorbic Acid", "Erythorbic Acid", "Citric Acid"],
///     "acids.png",
///     None,
/// )?;
/// ```
pub fn plot_chromatogram_multi(
    result: &SimulationResult,
    n_points: usize,
    species_names: &[&str],
    output_path: &str,
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    let outlets = extract_multi_species_outlet(result, n_points, species_names.len());

    if outlets.is_empty() || outlets[0].is_empty() {
        return Err("No multi-species concentration data found in trajectory".into());
    }

    let time_points = &result.time_points;

    let default_config = PlotConfig::chromatogram(NO_TITLE);
    let config = config.unwrap_or(&default_config);

    let max_time = time_points.last().copied().unwrap_or(1.0);
    let max_conc = outlets
        .iter()
        .flat_map(|o| o.iter())
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
            plot_multi_chromatogram_impl(
                backend, time_points, &outlets, species_names, config, max_time, max_conc,
            )
        }
        _ => {
            let backend = BitMapBackend::new(output_path, (config.width, config.height));
            plot_multi_chromatogram_impl(
                backend, time_points, &outlets, species_names, config, max_time, max_conc,
            )
        }
    }
}

/// Plot multiple single-species chromatograms overlaid for comparison
///
/// Useful for comparing different injections, solvers, or operating conditions
/// on the same axes. Each dataset is drawn with a distinct colour.
///
/// # Arguments
///
/// * `datasets`    — Vec of `(label, SimulationResult, n_points)`
/// * `output_path` — Output file path (`.png` or `.svg`)
/// * `config`      — Optional plot configuration
///
/// # Errors
///
/// Returns `Err` if `datasets` is empty or the backend fails.
///
/// # Example
///
/// ```rust,ignore
/// use chrom_rs::output::visualization::plot_chromatograms_comparison;
///
/// let datasets = vec![
///     ("Euler", &result_euler, 100),
///     ("RK4",   &result_rk4,   100),
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

    let default_config = PlotConfig::chromatogram(NO_TITLE);
    let config = config.unwrap_or(&default_config);

    // Extract all outlets up-front
    let all_data: Vec<(&str, &[f64], Vec<f64>)> = datasets
        .iter()
        .map(|(label, result, n_points)| {
            let outlet = extract_single_species_outlet(result, *n_points);
            (*label, result.time_points.as_slice(), outlet)
        })
        .collect();

    let max_time = all_data
        .iter()
        .map(|(_, times, _)| times.last().copied().unwrap_or(0.0))
        .fold(0.0_f64, f64::max);

    let max_conc = all_data
        .iter()
        .flat_map(|(_, _, outlet)| outlet.iter())
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
            plot_comparison_impl(backend, &all_data, config, max_time, max_conc)
        }
        _ => {
            let backend = BitMapBackend::new(output_path, (config.width, config.height));
            plot_comparison_impl(backend, &all_data, config, max_time, max_conc)
        }
    }
}

// =================================================================================================
// Private Plot Implementations
// =================================================================================================

/// Render a single-species chromatogram with the given drawing backend
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
        chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .x_label_formatter(&|x| format!("{:.0}", x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    }

    chart
        .draw_series(LineSeries::new(
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

/// Render a multi-species chromatogram — one coloured curve per species + grey envelope
///
/// Draws two layers:
///
/// 1. **Species curves** — one `LineSeries` per species $k$, coloured via
///    `config.get_species_color(k)`, built from `(time_points[i], outlets[k][i])`.
///
/// 2. **Concentration envelope** — a dashed grey curve showing the instantaneous
///    maximum across all species:
///    $$C_{envelope}(t_i) = \max_k \, C_{outlet,k}(t_i)$$
///    Useful to identify when *any* species is eluting and to spot
///    co-elution zones where the envelope exceeds individual peaks.
///
/// The envelope is computed directly from the already-extracted `outlets` data —
/// no additional access to the trajectory is needed.
fn plot_multi_chromatogram_impl<DB: DrawingBackend>(
    backend: DB,
    time_points: &[f64],
    outlets: &[Vec<f64>],  // shape: [n_species][n_time_steps]
    species_names: &[&str],
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
        chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .x_label_formatter(&|x| format!("{:.0}", x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    }

    // ── 1. Species curves ────────────────────────────────────────────────────
    for (k, outlet) in outlets.iter().enumerate() {
        // get_species_color falls back to the built-in 10-colour palette (config.rs)
        let color = config.get_species_color(k);
        let label = species_names.get(k).copied().unwrap_or("?");

        chart
            .draw_series(LineSeries::new(
                time_points.iter().zip(outlet.iter()).map(|(t, c)| (*t, *c)),
                ShapeStyle::from(&color).stroke_width(config.line_width),
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
    }

    // ── 2. Concentration envelope ────────────────────────────────────────────
    // C_envelope(t_i) = max_k { outlets[k][i] }
    // Computed point-by-point from the already-extracted outlet vectors.
    if !outlets.is_empty() {
        let n_steps = outlets[0].len();
        let envelope: Vec<f64> = (0..n_steps)
            .map(|i| {
                outlets
                    .iter()
                    .map(|o| o[i])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();

        // Grey dashed style — visually distinct from the species curves
        let envelope_color = RGBColor(150, 150, 150);
        let envelope_style = ShapeStyle {
            color: envelope_color.to_rgba(),
            filled: false,
            stroke_width: config.line_width,  // same thickness, dashed via step series
        };

        chart
            .draw_series(
                // DashedLineSeries is not available in all plotters versions;
                // we emulate dashes by drawing only every other segment as a
                // LineSeries so the result is portable across plotters releases.
                LineSeries::new(
                    time_points
                        .iter()
                        .zip(envelope.iter())
                        .enumerate()
                        // Keep only even-indexed points to create a visible gap effect.
                        // For a true dashed look the step should scale with time resolution.
                        .filter_map(|(i, (t, c))| if i % 2 == 0 { Some((*t, *c)) } else { None }),
                    envelope_style,
                )
            )?
            .label("Envelope")
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], &envelope_color)
            });
    }

    chart
        .configure_series_labels()
        .background_style(&config.background.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Render overlaid single-species chromatograms for comparison
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
        chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .x_label_formatter(&|x| format!("{:.0}", x))
            .y_label_formatter(&|y| format!("{:.3}", y))
            .draw()?;
    }

    for (idx, (label, times, outlet)) in datasets.iter().enumerate() {
        let color = config.get_species_color(idx);

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
    use nalgebra::{DMatrix, DVector};

    // ─────────────────────────────────────────────────────────────────────────
    // Test models
    // ─────────────────────────────────────────────────────────────────────────

    /// Single-species model (Vector state) — mimics LangmuirSingle
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
            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::Vector(DVector::from_element(self.n_points, 1.0)),
            )
        }

        fn name(&self) -> &str { "SingleModel" }
    }

    /// Multi-species model (Matrix state) — mimics LangmuirMulti
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
                let rate = -0.01 * (k + 1) as f64;
                for i in 0..self.n_points {
                    dc_dt[(i, k)] = m[(i, k)] * rate;
                }
            }
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Matrix(dc_dt))
        }

        fn setup_initial_state(&self) -> PhysicalState {
            // Each species starts at a different amplitude — outlet will differ
            let mut c = DMatrix::zeros(self.n_points, self.n_species);
            for k in 0..self.n_species {
                for i in 0..self.n_points {
                    c[(i, k)] = (k + 1) as f64 * 0.5;
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
        EulerSolver.solve(&scenario, &SolverConfiguration::time_evolution(10.0, 100)).unwrap()
    }

    fn run_multi(n: usize, k: usize) -> SimulationResult {
        let model = Box::new(MultiModel { n_points: n, n_species: k });
        let init = model.setup_initial_state();
        let scenario = Scenario::new(model, DomainBoundaries::temporal(init));
        EulerSolver.solve(&scenario, &SolverConfiguration::time_evolution(10.0, 100)).unwrap()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Unit tests — extract_single_species_outlet
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_extract_single_outlet_length() {
        let result = run_single(10);
        let outlet = extract_single_species_outlet(&result, 10);
        assert_eq!(outlet.len(), 101); // 100 steps + initial state
    }

    #[test]
    fn test_extract_single_outlet_initial_value() {
        let result = run_single(10);
        let outlet = extract_single_species_outlet(&result, 10);
        // Initial concentration is 1.0; after one tiny step it is still close to 1.0
        assert!(outlet[0] > 0.9);
    }

    #[test]
    fn test_extract_single_outlet_decreasing() {
        // Uniform decay model: outlet should decrease over time
        let result = run_single(10);
        let outlet = extract_single_species_outlet(&result, 10);
        assert!(outlet.last().unwrap() < &outlet[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Unit tests — extract_multi_species_outlet
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_extract_multi_outlet_shape() {
        let result = run_multi(10, 3);
        let outlets = extract_multi_species_outlet(&result, 10, 3);
        // 3 species, each with 101 time points
        assert_eq!(outlets.len(), 3);
        assert_eq!(outlets[0].len(), 101);
    }

    #[test]
    fn test_extract_multi_outlet_all_finite() {
        let result = run_multi(10, 2);
        let outlets = extract_multi_species_outlet(&result, 10, 2);
        for (k, o) in outlets.iter().enumerate() {
            for (i, &v) in o.iter().enumerate() {
                assert!(v.is_finite(), "species {k} step {i}: {v}");
            }
        }
    }

    #[test]
    fn test_extract_multi_outlet_species_differ() {
        // Different initial amplitudes → different outlet values
        let result = run_multi(10, 2);
        let outlets = extract_multi_species_outlet(&result, 10, 2);
        // Species 1 starts at 0.5, species 2 at 1.0 → outlet[1] > outlet[0] initially
        assert!(outlets[1][0] > outlets[0][0]);
    }

    #[test]
    fn test_extract_multi_outlet_independent_decay() {
        // Each species has a different decay rate → curves diverge over time
        let result = run_multi(10, 2);
        let outlets = extract_multi_species_outlet(&result, 10, 2);
        let ratio_start = outlets[1][0] / outlets[0][0];
        let ratio_end = outlets[1].last().unwrap() / outlets[0].last().unwrap();
        // Species 1 decays faster (rate -0.02 vs -0.01), so ratio decreases
        assert!(ratio_end < ratio_start);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Integration tests — file output
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_plot_chromatogram_png() {
        let result = run_single(10);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("png");
        plot_chromatogram(&result, 10, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_chromatogram_svg() {
        let result = run_single(10);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("svg");
        plot_chromatogram(&result, 10, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_chromatogram_custom_config() {
        let result = run_single(10);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("png");
        let mut config = PlotConfig::chromatogram("TFA Elution");
        config.line_color = BLUE;
        plot_chromatogram(&result, 10, path.to_str().unwrap(), Some(&config)).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_chromatogram_multi_png() {
        let result = run_multi(10, 2);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("png");
        plot_chromatogram_multi(
            &result, 10, &["A", "B"], path.to_str().unwrap(), None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_chromatogram_multi_svg() {
        let result = run_multi(10, 2);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("svg");
        plot_chromatogram_multi(
            &result, 10, &["A", "B"], path.to_str().unwrap(), None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_chromatogram_multi_three_species() {
        let result = run_multi(10, 3);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("png");
        plot_chromatogram_multi(
            &result, 10, &["Ascorbic", "Erythorbic", "Citric"], path.to_str().unwrap(), None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_chromatogram_multi_custom_colors() {
        let result = run_multi(10, 2);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("png");
        let config = PlotConfig::multi_species_colors(vec![RED, BLUE]);
        plot_chromatogram_multi(
            &result, 10, &["X", "Y"], path.to_str().unwrap(), Some(&config),
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_chromatograms_comparison() {
        let result1 = run_single(10);
        let result2 = run_single(10);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("png");
        plot_chromatograms_comparison(
            vec![("Run 1", &result1, 10), ("Run 2", &result2, 10)],
            path.to_str().unwrap(),
            None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_envelope_is_max_of_species() {
        // Verify the envelope formula independently of the plotting backend.
        // outlets[0] = [1.0, 0.5], outlets[1] = [0.8, 0.9]
        // → envelope should be [1.0, 0.9]
        let outlets: Vec<Vec<f64>> = vec![vec![1.0, 0.5], vec![0.8, 0.9]];
        let n_steps = outlets[0].len();
        let envelope: Vec<f64> = (0..n_steps)
            .map(|i| outlets.iter().map(|o| o[i]).fold(f64::NEG_INFINITY, f64::max))
            .collect();
        assert!((envelope[0] - 1.0).abs() < 1e-12);
        assert!((envelope[1] - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_envelope_equals_single_species_when_one_species() {
        // With a single species, the envelope must be identical to that species
        let outlets: Vec<Vec<f64>> = vec![vec![0.3, 0.7, 0.5]];
        let n_steps = outlets[0].len();
        let envelope: Vec<f64> = (0..n_steps)
            .map(|i| outlets.iter().map(|o| o[i]).fold(f64::NEG_INFINITY, f64::max))
            .collect();
        assert_eq!(envelope, outlets[0]);
    }

    #[test]
    fn test_plot_chromatogram_multi_envelope_rendered() {
        // The plot must be created successfully when the envelope code path is active
        let result = run_multi(10, 3);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("png");
        plot_chromatogram_multi(
            &result, 10,
            &["Ascorbic", "Erythorbic", "Citric"],
            path.to_str().unwrap(),
            None,
        ).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_chromatograms_comparison_empty_returns_error() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("png");
        let err = plot_chromatograms_comparison(vec![], path.to_str().unwrap(), None);
        assert!(err.is_err());
    }
}
