//! Static plot generation for chromatography simulation results
//!
//! This module uses the `plotters` library to generate high-quality static images
//! (PNG, SVG) showing chromatograms and other simulation outputs.
//!
//! # Features
//!
//! - **Direct SimulationResult support**: Pass results directly from solvers
//! - **High-quality output**: Production-ready PNG and SVG images
//! - **Customizable**: PlotConfig for colors, labels, sizes, multi-species colors
//! - **Single and multi-species**: Supports both mono and multi-component chromatograms
//!
//! # Architecture Evolution
//!
//! **v0.1.0** (Current): Static images only
//! - `plot_chromatogram()`: Single species chromatogram
//! - `plot_chromatogram_multi()`: Multi-species with overlay
//! - `plot_result()`: Direct from SimulationResult
//! - Output: PNG/SVG files
//!
//! **v0.2.0+** (Future): Dynamic visualizations
//! - Animations using `state_trajectory` data
//! - Heatmaps showing C(z,t) evolution
//! - Interactive plots with real-time controls
//!
//! # Example: From SimulationResult (Recommended)
//!
//! ```rust,ignore
//! use chrom_rs::output::visualization::{plot_result, PlotConfig};
//! use chrom_rs::solver::{Solver, SolverConfiguration, EulerSolver};
//!
//! // Run simulation
//! let result = EulerSolver.solve(&scenario, &config)?;
//!
//! // Plot directly from result
//! plot_result(&result, "chromatogram.png", None)?;
//!
//! // Or with custom config
//! let mut plot_config = PlotConfig::default();
//! plot_config.title = "TFA Chromatogram".to_string();
//! plot_result(&result, "tfa.png", Some(&plot_config))?;
//! ```
//!
//! # Example: Multi-Species with Custom Colors
//!
//! ```rust,ignore
//! use chrom_rs::output::visualization::{plot_result_multi, PlotConfig};
//! use plotters::prelude::*;
//!
//! // After multi-species simulation
//! let mut config = PlotConfig::default();
//! config.title = "Acid Separation".to_string();
//! config.species_colors = Some(vec![RED, BLUE, GREEN]);
//!
//! plot_result_multi(
//!     &result,
//!     "acids.png",
//!     &["Ascorbic", "Erythorbic", "Citric"],
//!     Some(&config),
//! )?;
//! ```

use plotters::prelude::*;
use std::error::Error;

// Import SimulationResult from solver module
use crate::solver::SimulationResult;

// Import Physical specifications from physics module
use crate::physics::{PhysicalQuantity, PhysicalData};

// =================================================================================================
// Configuration
// =================================================================================================

/// Configuration for customizing plots
///
/// # Fields
///
/// - `width`, `height`: Dimensions in pixels
/// - `title`: Plot title
/// - `xlabel`, `ylabel`: Axis labels
/// - `line_color`: Line color for single-species plots
/// - `species_colors`: Optional colors for multi-species plots (one per species)
/// - `background`: Background color
/// - `line_width`: Line thickness in pixels
/// - `show_grid`: Whether to show grid lines
///
/// # Example: Single Species
///
/// ```rust,ignore
/// use plotters::prelude::*;
///
/// let mut config = PlotConfig::default();
/// config.title = "TFA Chromatogram".to_string();
/// config.line_color = RED;
/// config.width = 1920;  // Full HD
/// config.height = 1080;
/// ```
///
/// # Example: Multi-Species with Custom Colors
///
/// ```rust,ignore
/// let mut config = PlotConfig::default();
/// config.title = "Multi-Component Separation".to_string();
/// config.species_colors = Some(vec![
///     RED,
///     BLUE,
///     GREEN,
///     RGBColor(255, 165, 0),  // Orange
///     MAGENTA,
/// ]);
/// ```
#[derive(Clone)]
pub struct PlotConfig {
    /// Image width in pixels (default: 1024)
    pub width: u32,

    /// Image height in pixels (default: 768)
    pub height: u32,

    /// Plot title (default: "Chromatogram")
    pub title: String,

    /// X-axis label (default: "Time (s)")
    pub xlabel: String,

    /// Y-axis label (default: "Concentration (mol/L)")
    pub ylabel: String,

    /// Line color for single-species plots (default: RED)
    pub line_color: RGBColor,

    /// Optional colors for multi-species plots (one per species)
    ///
    /// If None, uses default palette: [RED, BLUE, GREEN, MAGENTA, CYAN, BLACK, YELLOW, ...]
    /// If Some, must have at least as many colors as species
    pub species_colors: Option<Vec<RGBColor>>,

    /// Background color (default: WHITE)
    pub background: RGBColor,

    /// Line width in pixels (default: 2)
    pub line_width: u32,

    /// Show grid lines (default: true)
    pub show_grid: bool,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            title: "Chromatogram".to_string(),
            xlabel: "Time (s)".to_string(),
            ylabel: "Concentration (mol/L)".to_string(),
            line_color: RED,
            species_colors: None,
            background: WHITE,
            line_width: 2,
            show_grid: true,
        }
    }
}

impl PlotConfig {
    /// Create config for multi-species with custom colors
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use plotters::prelude::*;
    ///
    /// let config = PlotConfig::multi_species_colors(vec![RED, BLUE, GREEN]);
    /// ```
    pub fn multi_species_colors(colors: Vec<RGBColor>) -> Self {
        let mut config = Self::default();
        config.species_colors = Some(colors);
        config
    }

    /// Get color for species at index i
    ///
    /// Uses custom colors if provided, otherwise falls back to default palette
    fn get_species_color(&self, species_index: usize) -> RGBColor {
        if let Some(ref colors) = self.species_colors {
            if species_index < colors.len() {
                return colors[species_index];
            }
        }

        // Default palette
        let default_colors = vec![
            RED,
            BLUE,
            GREEN,
            MAGENTA,
            CYAN,
            BLACK,
            RGBColor(255, 165, 0),  // Orange
            RGBColor(128, 0, 128),   // Purple
            RGBColor(255, 192, 203), // Pink
            RGBColor(165, 42, 42),   // Brown
        ];

        default_colors[species_index % default_colors.len()]
    }
}

// =================================================================================================
// Helper Functions - Extract Data from SimulationResult
// =================================================================================================

/// Extract time points and concentration data for single-species from SimulationResult
///
/// Returns (time_points, concentrations) where concentrations are extracted from
/// the PhysicalQuantity::Concentration in each state.
///
/// # Panics
///
/// Panics if concentration data is missing or not in expected format
fn extract_single_species_data(result: &SimulationResult) -> (Vec<f64>, Vec<f64>) {
    let time_points = result.time_points.clone();

    let concentrations: Vec<f64> = result.state_trajectory.iter()
        .map(|state| {
            let conc_data = state.get(PhysicalQuantity::Concentration)
                .expect("Concentration data missing in state");

            match conc_data {
                PhysicalData::Scalar(c) => *c,
                PhysicalData::Vector(v) => {
                    // If vector, sum all components (for outlet concentration)
                    v.sum()
                },
                _ => panic!("Unexpected concentration data format"),
            }
        })
        .collect();

    (time_points, concentrations)
}

/// Extract time points and multi-species concentration data from SimulationResult
///
/// Returns (time_points, species_concentrations) where species_concentrations[i]
/// is the concentration vector for species i over time.
///
/// # Panics
///
/// Panics if concentration data is missing or not in vector format
fn extract_multi_species_data(result: &SimulationResult, n_species: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let time_points = result.time_points.clone();

    // Initialize vectors for each species
    let mut species_concs: Vec<Vec<f64>> = vec![Vec::new(); n_species];

    for state in &result.state_trajectory {
        let conc_data = state.get(PhysicalQuantity::Concentration)
            .expect("Concentration data missing in state");

        match conc_data {
            PhysicalData::Vector(v) => {
                assert_eq!(v.len(), n_species, "Species count mismatch");
                for i in 0..n_species {
                    species_concs[i].push(v[i]);
                }
            },
            _ => panic!("Multi-species requires Vector concentration data"),
        }
    }

    (time_points, species_concs)
}

/// Helper function to draw multi-species chromatogram on any drawing area
fn draw_multi_on_area<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    time_points: &[f64],
    species_concentrations: &[Vec<f64>],
    species_names: &[&str],
    config: &PlotConfig,
) -> Result<(), Box<dyn Error>> 
where 
    <DB as DrawingBackend>::ErrorType: 'static
{
    let n_species = species_concentrations.len();

    // Find global ranges
    let max_time = time_points.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut max_concentration = f64::NEG_INFINITY;
    let mut min_concentration = f64::INFINITY;

    for concs in species_concentrations {
        for &c in concs {
            max_concentration = max_concentration.max(c);
            min_concentration = min_concentration.min(c);
        }
    }

    let y_range = max_concentration - min_concentration;
    let y_min = (min_concentration - 0.1 * y_range).max(0.0);
    let y_max = max_concentration + 0.1 * y_range;

    root.fill(&config.background)?;

    // Create chart
    let mut chart = ChartBuilder::on(root)
        .caption(&config.title, ("sans-serif", 40.0).into_font())
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0..max_time, y_min..y_max)?;

    // Configure mesh
    let mut mesh = chart.configure_mesh();
    mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);

    if config.show_grid {
        mesh.draw()?;
    } else {
        mesh.disable_mesh().draw()?;
    }

    // Draw lines for each species
    for i in 0..n_species {
        let color = config.get_species_color(i);
        chart.draw_series(LineSeries::new(
            time_points.iter().zip(species_concentrations[i].iter()).map(|(t, c)| (*t, *c)),
            color.stroke_width(config.line_width)
        ))?
            .label(species_names[i])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(config.line_width)));
    }

    // Draw legend
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Helper function to draw single-species chromatogram on any drawing area
fn draw_single_on_area<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    time_serie: &[f64],
    concentration_serie: &[f64],
    config: &PlotConfig,
) -> Result<(), Box<dyn Error>> 
where 
    <DB as DrawingBackend>::ErrorType: 'static
{
    // Find ranges for axes
    let max_time = time_serie.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_concentration = concentration_serie.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_concentration = concentration_serie.iter().cloned().fold(f64::INFINITY, f64::min);

    // Build margins (10% space)
    let y_range = max_concentration - min_concentration;
    let y_min = (min_concentration - 0.1 * y_range).max(0.0);
    let y_max = max_concentration + 0.1 * y_range;

    root.fill(&config.background)?;

    // Create chart
    let mut chart = ChartBuilder::on(root)
        .caption(&config.title, ("sans-serif", 40.0).into_font())
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0..max_time, y_min..y_max)?;

    // Configure mesh
    let mut mesh = chart.configure_mesh();
    mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);

    if config.show_grid {
        mesh.draw()?;
    } else {
        mesh.disable_x_mesh().draw()?;
    }

    // Draw chromatogram line
    chart.draw_series(LineSeries::new(
        time_serie.iter().zip(concentration_serie.iter()).map(|(t, c)| (*t, *c)),
        config.line_color.stroke_width(config.line_width)
    ))?;

    root.present()?;
    Ok(())
}

// =================================================================================================
// Low-Level Plotting Functions - Direct Data Arrays (for flexibility)
// =================================================================================================

/// Plot chromatogram from raw time and concentration arrays
///
/// This low-level function provides maximum flexibility by accepting raw data arrays.
/// For most use cases, prefer `plot_result()` which extracts data from SimulationResult.
///
/// # Arguments
///
/// * `time` - Time points \[s\]
/// * `concentration` - Concentration values \[mol/L\]
/// * `output_path` - Output file path (.png or .svg)
/// * `config` - Optional PlotConfig
///
/// # Example
///
/// ```rust,ignore
/// let time = vec![0.0, 1.0, 2.0, 3.0];
/// let conc = vec![0.0, 0.5, 1.0, 0.5];
/// plot_chromatogram(&time, &conc, "manual.png", None)?;
/// ```
pub fn plot_chromatogram(
    time_serie: &[f64],
    concentration_serie: &[f64],
    output_path: &str,
    configuration: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    let owned_config = configuration.cloned().unwrap_or(PlotConfig::default());
    let config = &owned_config;

    assert_eq!(time_serie.len(),
               concentration_serie.len(),
               "Time and concentration series must have same length");

    // Create backend
    if output_path.ends_with(".svg") {
        let root = SVGBackend::new(output_path, (config.width, config.height)).into_drawing_area();
        draw_single_on_area(&root, time_serie, concentration_serie, &config)
    } else {
        let root = BitMapBackend::new(output_path, (config.width, config.height)).into_drawing_area();
        draw_single_on_area(&root, time_serie, concentration_serie, &config)
    }
}

/// Plot multi-species chromatogram from raw data arrays
///
/// # Arguments
///
/// * `time_points` - Shared time axis
/// * `species_concentrations` - Vector of concentration vectors (one per species)
/// * `output_path` - Output file path
/// * `species_names` - Names for legend
/// * `config` - Optional PlotConfig
///
/// # Errors
///
/// Returns error if:
/// - Data lengths are inconsistent
/// - Plotting fails
pub fn plot_chromatogram_multi(
    time_points: &[f64],
    species_concentrations: &[Vec<f64>],
    output_path: &str,
    species_names: &[&str],
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    let owned_config = config.cloned().unwrap_or(PlotConfig::default());
    let config = &owned_config;

    let n_species = species_concentrations.len();
    assert_eq!(n_species, species_names.len(), "Species concentrations and names count must match");

    // Create backend
    if output_path.ends_with(".svg") {
        let root = SVGBackend::new(output_path, (config.width, config.height)).into_drawing_area();
        draw_multi_on_area(&root, time_points, species_concentrations, species_names, config)
    } else {
        let root = BitMapBackend::new(output_path, (config.width, config.height)).into_drawing_area();
        draw_multi_on_area(&root, time_points, species_concentrations, species_names, config)
    }
}

// =================================================================================================
// Main Plotting Functions - Direct from SimulationResult (RECOMMENDED)
// =================================================================================================

/// Plot chromatogram directly from SimulationResult (single-species)
///
/// This is the **recommended** function for plotting single-species simulations.
/// It automatically extracts time and concentration data from the result.
///
/// # Arguments
///
/// * `result` - SimulationResult from solver
/// * `output_path` - Output file path (PNG/SVG based on extension)
/// * `config` - Optional PlotConfig (uses defaults if None)
///
/// # Example
///
/// ```rust,ignore
/// use chrom_rs::output::visualization::plot_result;
///
/// let result = EulerSolver.solve(&scenario, &config)?;
/// plot_result(&result, "tfa.png", None)?;
/// ```
///
/// # Errors
///
/// Returns error if:
/// - File cannot be written
/// - Concentration data is missing
/// - Plotting fails
pub fn plot_result(
    result: &SimulationResult,
    output_path: &str,
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    let (time_points, species_concs) = extract_single_species_data(result);
    plot_chromatogram(&time_points, &species_concs, output_path, config)
}

/// Plot multi-species chromatogram directly from SimulationResult
///
/// This is the **recommended** function for plotting multi-species simulations.
/// It automatically extracts time and multi-species concentration data.
///
/// # Arguments
///
/// * `result` - SimulationResult from solver
/// * `output_path` - Output file path (PNG/SVG)
/// * `species_names` - Names of species for legend
/// * `config` - Optional PlotConfig with custom colors
///
/// # Example
///
/// ```rust,ignore
/// use chrom_rs::output::visualization::{plot_result_multi, PlotConfig};
/// use plotters::prelude::*;
///
/// let result = RK4Solver.solve(&scenario, &config)?;
///
/// let mut plot_config = PlotConfig::default();
/// plot_config.species_colors = Some(vec![RED, BLUE, GREEN]);
///
/// plot_result_multi(
///     &result,
///     "acids.png",
///     &["Ascorbic", "Erythorbic", "Citric"],
///     Some(&plot_config),
/// )?;
/// ```
///
/// # Errors
///
/// Returns error if:
/// - File cannot be written
/// - Species names count doesn't match data
/// - Concentration data is missing or wrong format
pub fn plot_result_multi(
    result: &SimulationResult,
    output_path: &str,
    species_names: &[&str],
    config: Option<&PlotConfig>,
) -> Result<(), Box<dyn Error>> {
    let n_species = species_names.len();
    let (time_points, species_concs) = extract_multi_species_data(result, n_species);

    plot_chromatogram_multi(&time_points, &species_concs, output_path, species_names, config)
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use std::env::set_current_dir;
    use super::*;
    use tempfile::NamedTempFile;
    use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity};
    use crate::solver::{Scenario, SolverConfiguration, DomainBoundaries};
    use crate::solver::EulerSolver;
    use crate::solver::Solver;
    use nalgebra::DVector;
    use plotters::style::full_palette::{LIGHTBLUE, LIGHTGREEN, ORANGE};

    // ====== Mock models for graphic testing =====
    struct TestModel { amplitude: f64 }
    impl PhysicalModel for TestModel {
        fn points(&self) -> usize { 1 }
        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            let c = match state.get(PhysicalQuantity::Concentration).unwrap() {
                PhysicalData::Scalar(value) => *value,
                _ => panic!(),
            } ;

            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Scalar(-0.1 * c))
        }

        fn setup_initial_state(&self) -> PhysicalState {
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Scalar(self.amplitude))
        }

        fn name(&self) -> &str {
            "Model test for plots"
        }
    }

    struct TestModelMulti { n_species: usize }
    impl PhysicalModel for TestModelMulti {
        fn points(&self) -> usize { 1 }
        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            let c_vec = match state.get(PhysicalQuantity::Concentration).unwrap() {
                PhysicalData::Vector(v) => v.clone(),
                _ => panic!(),
            };
            let rates: Vec<f64> = (0..self.n_species)
                .map(|i| -0.1 * (i + 1) as f64 * c_vec[i])
                .collect();
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(DVector::from_vec(rates)))
        }
        fn setup_initial_state(&self) -> PhysicalState {
            let initial: Vec<f64> = (0..self.n_species).map(|i| 10.0 * (i + 1) as f64).collect();
            PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(DVector::from_vec(initial)))
        }
        fn name(&self) -> &str { "Model test for multi plot" }
    }

    // ====== Tests ======

    #[test]
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert!(config.show_grid);
    }

    #[test]
    fn test_get_species_color_default_palette() {
        let config = PlotConfig::default();
        assert_eq!(config.get_species_color(0), RED);
        assert_eq!(config.get_species_color(1), BLUE);
        assert_eq!(config.get_species_color(10), RED); // Wraparound
    }

    #[test]
    fn test_get_species_color_custom() {
        let config = PlotConfig::multi_species_colors(vec![ORANGE, LIGHTGREEN, LIGHTBLUE]);
        assert_eq!(config.get_species_color(0), ORANGE);
        assert_eq!(config.get_species_color(1), LIGHTGREEN);
        assert_eq!(config.get_species_color(2), LIGHTBLUE);
    }

    #[test]
    fn test_extract_single_species_scalar_data() {
        let model = Box::new(TestModel { amplitude: 10.0});
        let initial_state = model.setup_initial_state();
        let scenario = Scenario::new(
            model,
            DomainBoundaries::temporal(initial_state)
        );
        let result = EulerSolver.solve(&scenario, &SolverConfiguration::time_evolution(10.0, 100)).unwrap();
        let (time, concentration) = extract_single_species_data(&result);
        assert_eq!(time.len(), concentration.len());
        assert!(concentration[0] > 9.0 && concentration[0] < 11.0);
    }

    #[test]
    fn test_extract_multi_species_data() {
        let model = Box::new(TestModelMulti { n_species: 3 });
        let initial_state = model.setup_initial_state();
        let scenario = Scenario::new(
            model,
            DomainBoundaries::temporal(initial_state)
        );
        let result = EulerSolver.solve(&scenario, &SolverConfiguration::time_evolution(10.0, 100)).unwrap();
        let (time, species_concs) = extract_multi_species_data(&result, 3);
        assert_eq!(time.len(), species_concs[0].len());
        assert_eq!(species_concs.len(), 3);
    }

    #[test]
    fn test_plot_png_chromatogram() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");

        let time_serie = vec![0.0, 1.0, 2.0];
        let concentration_serie = vec![0.0, 0.5, 0.0];

        plot_chromatogram(&time_serie, &concentration_serie, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_svg_chromatogram() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("svg");

        let time_serie = vec![0.0, 1.0, 2.0];
        let concentration_serie = vec![0.0, 0.5, 0.0];

        plot_chromatogram(&time_serie, &concentration_serie, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    #[should_panic(expected = "Time and concentration series must have same length")]
    fn test_plot_chromatogram_series_failed() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");

        let time_serie = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let concentration_serie = vec![0.0, 0.5, 0.0];

        plot_chromatogram(&time_serie, &concentration_serie, path.to_str().unwrap(), None).unwrap();
    }

    #[test]
    fn test_plot_png_chromatogram_multi() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");

        let time = vec![0.0, 1.0, 2.0];
        let s1 = vec![0.0, 0.8, 0.0];
        let s2 = vec![0.0, 0.4, 0.0];

        plot_chromatogram_multi(&time, &[s1, s2], path.to_str().unwrap(), &["A", "B"], None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_svg_chromatogram_multi() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("svg");

        let time = vec![0.0, 1.0, 2.0];
        let s1 = vec![0.0, 0.8, 0.0];
        let s2 = vec![0.0, 0.4, 0.0];

        plot_chromatogram_multi(&time, &[s1, s2], path.to_str().unwrap(), &["A", "B"], None).unwrap();
        assert!(path.exists());
    }

    #[test]
    #[should_panic(expected = "Species concentrations and names count must match")]
    fn test_plot_svg_chromatogram_multi_failed() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("svg");

        let time = vec![0.0, 1.0, 2.0];
        let s1 = vec![0.0, 0.5, 0.0];
        let s2 = vec![0.0, 0.4, 0.0];

        plot_chromatogram_multi(&time, &[s1, s2], path.to_str().unwrap(), &["A", "B", "C"], None).unwrap();
    }

    #[test]
    fn test_plot_result() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");
        let model = Box::new(TestModel {amplitude: 10.0});
        let initial_state = model.setup_initial_state();
        let scenario = Scenario::new(
            model,
            DomainBoundaries::temporal(initial_state)
        );

        let result = EulerSolver.solve(
            &scenario,
            &SolverConfiguration::time_evolution(10.0, 50)
        ).unwrap();

        plot_result(&result, path.to_str().unwrap(), None).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_result_multi() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().with_extension("png");

        let model = Box::new(TestModelMulti {n_species: 2});
        let initial_state = model.setup_initial_state();
        let scenario = Scenario::new(
            model,
            DomainBoundaries::temporal(initial_state)
        );

        let result = EulerSolver.solve(
            &scenario,
            &SolverConfiguration::time_evolution(10.0, 50)
        ).unwrap();

        plot_result_multi(&result, path.to_str().unwrap(), &["S1", "S2"], None).unwrap();
        assert!(path.exists());
    }
}