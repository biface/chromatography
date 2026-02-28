//! Plot configuration shared across visualization modules
//!
//! This module defines common configuration structures used by both
//! steady-state and chromatogram plotting functions.

use plotters::prelude::*;

/// Configuration for customizing plots
///
/// Used by both steady-state (spatial) and chromatogram (temporal) plots.
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
/// use chrom_rs::output::visualization::PlotConfig;
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

    /// Plot title (default: "Plot")
    pub title: String,

    /// X-axis label (default: auto-set by plot type)
    pub xlabel: String,

    /// Y-axis label (default: "Concentration (mol/L)")
    pub ylabel: String,

    /// Line color for single-species plots (default: RED)
    pub line_color: RGBColor,

    /// Optional colors for multi-species plots (one per species)
    ///
    /// If None, uses default palette: [RED, BLUE, GREEN, MAGENTA, CYAN, ...]
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
            title: "Plot".to_string(),
            xlabel: String::new(),  // Set by specific plot type
            ylabel: "Concentration (mol/L)".to_string(),
            line_color: RED,
            species_colors: None,
            background: WHITE,
            line_width: 2,
            show_grid: true,
        }
    }
}

/// Helper trait to accept both `String` and `None` for optional titles
pub trait IntoOptionalTitle {
    fn into_optional_title(self) -> Option<String>;
}

impl IntoOptionalTitle for &str {
    fn into_optional_title(self) -> Option<String> {
        Some(self.to_string())
    }
}

impl IntoOptionalTitle for String {
    fn into_optional_title(self) -> Option<String> {
        Some(self)
    }
}

impl<T: IntoOptionalTitle> IntoOptionalTitle for Option<T> {
    fn into_optional_title(self) -> Option<String> {
        self.and_then(|t| t.into_optional_title())
    }
}

/// Constant for no title (default title will be used)
///
/// # Example
///
/// ```rust,ignore
/// let config = PlotConfig::chromatogram(NO_TITLE);
/// ```
pub const NO_TITLE: Option<&str> = None;

impl PlotConfig {
    /// Create config for chromatograms with optional custom title
    ///
    /// Sets xlabel to "Time (s)" and title to custom value or "Chromatogram"
    ///
    /// # Arguments
    ///
    /// * `title` - Custom title (String, &str) or None for default
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // With custom title (no Some() needed!)
    /// let config = PlotConfig::chromatogram("TFA Elution");
    /// let config = PlotConfig::chromatogram(format!("TFA: {}", method));
    ///
    /// // With default title
    /// let config = PlotConfig::chromatogram(None::<&str>);
    /// ```
    pub fn chromatogram(title: impl IntoOptionalTitle) -> Self {
        let mut config = Self::default();
        config.xlabel = "Time (s)".to_string();
        config.title = title
            .into_optional_title()
            .unwrap_or_else(|| "Chromatogram".to_string());
        config
    }

    /// Create config for steady-state spatial profiles with optional custom title
    ///
    /// Sets xlabel to "Position (m)" and title to custom value or "Spatial Profile"
    ///
    /// # Arguments
    ///
    /// * `title` - Custom title (String, &str) or None for default
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // With custom title (no Some() needed!)
    /// let config = PlotConfig::steady_state("Final Equilibrium");
    /// let config = PlotConfig::steady_state(format!("Profile at t={}", time));
    ///
    /// // With default title
    /// let config = PlotConfig::steady_state(None::<&str>);
    /// ```
    pub fn steady_state(title: impl IntoOptionalTitle) -> Self {
        let mut config = Self::default();
        config.xlabel = "Position (m)".to_string();
        config.title = title
            .into_optional_title()
            .unwrap_or_else(|| "Spatial Profile".to_string());
        config
    }

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
    pub(crate) fn get_species_color(&self, species_index: usize) -> RGBColor {
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
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert!(config.show_grid);
    }

    #[test]
    fn test_chromatogram_config_default() {
        let config = PlotConfig::chromatogram(NO_TITLE);
        assert_eq!(config.xlabel, "Time (s)");
        assert_eq!(config.title, "Chromatogram");
    }

    #[test]
    fn test_chromatogram_config_with_str() {
        let config = PlotConfig::chromatogram("TFA Elution");
        assert_eq!(config.xlabel, "Time (s)");
        assert_eq!(config.title, "TFA Elution");
    }

    #[test]
    fn test_chromatogram_config_with_string() {
        let title = format!("TFA: {}", "Gaussian");
        let config = PlotConfig::chromatogram(title);
        assert_eq!(config.xlabel, "Time (s)");
        assert_eq!(config.title, "TFA: Gaussian");
    }

    #[test]
    fn test_steady_state_config_default() {
        let config = PlotConfig::steady_state(NO_TITLE);
        assert_eq!(config.xlabel, "Position (m)");
        assert_eq!(config.title, "Spatial Profile");
    }

    #[test]
    fn test_steady_state_config_with_title() {
        let config = PlotConfig::steady_state("Final Equilibrium");
        assert_eq!(config.xlabel, "Position (m)");
        assert_eq!(config.title, "Final Equilibrium");
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
        use plotters::style::full_palette::{LIGHTBLUE, LIGHTGREEN, ORANGE};
        let config = PlotConfig::multi_species_colors(vec![ORANGE, LIGHTGREEN, LIGHTBLUE]);
        assert_eq!(config.get_species_color(0), ORANGE);
        assert_eq!(config.get_species_color(1), LIGHTGREEN);
        assert_eq!(config.get_species_color(2), LIGHTBLUE);
    }
}