//! Chromatographic signal detector.
//!
//! This module defines [`Detector`] and [`DetectorPosition`], which specify
//! the location of the measurement point along the column axis.
//!
//! # Physical background
//!
//! In chromatography, the measured signal is the mobile-phase concentration
//! at the detection point $z_d$. The default position is the column outlet
//! ($z_d = L$), but some experimental setups place the detector at an
//! intermediate point along the column.
//!
//! # Position variants
//!
//! | Variant | Description | Constraint |
//! |---------|-------------|-----------|
//! | [`Outlet`](DetectorPosition::Outlet) | Column outlet $z_d = L$ | none |
//! | [`Relative`](DetectorPosition::Relative) | $z_d = r \cdot L$, $r \in [0, 1]$ | $r \in [0, 1]$ |
//! | [`Absolute`](DetectorPosition::Absolute) | $z_d$ \[m\] | $z_d > 0$, validated against $L$ at use time |
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::domain::{Detector, DetectorPosition};
//!
//! // Detector at column outlet (current default behaviour)
//! let det = Detector::outlet();
//! assert!(matches!(det.position, DetectorPosition::Outlet));
//!
//! // Detector at mid-column
//! let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
//! assert!((det.absolute_position(0.25) - 0.125).abs() < 1e-12);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// DetectorError
// =============================================================================

/// Errors returned by [`Detector::new`] or [`Detector::validate_against_column`].
#[derive(Debug)]
pub enum DetectorError {
    /// Relative position must lie in $[0, 1]$.
    InvalidRelativePosition(f64),

    /// Absolute position must be strictly positive ($z_d > 0$).
    InvalidAbsolutePosition(f64),

    /// Absolute position exceeds the column length.
    PositionExceedsColumn { position: f64, column_length: f64 },
}

impl fmt::Display for DetectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectorError::InvalidRelativePosition(r) => {
                write!(f, "detector: relative position must be in [0, 1], got {r}")
            }
            DetectorError::InvalidAbsolutePosition(z) => {
                write!(f, "detector: absolute position must be > 0, got {z}")
            }
            DetectorError::PositionExceedsColumn {
                position,
                column_length,
            } => {
                write!(
                    f,
                    "detector: position {position} m exceeds column length {column_length} m"
                )
            }
        }
    }
}

impl std::error::Error for DetectorError {}

// =============================================================================
// DetectorPosition
// =============================================================================

/// Detector position along the column axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DetectorPosition {
    /// Column outlet $z_d = L$ — default behaviour.
    Outlet,

    /// Relative position $z_d = r \cdot L$, $r \in [0, 1]$.
    Relative(f64),

    /// Absolute axial position $z_d$ \[m\].
    ///
    /// Doit être validée contre la longueur de la colonne avant usage via
    /// [`Detector::validate_against_column`].
    /// Must be validated against the column length before use.
    Absolute(f64),
}

// =============================================================================
// Detector
// =============================================================================

/// Chromatographic detector with configurable position.
///
/// `Detector` encapsulates the measurement point along the column axis.
/// The default position is the column outlet ([`DetectorPosition::Outlet`]),
/// reproducing the behaviour of `outlet_data`.
///
/// # Example
///
/// ```rust
/// use chrom_rs::domain::{Detector, DetectorPosition};
///
/// // Column outlet (default)
/// let det = Detector::outlet();
///
/// // Mid-column detector
/// let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
/// // For L = 0.25 m: z_d = 0.125 m
/// assert!((det.absolute_position(0.25) - 0.125).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detector {
    /// Detector position along the column axis.
    pub position: DetectorPosition,
}

impl Detector {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Creates a validated detector.
    ///
    /// # Errors
    ///
    /// Returns [`DetectorError`] if:
    /// - `Relative(r)` with $r \notin [0, 1]$
    /// - `Absolute(z)` with $z \leq 0$
    ///
    /// Note: `Absolute(z)` is not validated against the column length here
    /// (it is not known at construction time). Use
    /// [`validate_against_column`](Self::validate_against_column) to check
    /// compatibility at simulation time.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::{Detector, DetectorPosition, DetectorError};
    ///
    /// assert!(Detector::new(DetectorPosition::Outlet).is_ok());
    /// assert!(Detector::new(DetectorPosition::Relative(0.5)).is_ok());
    /// assert!(Detector::new(DetectorPosition::Absolute(0.1)).is_ok());
    /// assert!(matches!(
    ///     Detector::new(DetectorPosition::Relative(1.5)),
    ///     Err(DetectorError::InvalidRelativePosition(_))
    /// ));
    /// assert!(matches!(
    ///     Detector::new(DetectorPosition::Absolute(-0.1)),
    ///     Err(DetectorError::InvalidAbsolutePosition(_))
    /// ));
    /// ```
    pub fn new(position: DetectorPosition) -> Result<Self, DetectorError> {
        match &position {
            DetectorPosition::Outlet => {}
            DetectorPosition::Relative(r) => {
                if *r < 0.0 || *r > 1.0 {
                    return Err(DetectorError::InvalidRelativePosition(*r));
                }
            }
            DetectorPosition::Absolute(z) => {
                if *z <= 0.0 {
                    return Err(DetectorError::InvalidAbsolutePosition(*z));
                }
            }
        }
        Ok(Self { position })
    }

    /// Creates a detector at the column outlet (default behaviour).
    ///
    /// Equivalent to `Detector::new(DetectorPosition::Outlet)` — shorthand
    /// for the most common case, reproducing the current `outlet_data` usage.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::{Detector, DetectorPosition};
    ///
    /// let det = Detector::outlet();
    /// assert_eq!(det.position, DetectorPosition::Outlet);
    /// ```
    pub fn outlet() -> Self {
        Self {
            position: DetectorPosition::Outlet,
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Returns the absolute axial position $z_d$ \[m\] for a given column length $L$.
    ///
    /// | Variant | Result |
    /// |---------|--------|
    /// | `Outlet` | $L$ |
    /// | `Relative(r)` | $r \cdot L$ |
    /// | `Absolute(z)` | $z$ |
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::{Detector, DetectorPosition};
    ///
    /// assert!((Detector::outlet().absolute_position(0.25) - 0.25).abs() < 1e-12);
    ///
    /// let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
    /// assert!((det.absolute_position(0.25) - 0.125).abs() < 1e-12);
    ///
    /// let det = Detector::new(DetectorPosition::Absolute(0.1)).unwrap();
    /// assert!((det.absolute_position(0.25) - 0.1).abs() < 1e-12);
    /// ```
    pub fn absolute_position(&self, column_length: f64) -> f64 {
        match &self.position {
            DetectorPosition::Outlet => column_length,
            DetectorPosition::Relative(r) => r * column_length,
            DetectorPosition::Absolute(z) => *z,
        }
    }

    /// Validates an `Absolute` position against the column length.
    ///
    /// No-op for `Outlet` and `Relative` variants.
    ///
    /// # Errors
    ///
    /// Returns [`DetectorError::PositionExceedsColumn`] if the absolute
    /// position exceeds `column_length`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::{Detector, DetectorPosition, DetectorError};
    ///
    /// let det = Detector::new(DetectorPosition::Absolute(0.3)).unwrap();
    /// assert!(det.validate_against_column(0.25).is_err());
    /// assert!(det.validate_against_column(0.50).is_ok());
    /// ```
    pub fn validate_against_column(&self, column_length: f64) -> Result<(), DetectorError> {
        if let DetectorPosition::Absolute(z) = &self.position
            && *z > column_length
        {
            return Err(DetectorError::PositionExceedsColumn {
                position: *z,
                column_length,
            });
        }
        Ok(())
    }

    /// Returns the spatial node index closest to the detector position.
    ///
    /// Used to extract the signal from a discretised concentration profile.
    /// Utilisé pour extraire le signal depuis un profil de concentration discrétisé.
    ///
    /// # Arguments
    ///
    /// * `column_length` — column length $L$ \[m\]
    /// * `n_points`      — number of spatial nodes $N_z$
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::{Detector, DetectorPosition};
    ///
    /// // Outlet → last node
    /// assert_eq!(Detector::outlet().node_index(0.25, 100), 99);
    ///
    /// // Mid-column → node 50
    /// let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
    /// assert_eq!(det.node_index(0.25, 100), 50);
    /// ```
    pub fn node_index(&self, column_length: f64, n_points: usize) -> usize {
        let z = self.absolute_position(column_length);
        let dz = column_length / n_points as f64;
        let idx = (z / dz).round() as usize;
        idx.min(n_points - 1)
    }
}

impl Default for Detector {
    /// Default detector: column outlet.
    fn default() -> Self {
        Self::outlet()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_outlet() {
        let det = Detector::outlet();
        assert_eq!(det.position, DetectorPosition::Outlet);
    }

    #[test]
    fn test_detector_relative_valid() {
        assert!(Detector::new(DetectorPosition::Relative(0.0)).is_ok());
        assert!(Detector::new(DetectorPosition::Relative(0.5)).is_ok());
        assert!(Detector::new(DetectorPosition::Relative(1.0)).is_ok());
    }

    #[test]
    fn test_detector_relative_invalid() {
        assert!(matches!(
            Detector::new(DetectorPosition::Relative(-0.1)),
            Err(DetectorError::InvalidRelativePosition(_))
        ));
        assert!(matches!(
            Detector::new(DetectorPosition::Relative(1.1)),
            Err(DetectorError::InvalidRelativePosition(_))
        ));
    }

    #[test]
    fn test_detector_absolute_valid() {
        assert!(Detector::new(DetectorPosition::Absolute(0.1)).is_ok());
    }

    #[test]
    fn test_detector_absolute_invalid() {
        assert!(matches!(
            Detector::new(DetectorPosition::Absolute(0.0)),
            Err(DetectorError::InvalidAbsolutePosition(_))
        ));
        assert!(matches!(
            Detector::new(DetectorPosition::Absolute(-0.1)),
            Err(DetectorError::InvalidAbsolutePosition(_))
        ));
    }

    #[test]
    fn test_absolute_position_outlet() {
        assert!((Detector::outlet().absolute_position(0.25) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_absolute_position_relative() {
        let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
        assert!((det.absolute_position(0.25) - 0.125).abs() < 1e-12);
    }

    #[test]
    fn test_absolute_position_absolute() {
        let det = Detector::new(DetectorPosition::Absolute(0.1)).unwrap();
        assert!((det.absolute_position(0.25) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_validate_against_column_ok() {
        let det = Detector::new(DetectorPosition::Absolute(0.1)).unwrap();
        assert!(det.validate_against_column(0.25).is_ok());
    }

    #[test]
    fn test_validate_against_column_exceeds() {
        let det = Detector::new(DetectorPosition::Absolute(0.3)).unwrap();
        assert!(matches!(
            det.validate_against_column(0.25),
            Err(DetectorError::PositionExceedsColumn { .. })
        ));
    }

    #[test]
    fn test_validate_outlet_always_ok() {
        // Outlet and Relative are never rejected by validate_against_column
        assert!(Detector::outlet().validate_against_column(0.01).is_ok());
        let det = Detector::new(DetectorPosition::Relative(0.9)).unwrap();
        assert!(det.validate_against_column(0.01).is_ok());
    }

    #[test]
    fn test_node_index_outlet() {
        assert_eq!(Detector::outlet().node_index(0.25, 100), 99);
    }

    #[test]
    fn test_node_index_relative_half() {
        let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
        assert_eq!(det.node_index(0.25, 100), 50);
    }

    #[test]
    fn test_node_index_clamped() {
        // Position absolue légèrement supérieure au dernier nœud → clamp
        // Absolute position slightly beyond the last node → clamped to n_points - 1
        let det = Detector::new(DetectorPosition::Absolute(0.25)).unwrap();
        assert_eq!(det.node_index(0.25, 100), 99);
    }

    #[test]
    fn test_detector_default() {
        let det = Detector::default();
        assert_eq!(det.position, DetectorPosition::Outlet);
    }

    #[test]
    fn test_detector_serde_roundtrip() {
        let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
        let json = serde_json::to_string(&det).unwrap();
        let det2: Detector = serde_json::from_str(&json).unwrap();
        assert_eq!(det.position, det2.position);
    }
}
