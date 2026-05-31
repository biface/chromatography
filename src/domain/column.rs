//! Physical geometry of the chromatographic column.
//!
//! This module defines [`Column`], the canonical type describing the physical
//! geometry of a chromatographic column. It is independent of any mathematical
//! model and serves as a validated construction facade for
//! [`LangmuirSingle`](crate::models::LangmuirSingle),
//! [`LangmuirMulti`](crate::models::LangmuirMulti), and any future transport model.
//!
//! # Physical background
//!
//! A chromatographic column is characterised by:
//!
//! - Its **length** $L$ \[m\] — the axial dimension over which separation occurs.
//! - Its **extragranular porosity** $\varepsilon_e \in (0, 1)$ — the void fraction
//!   available to the mobile phase between the stationary-phase particles.
//! - The **number of spatial nodes** $N_z \geq 2$ — the discretisation of the
//!   axial coordinate used by the numerical solver.
//! - Optionally, its **inner diameter** $d$ \[m\] — required only when converting
//!   between volumetric flow rate $F$ \[m³/s\] and superficial velocity $u$ \[m/s\].
//!
//! # Derived quantities
//!
//! | Symbol | Formula | Meaning |
//! |--------|---------|---------|
//! | $\Delta z$ | $L / N_z$ | Spatial cell width |
//! | $F_e$ | $(1 - \varepsilon_e) / \varepsilon_e$ | Phase ratio |
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::domain::Column;
//!
//! let col = Column::new(0.25, 100, 0.4, None).unwrap();
//!
//! assert!((col.dz() - 0.0025).abs() < 1e-12);
//! assert!((col.phase_ratio() - 1.5).abs() < 1e-12);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// ColumnError
// =============================================================================

/// Errors returned by [`Column::new`] when a physical constraint is violated.
#[derive(Debug)]
pub enum ColumnError {
    /// Column length must be strictly positive ($L > 0$).
    InvalidLength(f64),

    /// Number of spatial nodes must be at least 2 ($N_z \geq 2$).
    InvalidPoints(usize),

    /// Extragranular porosity must lie in the open interval $(0, 1)$.
    InvalidPorosity(f64),

    /// Inner diameter, when provided, must be strictly positive ($d > 0$).
    InvalidDiameter(f64),
}

impl fmt::Display for ColumnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnError::InvalidLength(v) => {
                write!(f, "column: length must be > 0, got {v}")
            }
            ColumnError::InvalidPoints(n) => {
                write!(f, "column: n_points must be ≥ 2, got {n}")
            }
            ColumnError::InvalidPorosity(v) => {
                write!(f, "column: porosity must be in (0, 1), got {v}")
            }
            ColumnError::InvalidDiameter(v) => {
                write!(f, "column: diameter must be > 0, got {v}")
            }
        }
    }
}

impl std::error::Error for ColumnError {}

// =============================================================================
// Column
// =============================================================================

/// Physical geometry of a chromatographic column.
///
/// `Column` encapsulates the geometric and discretisation parameters of a
/// column. It acts as a validated construction facade: physical models
/// (`LangmuirSingle`, `LangmuirMulti`, …) accept a `&Column` in their
/// `from_domain` constructor and extract the fields they need.
///
/// # Fields
///
/// | Field | Symbol | Unit | Constraint |
/// |-------|--------|------|-----------|
/// | `column_length` | $L$ | m | $> 0$ |
/// | `n_points` | $N_z$ | — | $\geq 2$ |
/// | `porosity` | $\varepsilon_e$ | — | $(0, 1)$ |
/// | `diameter` | $d$ | m | $> 0$ if `Some` |
///
/// # Example
///
/// ```rust
/// use chrom_rs::domain::Column;
///
/// let col = Column::new(0.25, 100, 0.4, None).unwrap();
/// assert_eq!(col.n_points, 100);
/// assert!((col.phase_ratio() - 1.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    /// Column length $L$ \[m\].
    pub column_length: f64,

    /// Number of spatial discretisation nodes $N_z$.
    pub n_points: usize,

    /// Extragranular porosity $\varepsilon_e \in (0, 1)$.
    ///
    /// Fraction de vide disponible pour la phase mobile entre les grains.
    /// Void fraction available to the mobile phase between stationary-phase particles.
    pub porosity: f64,

    /// Inner diameter $d$ \[m\] (optional).
    ///
    /// Requis uniquement pour convertir un débit volumique $F$ en vitesse
    /// superficielle $u = F / (\pi d^2 / 4)$.
    /// Required only to convert volumetric flow rate to superficial velocity.
    pub diameter: Option<f64>,
}

impl Column {
    // =========================================================================
    // Constructor
    // =========================================================================

    /// Creates a validated column.
    ///
    /// # Arguments
    ///
    /// * `column_length` — $L$ \[m\], must be > 0
    /// * `n_points`      — $N_z$, must be ≥ 2
    /// * `porosity`      — $\varepsilon_e$, must be in (0, 1)
    /// * `diameter`      — $d$ \[m\], must be > 0 when `Some`
    ///
    /// # Errors
    ///
    /// Returns [`ColumnError`] if any constraint is violated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::{Column, ColumnError};
    ///
    /// assert!(Column::new(0.25, 100, 0.4, None).is_ok());
    /// assert!(matches!(Column::new(-1.0, 100, 0.4, None), Err(ColumnError::InvalidLength(_))));
    /// assert!(matches!(Column::new(0.25, 1, 0.4, None),   Err(ColumnError::InvalidPoints(_))));
    /// assert!(matches!(Column::new(0.25, 100, 1.5, None), Err(ColumnError::InvalidPorosity(_))));
    /// assert!(matches!(Column::new(0.25, 100, 0.4, Some(-0.01)), Err(ColumnError::InvalidDiameter(_))));
    /// ```
    pub fn new(
        column_length: f64,
        n_points: usize,
        porosity: f64,
        diameter: Option<f64>,
    ) -> Result<Self, ColumnError> {
        if column_length <= 0.0 {
            return Err(ColumnError::InvalidLength(column_length));
        }
        if n_points < 2 {
            return Err(ColumnError::InvalidPoints(n_points));
        }
        if porosity <= 0.0 || porosity >= 1.0 {
            return Err(ColumnError::InvalidPorosity(porosity));
        }
        if let Some(d) = diameter
            && d <= 0.0
        {
            return Err(ColumnError::InvalidDiameter(d));
        }
        Ok(Self {
            column_length,
            n_points,
            porosity,
            diameter,
        })
    }

    // =========================================================================
    // Derived accessors
    // =========================================================================

    /// Spatial cell width $\Delta z = L / N_z$ \[m\].
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::Column;
    ///
    /// let col = Column::new(0.25, 100, 0.4, None).unwrap();
    /// assert!((col.dz() - 0.0025).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn dz(&self) -> f64 {
        self.column_length / self.n_points as f64
    }

    /// Phase ratio $F_e = (1 - \varepsilon_e) / \varepsilon_e$.
    ///
    /// Ratio of stationary-phase volume to mobile-phase volume.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::Column;
    ///
    /// let col = Column::new(0.25, 100, 0.4, None).unwrap();
    /// // Fe = (1 - 0.4) / 0.4 = 1.5
    /// assert!((col.phase_ratio() - 1.5).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn phase_ratio(&self) -> f64 {
        (1.0 - self.porosity) / self.porosity
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_valid() {
        let col = Column::new(0.25, 100, 0.4, None).unwrap();
        assert_eq!(col.column_length, 0.25);
        assert_eq!(col.n_points, 100);
        assert!((col.porosity - 0.4).abs() < 1e-12);
        assert!(col.diameter.is_none());
    }

    #[test]
    fn test_column_with_diameter() {
        let col = Column::new(0.25, 100, 0.4, Some(0.01)).unwrap();
        assert!((col.diameter.unwrap() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_column_invalid_length() {
        assert!(matches!(
            Column::new(0.0, 100, 0.4, None),
            Err(ColumnError::InvalidLength(_))
        ));
        assert!(matches!(
            Column::new(-1.0, 100, 0.4, None),
            Err(ColumnError::InvalidLength(_))
        ));
    }

    #[test]
    fn test_column_invalid_points() {
        assert!(matches!(
            Column::new(0.25, 0, 0.4, None),
            Err(ColumnError::InvalidPoints(_))
        ));
        assert!(matches!(
            Column::new(0.25, 1, 0.4, None),
            Err(ColumnError::InvalidPoints(_))
        ));
    }

    #[test]
    fn test_column_invalid_porosity() {
        assert!(matches!(
            Column::new(0.25, 100, 0.0, None),
            Err(ColumnError::InvalidPorosity(_))
        ));
        assert!(matches!(
            Column::new(0.25, 100, 1.0, None),
            Err(ColumnError::InvalidPorosity(_))
        ));
        assert!(matches!(
            Column::new(0.25, 100, 1.5, None),
            Err(ColumnError::InvalidPorosity(_))
        ));
    }

    #[test]
    fn test_column_invalid_diameter() {
        assert!(matches!(
            Column::new(0.25, 100, 0.4, Some(0.0)),
            Err(ColumnError::InvalidDiameter(_))
        ));
        assert!(matches!(
            Column::new(0.25, 100, 0.4, Some(-0.01)),
            Err(ColumnError::InvalidDiameter(_))
        ));
    }

    #[test]
    fn test_dz() {
        let col = Column::new(0.25, 100, 0.4, None).unwrap();
        assert!((col.dz() - 0.0025).abs() < 1e-12);
    }

    #[test]
    fn test_phase_ratio() {
        let col = Column::new(0.25, 100, 0.4, None).unwrap();
        // Fe = (1 - 0.4) / 0.4 = 1.5
        assert!((col.phase_ratio() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_column_serde_roundtrip() {
        let col = Column::new(0.25, 100, 0.4, Some(0.01)).unwrap();
        let json = serde_json::to_string(&col).unwrap();
        let col2: Column = serde_json::from_str(&json).unwrap();
        assert_eq!(col.column_length, col2.column_length);
        assert_eq!(col.n_points, col2.n_points);
        assert!((col.porosity - col2.porosity).abs() < 1e-12);
        assert_eq!(col.diameter, col2.diameter);
    }
}
