//! Colonne chromatographique — physical geometry of the chromatographic column.
//!
//! Ce module définit [`Column`], le type canonique décrivant la géométrie
//! physique d'une colonne chromatographique. Il est indépendant de tout
//! modèle mathématique et sert de couche de construction validée pour
//! [`LangmuirSingle`](crate::models::LangmuirSingle) et
//! [`LangmuirMulti`](crate::models::LangmuirMulti), ainsi que pour tout
//! futur modèle de transport.
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

/// Erreurs de construction d'une [`Column`] — Column construction errors.
#[derive(Debug)]
pub enum ColumnError {
    /// La longueur doit être strictement positive — column length must be > 0.
    InvalidLength(f64),

    /// Le nombre de points doit être ≥ 2 — spatial points must be ≥ 2.
    InvalidPoints(usize),

    /// La porosité doit appartenir à (0, 1) — porosity must be in (0, 1).
    InvalidPorosity(f64),

    /// Le diamètre, s'il est fourni, doit être strictement positif.
    /// Diameter, when provided, must be > 0.
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

/// Géométrie physique d'une colonne chromatographique.
///
/// `Column` encapsule les paramètres géométriques et de discrétisation d'une
/// colonne. Il sert de façade de construction validée : les modèles physiques
/// (`LangmuirSingle`, `LangmuirMulti`, …) acceptent une `&Column` dans leur
/// constructeur `from_domain` et en extraient les champs nécessaires.
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
    /// Longueur de la colonne $L$ \[m\] — column length.
    pub column_length: f64,

    /// Nombre de nœuds spatiaux $N_z$ — number of spatial discretisation nodes.
    pub n_points: usize,

    /// Porosité extragranulaire $\varepsilon_e \in (0, 1)$ — extragranular porosity.
    pub porosity: f64,

    /// Diamètre interne $d$ \[m\] — inner diameter (optional).
    ///
    /// Requis uniquement pour convertir un débit volumique $F$ en vitesse
    /// superficielle $u$. Required only to convert volumetric flow rate to
    /// superficial velocity.
    pub diameter: Option<f64>,
}

impl Column {
    // =========================================================================
    // Constructor
    // =========================================================================

    /// Construit une colonne après validation des paramètres.
    ///
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

    /// Largeur d'une cellule spatiale $\Delta z = L / N_z$ \[m\].
    ///
    /// Spatial cell width.
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

    /// Rapport de phase $F_e = (1 - \varepsilon_e) / \varepsilon_e$.
    ///
    /// Phase ratio — ratio of stationary-phase volume to mobile-phase volume.
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
