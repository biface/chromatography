//! Détecteur chromatographique — chromatographic signal detector.
//!
//! Ce module définit [`Detector`] et [`DetectorPosition`], qui spécifient
//! l'emplacement du détecteur le long de l'axe de la colonne.
//!
//! # Physical background
//!
//! En chromatographie, le signal mesuré est la concentration en phase mobile
//! au point de détection $z_d$. La position par défaut est la sortie de
//! colonne ($z_d = L$), mais certains montages expérimentaux placent le
//! détecteur en un point intermédiaire.
//!
//! # Position variants
//!
//! | Variante | Description | Contrainte |
//! |----------|-------------|-----------|
//! | [`Outlet`](DetectorPosition::Outlet) | Sortie de colonne $z_d = L$ | aucune |
//! | [`Relative`](DetectorPosition::Relative) | $z_d = r \cdot L$, $r \in [0, 1]$ | $r \in [0, 1]$ |
//! | [`Absolute`](DetectorPosition::Absolute) | $z_d$ \[m\] | $z_d > 0$, validé contre $L$ à l'usage |
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::domain::{Detector, DetectorPosition};
//!
//! // Détecteur en sortie de colonne (comportement actuel)
//! let det = Detector::outlet();
//! assert!(matches!(det.position, DetectorPosition::Outlet));
//!
//! // Détecteur à mi-colonne
//! let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
//! assert!((det.absolute_position(0.25) - 0.125).abs() < 1e-12);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// DetectorError
// =============================================================================

/// Erreurs de construction d'un [`Detector`] — Detector construction errors.
#[derive(Debug)]
pub enum DetectorError {
    /// La position relative doit appartenir à \[0, 1\].
    /// Relative position must be in \[0, 1\].
    InvalidRelativePosition(f64),

    /// La position absolue doit être strictement positive.
    /// Absolute position must be > 0.
    InvalidAbsolutePosition(f64),

    /// La position absolue dépasse la longueur de la colonne.
    /// Absolute position exceeds column length.
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

/// Position du détecteur le long de l'axe de la colonne.
///
/// Detector position along the column axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DetectorPosition {
    /// Sortie de colonne $z_d = L$ — column outlet (default behaviour).
    Outlet,

    /// Position relative $z_d = r \cdot L$, $r \in [0, 1]$.
    ///
    /// Relative position as a fraction of the column length.
    Relative(f64),

    /// Position absolue $z_d$ \[m\] — absolute axial position.
    ///
    /// Doit être validée contre la longueur de la colonne avant usage.
    /// Must be validated against the column length before use.
    Absolute(f64),
}

// =============================================================================
// Detector
// =============================================================================

/// Détecteur chromatographique à position configurable.
///
/// `Detector` encapsule la position du point de mesure du signal. La position
/// par défaut est la sortie de colonne ([`DetectorPosition::Outlet`]),
/// reproduisant le comportement actuel de `outlet_data`.
///
/// # Example
///
/// ```rust
/// use chrom_rs::domain::{Detector, DetectorPosition};
///
/// // Sortie de colonne (défaut)
/// let det = Detector::outlet();
///
/// // Mi-colonne
/// let det = Detector::new(DetectorPosition::Relative(0.5)).unwrap();
/// // Pour L = 0.25 m : z_d = 0.125 m
/// assert!((det.absolute_position(0.25) - 0.125).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detector {
    /// Position du détecteur — detector position.
    pub position: DetectorPosition,
}

impl Detector {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Construit un détecteur après validation de la position.
    ///
    /// Creates a validated detector.
    ///
    /// # Errors
    ///
    /// Returns [`DetectorError`] if:
    /// - `Relative(r)` with $r \notin [0, 1]$
    /// - `Absolute(z)` with $z \leq 0$
    ///
    /// Note : `Absolute(z)` n'est pas validé contre la longueur de colonne ici
    /// (elle n'est pas connue à ce stade). Use [`validate_against_column`](Self::validate_against_column)
    /// to check compatibility at simulation time.
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

    /// Construit un détecteur en sortie de colonne (comportement par défaut).
    ///
    /// Creates an outlet detector (default behaviour — equivalent to current
    /// `outlet_data` usage).
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

    /// Retourne la position absolue $z_d$ \[m\] pour une longueur de colonne $L$ donnée.
    ///
    /// Returns the absolute axial position $z_d$ \[m\] for a given column length $L$.
    ///
    /// | Variante | Résultat |
    /// |----------|---------|
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

    /// Valide la position absolue contre la longueur de la colonne.
    ///
    /// Validates an `Absolute` position against the column length. No-op for
    /// `Outlet` and `Relative` variants.
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

    /// Retourne l'indice spatial du nœud le plus proche de la position du détecteur.
    ///
    /// Returns the spatial node index closest to the detector position.
    ///
    /// Utilisé pour extraire le signal depuis un profil de concentration discrétisé.
    /// Used to extract the signal from a discretised concentration profile.
    ///
    /// # Arguments
    ///
    /// * `column_length` — longueur de la colonne $L$ \[m\]
    /// * `n_points`      — nombre de nœuds spatiaux $N_z$
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::{Detector, DetectorPosition};
    ///
    /// // Sortie de colonne → dernier nœud
    /// assert_eq!(Detector::outlet().node_index(0.25, 100), 99);
    ///
    /// // Mi-colonne → nœud 50
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
    /// Détecteur par défaut : sortie de colonne.
    ///
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
        // Outlet et Relative ne sont jamais rejetés par validate_against_column
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
