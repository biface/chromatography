//! Phases chromatographiques — chromatographic phases.
//!
//! Ce module regroupe les types décrivant les phases présentes dans un
//! système chromatographique. Seule la **phase mobile** est modélisée en
//! v0.3.0 ; d'autres phases (gradient d'élution, phases couplées) pourront
//! être ajoutées ici dans les jalons futurs.
//!
//! # Structure
//!
//! | Type | Rôle |
//! |------|------|
//! | [`MobilePhase`] | Solvant porteur — carrier fluid properties |
//!
//! # Physical background
//!
//! La **phase mobile** est le solvant qui circule à travers la colonne et
//! transporte les solutés. Ses paramètres clés sont la vitesse superficielle
//! $u$ et, optionnellement, la viscosité dynamique $\eta$.
//!
//! La **vitesse interstitielle** $u_e = u / \varepsilon_e$ est la vitesse
//! réelle du fluide entre les grains — elle gouverne le transport convectif
//! dans les équations de conservation.

use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Section : Mobile Phase
// =============================================================================

// -----------------------------------------------------------------------------
// MobilePhaseError
// -----------------------------------------------------------------------------

/// Erreurs de construction d'une [`MobilePhase`] — MobilePhase construction errors.
#[derive(Debug)]
pub enum MobilePhaseError {
    /// La vitesse superficielle doit être strictement positive.
    /// Superficial velocity must be > 0.
    InvalidVelocity(f64),

    /// La viscosité, si fournie, doit être strictement positive.
    /// Viscosity, when provided, must be > 0.
    InvalidViscosity(f64),
}

impl fmt::Display for MobilePhaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MobilePhaseError::InvalidVelocity(v) => {
                write!(f, "mobile phase: velocity must be > 0, got {v}")
            }
            MobilePhaseError::InvalidViscosity(v) => {
                write!(f, "mobile phase: viscosity must be > 0, got {v}")
            }
        }
    }
}

impl std::error::Error for MobilePhaseError {}

// -----------------------------------------------------------------------------
// MobilePhase
// -----------------------------------------------------------------------------

/// Phase mobile d'un système chromatographique.
///
/// `MobilePhase` encapsule les propriétés du solvant porteur. Elle est
/// indépendante du modèle mathématique et sert de façade de construction
/// validée pour les modèles physiques via leur constructeur `from_domain`.
///
/// # Fields
///
/// | Field | Symbol | Unit | Constraint |
/// |-------|--------|------|-----------|
/// | `velocity` | $u$ | m/s | $> 0$ |
/// | `viscosity` | $\eta$ | Pa·s | $> 0$ if `Some` |
///
/// # Example
///
/// ```rust
/// use chrom_rs::domain::MobilePhase;
///
/// let mp = MobilePhase::new(1e-4, None).unwrap();
/// // Vitesse interstitielle pour ε_e = 0.4
/// assert!((mp.interstitial_velocity(0.4) - 2.5e-4).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobilePhase {
    /// Vitesse superficielle $u$ \[m/s\] — superficial velocity.
    ///
    /// Vitesse du solvant rapportée à la section totale de la colonne
    /// (vides + solide). The carrier velocity referred to the total
    /// column cross-section.
    pub velocity: f64,

    /// Viscosité dynamique $\eta$ \[Pa·s\] — dynamic viscosity (optional).
    ///
    /// Requise uniquement pour les calculs de perte de charge. Required
    /// only for pressure-drop calculations.
    pub viscosity: Option<f64>,
}

impl MobilePhase {
    // =========================================================================
    // Constructor
    // =========================================================================

    /// Construit une phase mobile après validation des paramètres.
    ///
    /// Creates a validated mobile phase.
    ///
    /// # Arguments
    ///
    /// * `velocity`  — $u$ \[m/s\], must be > 0
    /// * `viscosity` — $\eta$ \[Pa·s\], must be > 0 when `Some`
    ///
    /// # Errors
    ///
    /// Returns [`MobilePhaseError`] if any constraint is violated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::{MobilePhase, MobilePhaseError};
    ///
    /// assert!(MobilePhase::new(1e-4, None).is_ok());
    /// assert!(MobilePhase::new(1e-4, Some(1e-3)).is_ok());
    /// assert!(matches!(
    ///     MobilePhase::new(-1e-4, None),
    ///     Err(MobilePhaseError::InvalidVelocity(_))
    /// ));
    /// assert!(matches!(
    ///     MobilePhase::new(1e-4, Some(-1e-3)),
    ///     Err(MobilePhaseError::InvalidViscosity(_))
    /// ));
    /// ```
    pub fn new(velocity: f64, viscosity: Option<f64>) -> Result<Self, MobilePhaseError> {
        if velocity <= 0.0 {
            return Err(MobilePhaseError::InvalidVelocity(velocity));
        }
        if let Some(eta) = viscosity
            && eta <= 0.0
        {
            return Err(MobilePhaseError::InvalidViscosity(eta));
        }
        Ok(Self {
            velocity,
            viscosity,
        })
    }

    // =========================================================================
    // Derived accessors
    // =========================================================================

    /// Vitesse interstitielle $u_e = u / \varepsilon_e$ \[m/s\].
    ///
    /// Interstitial velocity — the actual fluid velocity between particles.
    ///
    /// # Arguments
    ///
    /// * `porosity` — extragranular porosity $\varepsilon_e \in (0, 1)$
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::MobilePhase;
    ///
    /// let mp = MobilePhase::new(1e-4, None).unwrap();
    /// // u_e = 1e-4 / 0.4 = 2.5e-4
    /// assert!((mp.interstitial_velocity(0.4) - 2.5e-4).abs() < 1e-15);
    /// ```
    #[inline]
    pub fn interstitial_velocity(&self, porosity: f64) -> f64 {
        self.velocity / porosity
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // MobilePhase
    // -------------------------------------------------------------------------

    #[test]
    fn test_mobile_phase_valid() {
        let mp = MobilePhase::new(1e-4, None).unwrap();
        assert!((mp.velocity - 1e-4).abs() < 1e-15);
        assert!(mp.viscosity.is_none());
    }

    #[test]
    fn test_mobile_phase_with_viscosity() {
        let mp = MobilePhase::new(1e-4, Some(1e-3)).unwrap();
        assert!((mp.viscosity.unwrap() - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn test_mobile_phase_invalid_velocity() {
        assert!(matches!(
            MobilePhase::new(0.0, None),
            Err(MobilePhaseError::InvalidVelocity(_))
        ));
        assert!(matches!(
            MobilePhase::new(-1e-4, None),
            Err(MobilePhaseError::InvalidVelocity(_))
        ));
    }

    #[test]
    fn test_mobile_phase_invalid_viscosity() {
        assert!(matches!(
            MobilePhase::new(1e-4, Some(0.0)),
            Err(MobilePhaseError::InvalidViscosity(_))
        ));
        assert!(matches!(
            MobilePhase::new(1e-4, Some(-1e-3)),
            Err(MobilePhaseError::InvalidViscosity(_))
        ));
    }

    #[test]
    fn test_interstitial_velocity() {
        let mp = MobilePhase::new(1e-4, None).unwrap();
        // u_e = 1e-4 / 0.4 = 2.5e-4
        assert!((mp.interstitial_velocity(0.4) - 2.5e-4).abs() < 1e-15);
    }

    #[test]
    fn test_mobile_phase_serde_roundtrip() {
        let mp = MobilePhase::new(1e-4, Some(1e-3)).unwrap();
        let json = serde_json::to_string(&mp).unwrap();
        let mp2: MobilePhase = serde_json::from_str(&json).unwrap();
        assert!((mp.velocity - mp2.velocity).abs() < 1e-15);
        assert!((mp.viscosity.unwrap() - mp2.viscosity.unwrap()).abs() < 1e-15);
    }
}
