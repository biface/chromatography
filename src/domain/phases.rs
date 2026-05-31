//! Chromatographic phases.
//!
//! This module groups the types describing the phases present in a
//! chromatographic system. Only the **mobile phase** is modelled in v0.3.0;
//! additional phases (gradient elution, coupled columns) may be added here
//! in future milestones.
//!
//! # Structure
//!
//! | Type | Role |
//! |------|------|
//! | [`MobilePhase`] | Carrier fluid properties |
//!
//! # Physical background
//!
//! The **mobile phase** is the solvent flowing through the column that
//! carries the solutes. Its key parameters are the superficial velocity $u$
//! and, optionally, the dynamic viscosity $\eta$.
//!
//! The **interstitial velocity** $u_e = u / \varepsilon_e$ is the actual fluid
//! velocity between the stationary-phase particles — it governs convective
//! transport in the conservation equations.

use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Section: Mobile Phase
// =============================================================================

// -----------------------------------------------------------------------------
// MobilePhaseError
// -----------------------------------------------------------------------------

/// Errors returned by [`MobilePhase::new`] when a physical constraint is violated.
#[derive(Debug)]
pub enum MobilePhaseError {
    /// Superficial velocity must be strictly positive ($u > 0$).
    InvalidVelocity(f64),

    /// Dynamic viscosity, when provided, must be strictly positive ($\eta > 0$).
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

/// Carrier fluid properties of a chromatographic system.
///
/// `MobilePhase` encapsulates the properties of the carrier solvent. It is
/// independent of the mathematical model and serves as a validated construction
/// facade for physical models via their `from_domain` constructor.
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
/// // Interstitial velocity for ε_e = 0.4
/// assert!((mp.interstitial_velocity(0.4) - 2.5e-4).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobilePhase {
    /// Superficial velocity $u$ \[m/s\].
    ///
    /// Vitesse du solvant rapportée à la section totale de la colonne
    /// (vides + solide). Carrier velocity referred to the total column
    /// cross-section (voids + solid).
    pub velocity: f64,

    /// Dynamic viscosity $\eta$ \[Pa·s\] (optional).
    ///
    /// Requise uniquement pour les calculs de perte de charge.
    /// Required only for pressure-drop calculations.
    pub viscosity: Option<f64>,
}

impl MobilePhase {
    // =========================================================================
    // Constructor
    // =========================================================================

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

    /// Interstitial velocity $u_e = u / \varepsilon_e$ \[m/s\].
    ///
    /// Actual fluid velocity between the stationary-phase particles.
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
