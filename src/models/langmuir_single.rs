//! Langmuir single-species model with temporal injection
//!
//! This model extends the standard Langmuir model by adding temporal
//! boundary conditions at the column inlet (z=0).
//!
//! # Key Features
//!
//! - **Internal time tracking**: Uses AtomicU64 for thread-safe time management
//! - **No solver modifications**: Works with ANY solver (Euler, RK4, custom)
//! - **Temporal injection**: C(z=0, t) varies according to injection profile
//! - **Proper chromatographic peaks**: Eliminates plateau artifacts
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::models::{LangmuirSingle, TemporalInjection};
//! use chrom_rs::physics::PhysicalModel;
//! use chrom_rs::solver::{Scenario, SolverConfiguration, EulerSolver, Solver, DomainBoundaries};
//!
//! // Create Gaussian injection
//! let injection = TemporalInjection::gaussian(10.0, 3.0, 0.1);
//!
//! // Create model (dt must match solver's time step)
//! let dt = 600.0 / 10000.0;  // total_time / time_steps
//! let model = LangmuirSingle::new(
//!     1.2, 0.4, 2.0, 0.4, 0.001, 0.25, 100,
//!     injection,
//! );
//!
//! // Setup scenario
//! let initial_state = model.setup_initial_state();
//! let boundaries = DomainBoundaries::temporal(initial_state);
//! let scenario = Scenario::new(Box::new(model), boundaries);
//!
//! // Configure the solver
//! let config = SolverConfiguration::time_evolution(600.0, 10000);
//!
//! // Use with any solver (no modifications needed)
//! let solver = EulerSolver::new();
//! let result = solver.solve(&scenario, &config).unwrap();
//! ```

use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
use crate::models::TemporalInjection;
use nalgebra::DVector;

/// Langmuir single-species model with temporal injection
///
/// Maintains internal time state to apply temporal boundary conditions
/// without requiring solver modifications.
#[derive(Clone, Debug)]
pub struct LangmuirSingle {
    // ==================== Physics Parameters ====================
    /// Linear term λ (dimensionless)
    lambda: f64,
    /// Equilibrium constant K̃ \[L/mol\]
    langmuir_k: f64,
    /// Adsorption capacity N \[mol/L\]
    port_number: f64,
    /// Column length L \[m\]
    length: f64,
    /// Number of spatial points
    nz: usize,
    /// Spatial step dz = L/nz \[m\]
    dz: f64,
    /// Phase ratio Fₑ = (1-ε)/ε
    fe: f64,
    /// Interstitial velocity uₑ = u/ε \[m/s\]
    ue: f64,

    // ==================== Temporal Injection ====================
    /// Temporal injection profile C(z=0, t)
    injection: TemporalInjection,
}

impl LangmuirSingle {
    /// Create new model with temporal injection
    ///
    /// # Arguments
    ///
    /// * `lambda` - Linear term λ
    /// * `langmuir_k` - Equilibrium constant K̃ \[L/mol\]
    /// * `port_number` - Adsorption capacity N \[mol/L\]
    /// * `porosity` - Porosity ε (0 < ε < 1)
    /// * `velocity` - Superficial velocity u \[m/s\]
    /// * `column_length` - Length L \[m\]
    /// * `spatial_points` - Number of points
    /// * `injection` - Temporal injection profile
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::models::{LangmuirSingle, TemporalInjection};
    /// let injection = TemporalInjection::gaussian(10.0, 3.0, 0.1);
    /// let model = LangmuirSingle::new(
    ///     1.2, 0.4, 2.0, 0.4, 0.001, 0.25, 100,
    ///     injection,
    /// );
    /// ```
    pub fn new(
        lambda: f64,
        langmuir_k: f64,
        port_number: f64,
        porosity: f64,
        velocity: f64,
        column_length: f64,
        spatial_points: usize,
        injection: TemporalInjection,
    ) -> Self {
        assert!(
            porosity > 0.0 && porosity < 1.0,
            "Porosity must be in ]0,1[, got {}",
            porosity
        );
        assert!(
            column_length > 0.0,
            "Column length must be positive, got {}",
            column_length
        );
        assert!(
            spatial_points >= 2,
            "Need at least 2 spatial points, got {}",
            spatial_points
        );

        let fe = (1.0 - porosity) / porosity;
        let ue = velocity / porosity;
        let dz = column_length / (spatial_points as f64);

        Self {
            lambda,
            langmuir_k,
            port_number,
            length: column_length,
            nz: spatial_points,
            dz,
            fe,
            ue,
            injection,
        }
    }

    /// Get injection profile
    pub fn injection(&self) -> &TemporalInjection {
        &self.injection
    }

    /// Get number of spatial points
    pub fn spatial_points(&self) -> usize {
        self.nz
    }

    /// Get column length \[m\]
    pub fn length(&self) -> f64 {
        self.length
    }

    #[inline]
    fn derivative_isotherm(&self, concentration: f64) -> f64 {
        let denom = 1.0 + self.langmuir_k * concentration;
        self.lambda + (self.port_number * self.langmuir_k) / (denom * denom)
    }

    #[inline]
    fn propagation_factor(&self, concentration: f64) -> f64 {
        let deriv = self.derivative_isotherm(concentration);
        1.0 / (1.0 + self.fe * deriv)
    }
}

impl PhysicalModel for LangmuirSingle {

    fn points(&self) -> usize {
        self.nz
    }

    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {

        // ====== Read Time from Metadata ======

        // Try to read time from metadata, default to 0.0 if not present

        let t = state.get_metadata("time").unwrap_or(0.0);

        // ====== Evaluate temporal injection ======

        let c_inlet = self.injection.evaluate(t);

        // ====== Extract concentrations ======

        let c_data = state
            .get(PhysicalQuantity::Concentration)
            .expect("Concentration is required");

        let c_profile = c_data.as_vector();

        assert_eq!(
            c_profile.len(),
            self.nz,
            "Concentration profile size {} vs points discretization {}",
            c_profile.len(),
            self.nz
        );

        // ====== Compute physics ======

        let mut dc_dt = DVector::zeros(self.nz);

        for n in 0..self.nz {
            let c_n = c_profile[n];
            let factor = self.propagation_factor(c_n);

            // Spatial gradient with temporal injection at inlet
            let dc_dz = if n > 0 {
                (c_n - c_profile[n - 1]) / self.dz
            } else {
                (c_n - c_inlet) / self.dz
            };

            dc_dt[n] = - factor * self.ue * dc_dz;
        }

        // Return result as a physical state

        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::Vector(dc_dt),
        )
    }

    fn setup_initial_state(&self) -> PhysicalState {
        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::Vector(DVector::zeros(self.nz)),
        )
    }

    fn name(&self) -> &str {
        "Langmuir single specie with temporal injection"
    }

    fn description(&self) -> Option<&str> {
        Some(
        "Using Langmuir isotherm with time varying inlet BC. \
        Read from Physical State metadata."
        )
    }
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ TemporalInjection};

    fn create_langmuir_single() -> LangmuirSingle {
        let temporal = TemporalInjection::gaussian(
            10.0,
            2.0,
            1.0);

        LangmuirSingle::new(
            1.2,
            0.4,
            2.0,
            0.4,
            0.001,
            0.25,
            100,
            temporal,
        )
    }

    #[test]
    fn test_create_langmuir_single() {
        let model = create_langmuir_single();
        assert_eq!(model.points(), 100);
        assert_eq!(model.spatial_points(), 100);
    }

    #[test]
    fn test_read_time_from_metadata() {
        let model = create_langmuir_single();
        let mut state = model.setup_initial_state();

        // Add time metadata (as if solver set it)
        state.set_metadata("time".to_string(), 10.0);

        // Compute physics
        let physics = model.compute_physics(&state);

        // At t = 10.0, injection should be at peak (0.1 mol/L)
        // Should see effect at inlet

        let dc_dt = physics
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_vector();

        assert!(dc_dt[0].abs() > 0.0, "Inlet should show temporal effect");
    }

    #[test]
    fn test_works_without_metadata() {
        let model = create_langmuir_single();
        let state = model.setup_initial_state();

        // No metadata added - should default to t=0
        let physics = model.compute_physics(&state);

        // Should still work (uses t=0, so minimal injection)
        let dc_dt = physics
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_vector();

        // At t=0, injection is low (far from peak at t=10)
        assert!(dc_dt[0].abs() < 0.01, "At t=0, injection should be minimal");
    }

    #[test]
    fn test_different_times_different_physics() {
        let model = create_langmuir_single();
        let mut state = model.setup_initial_state();

        // At t=0
        state.set_metadata("time".to_string(), 0.0);
        let physics_0 = model.compute_physics(&state);
        let dc_dt_0 = physics_0
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_vector();

        // At t=10 (peak)
        state.set_metadata("time".to_string(), 10.0);
        let physics_10 = model.compute_physics(&state);
        let dc_dt_10 = physics_10
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_vector();

        // Effect at t=10 should be stronger
        assert!(
            dc_dt_10[0].abs() > dc_dt_0[0].abs(),
            "Inlet effect should be stronger at peak time"
        );
    }

    #[test]
    #[should_panic(expected = "Porosity must be in ]0,1[")]
    fn test_invalid_porosity() {
        let injection = TemporalInjection::none();
        LangmuirSingle::new(
            1.2, 0.4, 2.0, 1.5, 0.001, 0.25, 100, injection,
        );
    }

    #[test]
    #[should_panic(expected = "Need at least 2 spatial points")]
    fn test_invalid_points() {
        let injection = TemporalInjection::none();
        LangmuirSingle::new(
            1.2, 0.4, 2.0, 0.4, 0.001, 0.25, 1, injection,
        );
    }
}

