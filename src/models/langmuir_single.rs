//! Langmuir single-species model with temporal injection
//!
//! # Physical background
//!
//! In liquid chromatography, a solute injected at the column inlet ($z = 0$)
//! is transported by the mobile phase while partitioning between the mobile
//! and stationary phases. The **Langmuir isotherm** describes this equilibrium:
//! adsorption sites on the stationary phase are finite, so retention decreases
//! as concentration increases — producing the characteristic asymmetric
//! (Langmuir-type) chromatographic peaks.
//!
//! # Model equations
//!
//! ## Langmuir isotherm
//!
//! The stationary phase concentration $\bar{C}$ in equilibrium with the mobile
//! phase concentration $C$:
//!
//! $$\bar{C} = \lambda \cdot C + \frac{\bar{N} \cdot \tilde{K} \cdot C}{1 + \tilde{K} \cdot C}$$
//!
//! where $\bar{N} = (1 - \varepsilon) \cdot N$ is the effective adsorption capacity.
//!
//! ## Isotherm derivative
//!
//! The derivative $\partial \bar{C} / \partial C$ governs how adsorption slows
//! down the propagation of the concentration front:
//!
//! $$\frac{\partial \bar{C}}{\partial C} = \lambda + \frac{\bar{N} \cdot \tilde{K}}{(1 + \tilde{K} \cdot C)^2}$$
//!
//! ## Propagation factor
//!
//! The effective velocity of a concentration front relative to the interstitial
//! velocity $u_e$ is reduced by the propagation factor:
//!
//! $$\sigma(C) = \frac{1}{1 + F_e \cdot \partial \bar{C} / \partial C}$$
//!
//! with $F_e = (1 - \varepsilon) / \varepsilon$.
//!
//! ## Transport equation
//!
//! $$\frac{\partial C}{\partial t} = -\sigma(C) \cdot u_e \cdot \frac{\partial C}{\partial z}$$
//!
//! with $u_e = u / \varepsilon$ the interstitial velocity.
//!
//! # Spatial discretisation
//!
//! Concentrations are stored in a vector of length $N_z$:
//! - **Elements**: spatial points $z_0, z_1, \ldots, z_{N_z - 1}$
//!
//! The spatial gradient uses an **upwind scheme** (backward difference),
//! unconditionally stable for left-to-right convection ($u_e > 0$):
//!
//! $$\left.\frac{\partial C}{\partial z}\right\vert_{z_i} \approx \frac{C_i - C_{i-1}}{\Delta z}$$
//!
//! **Finite volume convention**: $\Delta z = L / N_z$ (cell width, not node spacing).
//! Each cell $i$ spans $[i \cdot \Delta z,\ (i+1) \cdot \Delta z]$.
//!
//! # Injection strategy
//!
//! Injection is modelled as a **temporal boundary condition** at the column inlet ($z = 0$).
//! The column starts empty at $t = 0$ and concentration enters at each time step via
//! `injection.evaluate(t)`, where `t` is read from the `PhysicalState` metadata.
//!
//! The [`TemporalInjection`] profile is evaluated at the fictitious upstream cell
//! just left of $z_0$, providing the upstream concentration for the upwind gradient.
//! This works with any solver that writes `"time"` into the state metadata before
//! each call to `compute_physics`.
//!
//! # Example
//!
//! ```rust,ignore
//! use chrom_rs::models::{LangmuirSingle, TemporalInjection};
//! use chrom_rs::physics::PhysicalModel;
//! use chrom_rs::solver::{Scenario, SolverConfiguration, EulerSolver, Solver, DomainBoundaries};
//!
//! // Gaussian injection: peak at t=10s, width 3s, max concentration 0.1 mol/L
//! let injection = TemporalInjection::gaussian(10.0, 3.0, 0.1);
//!
//! let model = LangmuirSingle::new(
//!     1.2,   // λ  [dimensionless]
//!     0.4,   // K̃  [L/mol]
//!     2.0,   // N  [dimensionless]
//!     0.4,   // ε  (porosity)
//!     0.001, // u  [m/s]
//!     0.25,  // L  [m]
//!     100,   // N_z spatial points
//!     injection,
//! );
//!
//! let initial_state = model.setup_initial_state();
//! let boundaries = DomainBoundaries::temporal(initial_state);
//! let scenario = Scenario::new(Box::new(model), boundaries);
//! let config = SolverConfiguration::time_evolution(600.0, 10000);
//!
//! let solver = EulerSolver::new();
//! let result = solver.solve(&scenario, &config).unwrap();
//! ```

use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
use crate::models::TemporalInjection;
use nalgebra::DVector;

// =================================================================================================
// LangmuirSingle
// =================================================================================================

/// Langmuir single-species model with temporal injection
///
/// Models the chromatographic transport of a single solute through a column
/// using the Langmuir isotherm and an upwind finite-volume scheme.
///
/// # Parameters
///
/// | Field        | Symbol      | Unit              | Role                                      |
/// |--------------|-------------|-------------------|-------------------------------------------|
/// | `lambda`     | $\lambda$   | dimensionless     | Residual linear retention                 |
/// | `langmuir_k` | $\tilde{K}$ | $\text{L/mol}$    | Affinity for the stationary phase         |
/// | `port_number`| $N$         | dimensionless     | Maximum adsorption capacity               |
/// | `fe`         | $F_e$       | dimensionless     | Phase ratio $(1-\varepsilon)/\varepsilon$ |
/// | `ue`         | $u_e$       | $\text{m/s}$      | Interstitial velocity $u/\varepsilon$     |
/// | `dz`         | $\Delta z$  | $\text{m}$        | Cell width $L / N_z$                      |
///
/// # State layout
///
/// The physical state is a vector of length $N_z$:
///
/// ```text
/// z_0    z_1    z_2   ...   z_{Nz-1}
/// [C_0,  C_1,   C_2,  ...,  C_{Nz-1}]
/// ```
///
/// The **column outlet** (chromatogram signal) is the last element: `C[nz-1]`.
///
/// # Example
///
/// ```rust
/// use chrom_rs::models::{LangmuirSingle, TemporalInjection};
/// use chrom_rs::physics::PhysicalModel;
///
/// let model = LangmuirSingle::new(
///     1.2, 0.4, 2.0, 0.4, 0.001, 0.25, 100,
///     TemporalInjection::dirac(0.0, 1e-3),
/// );
/// assert_eq!(model.length(), 0.25);
/// assert_eq!(model.spatial_points(), 100);
/// ```
#[derive(Clone, Debug)]
pub struct LangmuirSingle {
    // ── Isotherm parameters ───────────────────────────────────────────────────

    /// Linear retention term $\lambda$ **\[dimensionless\]**, must be $\geq 0$
    ///
    /// Represents residual linear retention at low concentrations.
    /// $\lambda = 0$ gives a pure Langmuir isotherm with no linear term.
    lambda: f64,

    /// Langmuir equilibrium constant $\tilde{K}$ **\[L/mol\]**, must be $> 0$
    ///
    /// Controls affinity for the stationary phase. Higher $\tilde{K}$ →
    /// stronger retention → later elution
    langmuir_k: f64,

    /// Adsorption capacity $N$ **\[dimensionless\]**, must be $> 0$
    ///
    /// Maximum number of sites available on the stationary phase.
    /// The effective capacity $\bar{N} = (1 - \varepsilon) \cdot N$ is
    /// computed inside [`derivative_isotherm`](Self::derivative_isotherm).
    port_number: f64,

    // ── Column geometry ───────────────────────────────────────────────────────

    /// Column length $L$ **\[m\]**
    length: f64,

    /// Number of spatial discretization points $N_z$
    nz: usize,

    /// Cell width $\Delta z = L / N_z$ **\[m\]** — precomputed
    ///
    /// Finite volume convention: $N_z$ cells of equal width,
    /// boundaries at $0, \Delta z, 2\Delta z, \ldots, L$.
    dz: f64,

    // ── Derived transport quantities ──────────────────────────────────────────

    /// Phase ratio $F_e = (1 - \varepsilon) / \varepsilon$ **\[dimensionless\]** — precomputed
    fe: f64,

    /// Interstitial velocity $u_e = u / \varepsilon$ **\[m/s\]** — precomputed
    ue: f64,

    // ── Injection ─────────────────────────────────────────────────────────────

    /// Temporal injection profile $C(z=0,\ t)$ at the column inlet
    injection: TemporalInjection,
}

impl LangmuirSingle {
    /// Creates a new single-species Langmuir model
    ///
    /// Precomputes $F_e$, $u_e$, and $\Delta z$ at construction to avoid
    /// repeated divisions in the time integration loop.
    ///
    /// # Arguments
    ///
    /// * `lambda`         — $\lambda \geq 0$ **\[dimensionless\]**
    /// * `langmuir_k`     — $\tilde{K} > 0$ **\[L/mol\]**
    /// * `port_number`    — $N > 0$ **\[dimensionless\]**
    /// * `porosity`       — $\varepsilon \in (0, 1)$ **\[dimensionless\]**
    /// * `velocity`       — superficial velocity $u > 0$ **\[m/s\]**
    /// * `column_length`  — $L > 0$ **\[m\]**
    /// * `spatial_points` — $N_z \geq 2$
    /// * `injection`      — temporal profile at $z = 0$
    ///
    /// # Panics
    ///
    /// - If `porosity` is not in $(0, 1)$
    /// - If `column_length` is not positive
    /// - If `spatial_points < 2`
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::models::{LangmuirSingle, TemporalInjection};
    ///
    /// let model = LangmuirSingle::new(
    ///     1.2, 0.4, 2.0, 0.4, 0.001, 0.25, 100,
    ///     TemporalInjection::gaussian(10.0, 3.0, 0.1),
    /// );
    /// assert_eq!(model.spatial_points(), 100);
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

    /// Returns the temporal injection profile at the column inlet
    pub fn injection(&self) -> &TemporalInjection {
        &self.injection
    }

    /// Returns the number of spatial discretisation points $N_z$
    pub fn spatial_points(&self) -> usize {
        self.nz
    }

    /// Returns the column length $L$ **\[m\]**
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Computes the isotherm derivative $\partial \bar{C} / \partial C$
    ///
    /// $$\frac{\partial \bar{C}}{\partial C} = \lambda + \frac{\bar{N} \cdot \tilde{K}}{(1 + \tilde{K} \cdot C)^2}$$
    ///
    /// where $\bar{N} = (1 - \varepsilon) \cdot N = F_e \cdot \varepsilon \cdot N$
    /// is approximated here as $N$ directly — the phase ratio $F_e$ is applied
    /// in [`propagation_factor`](Self::propagation_factor).
    ///
    /// # Note on concentration dependence
    ///
    /// At $C = 0$ (dilute limit), the derivative equals $\lambda + N \tilde{K}$ —
    /// the Henry constant. As $C \to \infty$, it tends to $\lambda$ (linear regime).
    /// This concentration dependence is what produces asymmetric chromatographic peaks.
    #[inline]
    fn derivative_isotherm(&self, concentration: f64) -> f64 {
        let denom = 1.0 + self.langmuir_k * concentration;
        self.lambda + (self.port_number * self.langmuir_k) / (denom * denom)
    }

    /// Computes the propagation factor $\sigma(C)$
    ///
    /// $$\sigma(C) = \frac{1}{1 + F_e \cdot \partial \bar{C} / \partial C}$$
    ///
    /// $\sigma$ lies in $(0, 1)$ and represents the effective velocity of a
    /// concentration front relative to $u_e$. A high adsorption derivative
    /// (strong retention) gives a small $\sigma$ — the front moves slowly.
    ///
    /// This is the scalar equivalent of the matrix $(I + F_e \cdot M)^{-1}$
    /// used in [`LangmuirMulti`](crate::models::LangmuirMulti).
    #[inline]
    fn propagation_factor(&self, concentration: f64) -> f64 {
        let deriv = self.derivative_isotherm(concentration);
        1.0 / (1.0 + self.fe * deriv)
    }
}


// =================================================================================================
// PhysicalModel implementation
// =================================================================================================

impl PhysicalModel for LangmuirSingle {

    /// Returns the number of spatial discretisation points $N_z$
    fn points(&self) -> usize {
        self.nz
    }

    /// Computes $\partial C / \partial t$ at all spatial points
    ///
    /// Implements the transport equation:
    ///
    /// $$\frac{\partial C_i}{\partial t} = -\sigma(C_i) \cdot u_e \cdot \frac{\partial C}{\partial z}\bigg|_i$$
    ///
    /// For each spatial point $i$:
    ///
    /// 1. Read current time $t$ from state metadata (default $0.0$)
    /// 2. Evaluate inlet concentration $C_\text{upstream} = \texttt{injection.evaluate}(t)$
    /// 3. Compute upwind gradient $\partial C / \partial z$
    /// 4. Compute propagation factor $\sigma(C_i)$
    /// 5. Apply $\partial C_i / \partial t = -\sigma(C_i) \cdot u_e \cdot \nabla_z C_i$
    ///
    /// **Left boundary** ($i = 0$): the upstream concentration is
    /// `injection.evaluate(t)` — the sole entry point of material into the column.
    ///
    /// **Interior and right boundary** ($i > 0$): standard backward difference
    /// using the real upstream neighbor $C_{i-1}$.
    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {

        // ── Time ──────────────────────────────────────────────────────────────
        //
        // The solver writes the current simulation time into the state metadata
        // before each call. If absent (e.g. solver does not support metadata),
        // defaults to t=0.0 — injection profiles return their value at t=0.

        let t = state.get_metadata("time").unwrap_or(0.0);

        // ── Inlet concentration ───────────────────────────────────────────────
        //
        // Evaluates the injection profile at the current time. This is the
        // fictitious upstream concentration just left of z_0, used as the
        // boundary condition for the upwind gradient at i=0.

        let c_inlet = self.injection.evaluate(t);

        // ── State extraction ──────────────────────────────────────────────────
        //
        // The concentration profile is a vector of length N_z.
        // Elements are indexed by spatial point: c_profile[i] = C(z_i, t).

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

        // ── Transport loop ────────────────────────────────────────────────────
        //
        // For each spatial point i, compute dC/dt using the upwind scheme.
        // The propagation factor σ(C_i) depends on the local concentration,
        // making this a nonlinear PDE — hence the point-by-point evaluation.

        let mut dc_dt = DVector::zeros(self.nz);

        for n in 0..self.nz {
            let c_n = c_profile[n];

            // Propagation factor: σ(C) = 1 / (1 + Fe · ∂C̄/∂C)
            // Scalar equivalent of (I + Fe·M)^{-1} in LangmuirMulti.
            let sigma = self.propagation_factor(c_n);

            // Upwind gradient ∂C/∂z — finite volume convention (Δz = L/Nz):
            //   i=0 : fictitious upstream cell = injection.evaluate(t)
            //   i>0 : real upstream neighbor C_{i-1}
            let dc_dz = if n > 0 {
                (c_n - c_profile[n - 1]) / self.dz
            } else {
                (c_n - c_inlet) / self.dz
            };

            // dC/dt = -σ(C) · u_e · ∂C/∂z
            dc_dt[n] = -sigma * self.ue * dc_dz;
        }

        // Return result as a physical state

        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::Vector(dc_dt),
        )
    }

    /// Initialises the column state to zero (empty column)
    ///
    /// The column is empty at $t = 0$. Concentration enters from the left
    /// boundary at each time step via `injection.evaluate(t)` in `compute_physics`.
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

