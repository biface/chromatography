//! Reference cases for scientific validation
//!
//! Each [`ReferenceCase`] encodes a set of physical parameters and the
//! analytical predictions derived from the literature against which chrom-rs
//! results are compared.
//!
//! # Bibliographic basis
//!
//! - **Lapidus & Amundson (1952)** — analytical solution for linear
//!   advection-dispersion in a chromatographic column; provides exact
//!   expressions for the retention time and peak variance.
//! - **Felinger & Guiochon, §7.1** — dissimilarity criterion $R_{sf}$
//!   and its thresholds for model equivalence (Figures 7.1–7.3).
//!
//! # Validation strategy
//!
//! Three complementary cases are defined:
//!
//! | Case | Regime | Criterion | Reference |
//! |------|--------|-----------|-----------|
//! | [`linear_tfa`] | Linear ($C_0 \to 0$) | $t_R$ ±1 %, $\sigma$ ±10 %, mass ≥ 90 % | Lapidus-Amundson analytical |
//! | [`nonlinear_tfa`] | Non-linear ($C_0 = 0.1$) | $t_R < t_R^{lin}$, mass ≥ 85 % | Langmuir qualitative properties |
//! | Euler vs RK4 | Internal | $R_{sf} < 0.05$ | §7.1, Figure 7.1 |
//!
//! The Euler-vs-RK4 comparison lives in the test module of `main.rs`
//! rather than in a `ReferenceCase`, because it requires running two
//! simulations rather than comparing against precomputed values.

// =================================================================================================
// Physical parameters — shared by all TFA cases
// =================================================================================================

/// Column length $L$ \[m\]
pub const COLUMN_LENGTH: f64 = 0.3;
/// Interstitial porosity $\varepsilon$
pub const POROSITY: f64 = 0.4;
/// Superficial velocity $u = \varepsilon \cdot u_e$ \[m/s\]
///
/// Passed to [`LangmuirSingle::new`] which internally computes
/// $u_e = u / \varepsilon = 2.5 \times 10^{-3}$ m/s.
pub const VELOCITY: f64 = 1.0e-3;
/// Linear retention term $\lambda$
pub const LAMBDA: f64 = 1.0;
/// Langmuir equilibrium constant $\tilde{K}$ \[L/mol\]
pub const KI: f64 = 0.5;
/// Adsorption capacity $N$ (number of sites)
pub const PORT_NUMBER: f64 = 6.0;

// =================================================================================================
// Simulation parameters — shared discretisation
// =================================================================================================

/// Number of spatial discretisation points $N_z$
pub const N_POINTS: usize = 100;
/// Total simulation time \[s\] — covers $t_R + 4\sigma \approx 870$ s
pub const T_TOTAL: f64 = 900.0;
/// Number of time steps $N_t$
pub const N_STEPS: usize = 4500;

/// Injection duration for the inlet pulse \[s\]
///
/// Set to two solver time steps ($2 \Delta t$) so that all four RK4 stage
/// evaluations within step 0 and step 1 fall inside the injection window.
/// This guarantees that the injected area is exactly $C_0 \cdot T_{inj}$
/// regardless of the solver.
///
/// A Dirac delta ($t_{inj} \to 0$) is not suitable for RK4 because the
/// intermediate stage evaluations at $t + \Delta t/2$ and $t + \Delta t$
/// fall outside the single time point and return zero.
pub const T_INJ: f64 = 2.0 * T_TOTAL / N_STEPS as f64;

// =================================================================================================
// ReferenceCase
// =================================================================================================

/// Physical parameters and analytical predictions for one validation case.
pub struct ReferenceCase {
    /// Human-readable identifier used in assertion messages
    pub name: &'static str,

    // ── Injection ─────────────────────────────────────────────────────────────
    /// Inlet concentration $C_0$ \[mol/L\]
    pub c0: f64,

    // ── Analytical predictions ────────────────────────────────────────────────
    /// Expected retention time $t_R$ \[s\]
    ///
    /// Linear case: $t_R = t_0 \cdot (1 + F_e \cdot K_a^0)$ (exact).
    /// Non-linear case: approximate upper bound (actual $t_R < t_R^{lin}$).
    pub t_retention: f64,

    /// Expected peak standard deviation $\sigma_t$ \[s\]
    ///
    /// Derived from numerical dispersion of the upwind scheme
    /// (Lapidus-Amundson): $\sigma_t = \sqrt{2 D_{num} L / u_{e,eff}^3}$.
    /// `None` for non-linear cases where no analytical expression applies.
    pub sigma_analytical: Option<f64>,

    /// Minimum acceptable mass recovery $\int C_{out}\,dt \;/\; \int C_{in}\,dt$
    pub mass_recovery_min: f64,

    /// Whether the simulated $t_R$ must be **strictly less than** `t_retention`.
    ///
    /// `true` for non-linear cases: Langmuir compression pushes the peak
    /// earlier than the linear prediction.
    pub peak_before_linear_tr: bool,
}

impl ReferenceCase {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// **Case A — Linear regime** ($C_0 = 10^{-3}$ mol/L, $\tilde{K} C_0 \ll 1$)
    ///
    /// The Langmuir isotherm reduces to its linear limit; the analytical
    /// solution of Lapidus & Amundson (1952) applies exactly.
    ///
    /// Analytical predictions (upwind scheme, $N_z = 100$, $N_t = 4500$,
    /// $T = 900$ s):
    ///
    /// | Quantity | Value |
    /// |----------|-------|
    /// | $t_R$    | 624.00 s |
    /// | $\sigma_t$ | 61.39 s |
    /// | Mass recovery | ≥ 90 % |
    ///
    /// Reference: Lapidus & Amundson, *J. Phys. Chem.* 56 (1952) 984.
    pub fn linear_tfa() -> Self {
        Self {
            name: "linear_tfa",
            c0: 1e-3,
            t_retention: 624.0,
            sigma_analytical: Some(61.39),
            mass_recovery_min: 0.90,
            peak_before_linear_tr: false,
        }
    }

    /// **Case B — Non-linear regime** ($C_0 = 1.0$ mol/L, $\tilde{K} C_0 = 0.5$)
    ///
    /// Strong non-linearity: the Langmuir isotherm compresses the peak,
    /// shifting its maximum earlier than the linear prediction.
    /// Validated via the **peak mode** (position of maximum concentration)
    /// rather than the first moment, which is insensitive to Langmuir compression
    /// at moderate non-linearity.
    ///
    /// | Quantity | Criterion |
    /// |----------|-----------|
    /// | $t_{mode}$ | < $t_{mode}^{lin}$ (linear reference at same $C_0 \to 0$) |
    /// | Mass recovery | ≥ 85 % |
    pub fn nonlinear_tfa() -> Self {
        Self {
            name: "nonlinear_tfa",
            c0: 1.0,
            t_retention: 624.0, // linear upper bound — actual t_mode must be below
            sigma_analytical: None,
            mass_recovery_min: 0.85,
            peak_before_linear_tr: true,
        }
    }

    // ── Derived physical quantities ───────────────────────────────────────────

    /// Phase ratio $F_e = (1 - \varepsilon) / \varepsilon$
    pub fn fe(&self) -> f64 {
        (1.0 - POROSITY) / POROSITY
    }

    /// Interstitial velocity $u_e = u / \varepsilon$ \[m/s\]
    pub fn interstitial_velocity(&self) -> f64 {
        VELOCITY / POROSITY
    }

    /// Dead time $t_0 = L / u_e$ \[s\]
    pub fn dead_time(&self) -> f64 {
        COLUMN_LENGTH / self.interstitial_velocity()
    }

    /// Linear retention factor at infinite dilution
    ///
    /// $K_a^0 = \lambda + \bar{N} \tilde{K}$ with $\bar{N} = (1-\varepsilon) N$
    pub fn linear_ka(&self) -> f64 {
        let n_bar = (1.0 - POROSITY) * PORT_NUMBER;
        LAMBDA + n_bar * KI
    }

    /// Linear retention time $t_R^{lin} = t_0 (1 + F_e K_a^0)$ \[s\]
    pub fn linear_retention_time(&self) -> f64 {
        self.dead_time() * (1.0 + self.fe() * self.linear_ka())
    }

    /// Inlet concentration area $\int C_{in}\, dt = C_0 \cdot T_{inj}$ \[mol·s/L\]
    pub fn injected_area(&self) -> f64 {
        self.c0 * T_INJ
    }
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_tfa_derived_quantities() {
        let case = ReferenceCase::linear_tfa();
        // t0 = 0.3 / 2.5e-3 = 120 s
        assert!((case.dead_time() - 120.0).abs() < 1e-6);
        // Fe = 0.6 / 0.4 = 1.5
        assert!((case.fe() - 1.5).abs() < 1e-10);
        // Ka0 = 1.0 + 3.6*0.5 = 2.8
        assert!((case.linear_ka() - 2.8).abs() < 1e-10);
        // tR = 120 * (1 + 1.5*2.8) = 624 s
        assert!((case.linear_retention_time() - 624.0).abs() < 1e-6);
    }

    #[test]
    fn nonlinear_tfa_peak_before_linear_tr() {
        let case = ReferenceCase::nonlinear_tfa();
        assert!(case.peak_before_linear_tr);
        assert!((case.t_retention - 624.0).abs() < 1e-6);
    }

    #[test]
    fn injected_area_equals_c0_times_t_inj() {
        let lin = ReferenceCase::linear_tfa();
        assert!((lin.injected_area() - lin.c0 * T_INJ).abs() < 1e-12);
        let nl = ReferenceCase::nonlinear_tfa();
        assert!((nl.injected_area() - nl.c0 * T_INJ).abs() < 1e-12);
    }
}
