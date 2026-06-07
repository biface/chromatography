//! Scientific validation — dissimilarity protocol (DD-012)
//!
//! Launched via `cargo test --test scientific_validation`.
//!
//! Validates chrom-rs against analytical predictions from the literature
//! and against internal solver consistency, using the surface resolution
//! criterion $R_{sf}$ (Felinger & Guiochon, §7.1).
//!
//! # Test inventory
//!
//! | Test | Case | Criterion |
//! |------|------|-----------|
//! | `diagnostic_outlet_profile` | A — linear | Signal inspection (always passes) |
//! | `linear_tfa_retention_time` | A — linear | $|t_R^{sim} - t_R^{ana}| / t_R < 1\%$ |
//! | `linear_tfa_peak_width` | A — linear | $|\sigma^{sim} - \sigma^{ana}| / \sigma < 10\%$ |
//! | `linear_tfa_mass_conservation` | A — linear | $\int C_{out} / \int C_{in} \geq 90\%$ |
//! | `nonlinear_tfa_peak_compression` | B — non-linear | $t_{mode}^{NL} < t_{mode}^{lin}$ |
//! | `nonlinear_tfa_mass_conservation` | B — non-linear | $\int C_{out} / \int C_{in} \geq 85\%$ |
//! | `euler_vs_rk4_rsf` | C — internal | $R_{sf}(\text{Euler}, \text{RK4}) < 0.05$ |
//!
//! # Bugs discovered during validation
//!
//! Three physical bugs were identified and fixed in this milestone:
//!
//! 1. **`LangmuirSingle::derivative_isotherm`** — used raw `port_number` (N) instead
//!    of the effective adsorption capacity $\bar{N} = (1-\varepsilon) \cdot N$.
//!    `LangmuirMulti` stored `stationary_fraction = 1 - \varepsilon` and was correct.
//! 2. **`RK4Solver`** — all four stages shared the same `ComputeContext` built at
//!    $t = n \cdot \Delta t$. Stages k₂, k₃, k₄ must be evaluated at
//!    $t + \Delta t/2$, $t + \Delta t/2$, $t + \Delta t$ respectively.
//! 3. **`TemporalInjection::Dirac`** — approximated as a narrow Gaussian, which
//!    places half its area at $t < 0$ and is incompatible with RK4 intermediate
//!    stage evaluation. Replaced by an exact discrete equality; validation uses
//!    `Rectangle(0, 2\Delta t, C_0)` for solver-agnostic injection.

mod dissimilarity;
mod reference;

use chrom_rs::models::{LangmuirSingle, TemporalInjection};
use chrom_rs::physics::{PhysicalData, PhysicalModel, PhysicalQuantity};
use chrom_rs::solver::{
    DomainBoundaries, EulerSolver, RK4Solver, Scenario, Solver, SolverConfiguration,
};
use dissimilarity::{rsf, trapezoid};
use reference::{
    COLUMN_LENGTH, KI, LAMBDA, N_POINTS, N_STEPS, POROSITY, PORT_NUMBER, ReferenceCase, T_INJ,
    T_TOTAL, VELOCITY,
};

// =================================================================================================
// Helpers
// =================================================================================================

/// Extract $C_{outlet}(t)$: concentration at the last spatial node over the full trajectory.
fn outlet_profile(result: &chrom_rs::solver::SimulationResult, n_points: usize) -> Vec<f64> {
    result
        .state_trajectory
        .iter()
        .filter_map(|state| {
            state
                .get(PhysicalQuantity::Concentration)
                .map(|data| match data {
                    PhysicalData::Vector(v) => v[n_points - 1],
                    PhysicalData::Matrix(m) => m[(n_points - 1, 0)],
                    _ => 0.0,
                })
        })
        .collect()
}

/// Run an RK4 simulation for the TFA case with inlet concentration `c0`.
///
/// Returns `(time_points, outlet_concentrations)`.
fn run_rk4(c0: f64) -> (Vec<f64>, Vec<f64>) {
    let injection = TemporalInjection::rectangle(0.0, T_INJ, c0);
    let model = LangmuirSingle::new(
        LAMBDA,
        KI,
        PORT_NUMBER,
        POROSITY,
        VELOCITY,
        COLUMN_LENGTH,
        N_POINTS,
        injection,
    );
    let n_pts = model.points();
    let boundaries = DomainBoundaries::temporal(model.setup_initial_state());
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(T_TOTAL, N_STEPS);
    let result = RK4Solver::new()
        .solve(&scenario, &config)
        .expect("RK4 solver failed");
    let c_out = outlet_profile(&result, n_pts);
    (result.time_points, c_out)
}

/// Run an Euler simulation for the TFA case with inlet concentration `c0`.
///
/// Returns `(time_points, outlet_concentrations)`.
fn run_euler(c0: f64) -> (Vec<f64>, Vec<f64>) {
    let injection = TemporalInjection::rectangle(0.0, T_INJ, c0);
    let model = LangmuirSingle::new(
        LAMBDA,
        KI,
        PORT_NUMBER,
        POROSITY,
        VELOCITY,
        COLUMN_LENGTH,
        N_POINTS,
        injection,
    );
    let n_pts = model.points();
    let boundaries = DomainBoundaries::temporal(model.setup_initial_state());
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(T_TOTAL, N_STEPS);
    let result = EulerSolver::new()
        .solve(&scenario, &config)
        .expect("Euler solver failed");
    let c_out = outlet_profile(&result, n_pts);
    (result.time_points, c_out)
}

/// Compute the first moment (retention time) of an outlet concentration profile.
fn first_moment(times: &[f64], concs: &[f64]) -> f64 {
    let area = trapezoid(times, concs);
    let num: Vec<f64> = times.iter().zip(concs).map(|(t, c)| t * c).collect();
    trapezoid(times, &num) / area
}

/// Compute the peak standard deviation (second centred moment).
fn peak_sigma(times: &[f64], concs: &[f64]) -> f64 {
    let t_mean = first_moment(times, concs);
    let area = trapezoid(times, concs);
    let num: Vec<f64> = times
        .iter()
        .zip(concs)
        .map(|(t, c)| (t - t_mean).powi(2) * c)
        .collect();
    (trapezoid(times, &num) / area).sqrt()
}

// =================================================================================================
// Diagnostic — run first to inspect the raw outlet signal
// =================================================================================================

/// Inspect the raw RK4 outlet profile (always passes).
///
/// Prints vector lengths, peak position and amplitude, first and last
/// non-zero indices, and the mass recovery ratio.
/// Used to verify signal integrity before running quantitative assertions.
#[test]
fn diagnostic_outlet_profile() {
    let c0 = ReferenceCase::linear_tfa().c0;
    let (t_sim, c_sim) = run_rk4(c0);

    let n_t = t_sim.len();
    let n_c = c_sim.len();
    println!("=== RK4 outlet profile diagnostic ===");
    println!("time_points.len()  = {n_t}");
    println!("c_sim.len()        = {n_c}");
    println!("t[0]    = {:.4}  c[0]    = {:.4e}", t_sim[0], c_sim[0]);
    println!(
        "t[last] = {:.4}  c[last] = {:.4e}",
        t_sim[n_t - 1],
        c_sim[n_c - 1]
    );

    let peak_idx = c_sim
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "Peak: t[{peak_idx}] = {:.4} s, C = {:.4e}",
        t_sim[peak_idx.min(n_t - 1)],
        c_sim[peak_idx]
    );

    let nonzero: Vec<usize> = c_sim
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 1e-15)
        .map(|(i, _)| i)
        .collect();
    if !nonzero.is_empty() {
        println!(
            "First non-zero: i={}, t={:.2} s",
            nonzero[0],
            t_sim[nonzero[0].min(n_t - 1)]
        );
        println!(
            "Last non-zero:  i={}, t={:.2} s",
            nonzero.last().unwrap(),
            t_sim[(*nonzero.last().unwrap()).min(n_t - 1)]
        );
    }

    let area = trapezoid(&t_sim, &c_sim);
    let area_in = c0 * T_INJ;
    println!("Outlet area = {:.6e}", area);
    println!("Inlet area  = {:.6e} (c0 * T_INJ)", area_in);
    println!(
        "Mass recovery = {:.4} ({:.1} %)",
        area / area_in,
        area / area_in * 100.0
    );

    if area > 0.0 {
        let tm = first_moment(&t_sim, &c_sim);
        println!("First moment (tR) = {:.2} s", tm);
    }
}

// =================================================================================================
// Case A — Linear regime
// =================================================================================================

/// Retention time in the linear regime must match the Lapidus-Amundson prediction
/// within 1 %.
///
/// Analytical value: $t_R = t_0 \cdot (1 + F_e \cdot K_a^0) = 624.0$ s.
///
/// Reference: Lapidus & Amundson, *J. Phys. Chem.* **56** (1952) 984–988.
#[test]
fn linear_tfa_retention_time() {
    let case = ReferenceCase::linear_tfa();
    let (t_sim, c_sim) = run_rk4(case.c0);
    let t_r_sim = first_moment(&t_sim, &c_sim);
    let error_pct = (t_r_sim - case.t_retention).abs() / case.t_retention * 100.0;
    println!("Analytical tR = {:.2} s", case.t_retention);
    println!("Simulated  tR = {:.2} s", t_r_sim);
    println!("Error         = {:.3} %", error_pct);
    assert!(
        error_pct < 1.0,
        "tR error = {error_pct:.3} % ≥ 1 % for case '{}'",
        case.name
    );
}

/// Peak width (standard deviation) must match the numerical dispersion prediction
/// within 10 %.
///
/// Analytical value derived from upwind scheme numerical diffusion:
/// $\sigma_t = \sqrt{2 D_{num} L / u_{e,eff}^3} \approx 61.4$ s.
#[test]
fn linear_tfa_peak_width() {
    let case = ReferenceCase::linear_tfa();
    let sigma_ana = case.sigma_analytical.unwrap();
    let (t_sim, c_sim) = run_rk4(case.c0);
    let sigma_sim = peak_sigma(&t_sim, &c_sim);
    let error_pct = (sigma_sim - sigma_ana).abs() / sigma_ana * 100.0;
    println!("Analytical σ = {:.2} s", sigma_ana);
    println!("Simulated  σ = {:.2} s", sigma_sim);
    println!("Error        = {:.3} %", error_pct);
    assert!(
        error_pct < 10.0,
        "σ error = {error_pct:.3} % ≥ 10 % for case '{}'",
        case.name
    );
}

/// Mass recovery must be at least 90 %: the integrated outlet area must represent
/// at least 90 % of the injected quantity.
#[test]
fn linear_tfa_mass_conservation() {
    let case = ReferenceCase::linear_tfa();
    let (t_sim, c_sim) = run_rk4(case.c0);
    let area_out = trapezoid(&t_sim, &c_sim);
    let area_in = case.injected_area();
    let recovery = area_out / area_in;
    println!("Injected area = {:.6}", area_in);
    println!("Outlet area   = {:.6}", area_out);
    println!(
        "Mass recovery = {:.4} ({:.1} %)",
        recovery,
        recovery * 100.0
    );
    assert!(
        recovery >= case.mass_recovery_min,
        "mass recovery = {recovery:.4} < {:.2} for case '{}'",
        case.mass_recovery_min,
        case.name
    );
}

// =================================================================================================
// Case B — Non-linear regime
// =================================================================================================

/// Langmuir peak compression: in the non-linear regime the peak mode must arrive
/// strictly earlier than the linear reference mode.
///
/// The first moment is insensitive to Langmuir compression at moderate
/// non-linearity. The peak mode (argmax) is the correct indicator
/// (Felinger & Guiochon, §4).
#[test]
fn nonlinear_tfa_peak_compression() {
    let case = ReferenceCase::nonlinear_tfa();

    // Non-linear simulation (C0 = 1.0 mol/L)
    let (t_nl, c_nl) = run_rk4(case.c0);
    let peak_idx_nl = c_nl
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let t_mode_nl = t_nl[peak_idx_nl];

    // Linear reference at the same injection (C0 → 0)
    let (t_lin, c_lin) = run_rk4(ReferenceCase::linear_tfa().c0);
    let peak_idx_lin = c_lin
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let t_mode_lin = t_lin[peak_idx_lin];

    println!("Linear     t_mode = {:.2} s", t_mode_lin);
    println!("Non-linear t_mode = {:.2} s", t_mode_nl);
    println!("Compression       = {:.2} s", t_mode_lin - t_mode_nl);

    assert!(
        t_mode_nl < t_mode_lin,
        "t_mode NL = {t_mode_nl:.2} s should be < t_mode lin = {t_mode_lin:.2} s \
         for case '{}'",
        case.name
    );
}

/// Mass recovery in the non-linear regime must be at least 85 %.
#[test]
fn nonlinear_tfa_mass_conservation() {
    let case = ReferenceCase::nonlinear_tfa();
    let (t_sim, c_sim) = run_rk4(case.c0);
    let area_out = trapezoid(&t_sim, &c_sim);
    let area_in = case.injected_area();
    let recovery = area_out / area_in;
    println!(
        "Mass recovery = {:.4} ({:.1} %)",
        recovery,
        recovery * 100.0
    );
    assert!(
        recovery >= case.mass_recovery_min,
        "mass recovery = {recovery:.4} < {:.2} for case '{}'",
        case.mass_recovery_min,
        case.name
    );
}

// =================================================================================================
// Case C — Internal solver consistency: Euler vs RK4
// =================================================================================================

/// Euler and RK4 must produce scientifically indistinguishable chromatograms
/// on the linear TFA case: $R_{sf} < 0.05$ (Figure 7.1, Felinger & Guiochon).
///
/// A value $R_{sf} \geq 0.05$ would indicate a solver-level divergence
/// requiring investigation independent of the physical model.
#[test]
fn euler_vs_rk4_rsf() {
    let c0 = ReferenceCase::linear_tfa().c0;
    let (t_euler, c_euler) = run_euler(c0);
    let (t_rk4, c_rk4) = run_rk4(c0);
    let criterion = rsf(&t_euler, &c_euler, &t_rk4, &c_rk4);
    println!("Rsf(Euler, RK4) = {criterion:.4}");
    assert!(
        criterion < 0.05,
        "Rsf(Euler, RK4) = {criterion:.4} ≥ 0.05 — solver divergence detected"
    );
}
