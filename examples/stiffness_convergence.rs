//! Δt convergence sweep — testing the stiffness hypothesis (issue #55).
//!
//! Runs both real reference cases (Ascorbic/Erythorbic, Glucose/Fructose)
//! at several time-step resolutions — the baseline `n_steps` used in
//! `validation/reference.rs`, then ×2, ×5, ×10, ×25, ×50 — and tracks
//! $R_{sf}$(Euler, RK4) at each resolution. If glucose/fructose has a
//! genuinely stiffer adsorption front (see the "Perspective" section of
//! the wiki write-up), its $R_{sf}$ should stay high over a wider range of
//! Δt before collapsing, compared to ascorbic/erythorbic.
//!
//! # Why the reference values and $R_{sf}$ are duplicated here
//!
//! Same reason as `examples/validation_report.rs`: `validation/` is a
//! standalone `[[test]]` crate, not part of the public `chrom-rs` library —
//! examples cannot depend on it. See that file's header for the full
//! rationale; the helpers below (`trapezoid`, `normalize`, `interpolate_at`,
//! `merge_grids`, `rsf`) are copied verbatim from it, not reimplemented.
//!
//! # A deliberate methodological choice: `t_inj` is held fixed
//!
//! The injection duration `t_inj` is normally computed as `2 * Δt` (see
//! `validation/reference.rs`'s `t_inj()`), which ties it to whatever
//! `n_steps` happens to be. Sweeping `n_steps` while recomputing `t_inj`
//! that way would change *two* things at once — the solver's resolution
//! **and** the physical injection profile — conflating the question this
//! sweep is meant to answer. `t_inj` is therefore fixed at its baseline
//! value (`2 * Δt_baseline`) across every resolution in the sweep; only
//! `n_steps` (hence Δt) varies.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example stiffness_convergence
//! ```
//!
//! Writes a JSON report to a temporary file (path printed on completion) and
//! prints a summary table to stdout.

use chrom_rs::models::{LangmuirMulti, SpeciesParams, TemporalInjection};
use chrom_rs::output::export::{JsonError, to_json};
use chrom_rs::physics::{PhysicalData, PhysicalModel, PhysicalQuantity};
use chrom_rs::solver::{
    DomainBoundaries, EulerSolver, RK4Solver, Scenario, SimulationResult, Solver,
    SolverConfiguration,
};
use serde_json::{Map, Value, json};

// =================================================================================================
// Dissimilarity criterion — minimal local copy of `validation/dissimilarity.rs`
// =================================================================================================

fn trapezoid(times: &[f64], values: &[f64]) -> f64 {
    times
        .windows(2)
        .zip(values.windows(2))
        .map(|(t, v)| 0.5 * (v[0] + v[1]) * (t[1] - t[0]))
        .sum()
}

fn normalize(times: &[f64], concentrations: &[f64]) -> Vec<f64> {
    let area = trapezoid(times, concentrations);
    concentrations.iter().map(|c| c / area).collect()
}

fn interpolate_at(times: &[f64], values: &[f64], t: f64) -> f64 {
    if t <= times[0] {
        return values[0];
    }
    if t >= *times.last().unwrap() {
        return *values.last().unwrap();
    }
    let idx = times.partition_point(|&x| x <= t).saturating_sub(1);
    let idx = idx.min(times.len() - 2);
    let dt = times[idx + 1] - times[idx];
    if dt < f64::EPSILON {
        return values[idx];
    }
    let alpha = (t - times[idx]) / dt;
    values[idx] * (1.0 - alpha) + values[idx + 1] * alpha
}

fn merge_grids(t1: &[f64], t2: &[f64], tol: f64) -> Vec<f64> {
    let mut merged: Vec<f64> = t1.iter().chain(t2.iter()).copied().collect();
    merged.sort_by(|a, b| a.partial_cmp(b).unwrap());
    merged.dedup_by(|a, b| (*a - *b).abs() < tol);
    merged
}

/// Surface resolution $R_{sf}$ between two concentration profiles (Nicoud, 2015, §7.1).
fn rsf(t1: &[f64], c1: &[f64], t2: &[f64], c2: &[f64]) -> f64 {
    let y1 = normalize(t1, c1);
    let y2 = normalize(t2, c2);
    let min_dt = t1
        .windows(2)
        .chain(t2.windows(2))
        .map(|w| (w[1] - w[0]).abs())
        .fold(f64::INFINITY, f64::min);
    let tol = min_dt * 0.01;
    let grid = merge_grids(t1, t2, tol);
    let y1_merged: Vec<f64> = grid.iter().map(|&t| interpolate_at(t1, &y1, t)).collect();
    let y2_merged: Vec<f64> = grid.iter().map(|&t| interpolate_at(t2, &y2, t)).collect();
    let diff: Vec<f64> = y1_merged
        .iter()
        .zip(y2_merged.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    trapezoid(&grid, &diff)
}

// =================================================================================================
// Reference cases — minimal local copy of `validation/reference.rs`
// =================================================================================================

struct SweepCase {
    name: &'static str,
    column_length: f64,
    porosity: f64,
    velocity: f64,
    n_points: usize,
    c0: f64,
    t_total: f64,
    /// Baseline `n_steps`, as used in `validation/reference.rs`. The sweep
    /// multiplies this by each factor in `MULTIPLIERS`.
    n_steps_baseline: usize,
    species: Vec<(&'static str, f64, f64, u32)>,
}

impl SweepCase {
    /// Injection duration, fixed at `2 * Δt_baseline` for every resolution
    /// in the sweep — see the module-level doc comment for why.
    fn t_inj_fixed(&self) -> f64 {
        2.0 * self.t_total / self.n_steps_baseline as f64
    }

    fn ascorbic_erythorbic() -> Self {
        Self {
            name: "ascorbic_erythorbic",
            column_length: 0.25,
            porosity: 0.4,
            velocity: 1.0e-3,
            n_points: 100,
            c0: 1.0e-3,
            t_total: 800.0,
            n_steps_baseline: 4000,
            species: vec![("Ascorbic", 1.0, 1.1, 2), ("Erythorbic", 1.0, 1.7, 2)],
        }
    }

    fn glucose_fructose_linear() -> Self {
        Self {
            name: "glucose_fructose_linear",
            column_length: 0.3,
            porosity: 0.4,
            velocity: 1.0e-3,
            n_points: 100,
            c0: 1.0e-3,
            t_total: 400.0,
            n_steps_baseline: 2000,
            species: vec![
                ("Glucose", 0.0, 0.45, 1),
                ("Fructose", 0.0, 0.7666666666666667, 1),
            ],
        }
    }
}

/// Multipliers applied to `n_steps_baseline`. Log-spaced-ish rather than
/// linear, to cover one order of magnitude in Δt without an excessive
/// number of runs.
const MULTIPLIERS: [usize; 6] = [1, 2, 5, 10, 25, 50];

// =================================================================================================
// Simulation
// =================================================================================================

fn outlet_profiles(result: &SimulationResult, n_points: usize, n_species: usize) -> Vec<Vec<f64>> {
    let mut per_species: Vec<Vec<f64>> = vec![Vec::new(); n_species];
    for state in &result.state_trajectory {
        if let Some(PhysicalData::Matrix(m)) = state.get(PhysicalQuantity::Concentration) {
            for s in 0..n_species {
                per_species[s].push(m[(n_points - 1, s)]);
            }
        }
    }
    per_species
}

/// Runs `case` at `n_steps` with the given solver, returning per-species
/// outlet concentration profiles against their shared time grid.
fn run(case: &SweepCase, n_steps: usize, solver: &dyn Solver) -> (Vec<f64>, Vec<Vec<f64>>) {
    let t_inj = case.t_inj_fixed();
    let species: Vec<SpeciesParams> = case
        .species
        .iter()
        .map(|&(name, lambda, k, n)| {
            SpeciesParams::new(
                name,
                lambda,
                k,
                n,
                TemporalInjection::rectangle(0.0, t_inj, case.c0),
            )
        })
        .collect();
    let n_species = species.len();
    let model = LangmuirMulti::new(
        species,
        case.n_points,
        case.porosity,
        case.velocity,
        case.column_length,
    )
    .expect("reference case parameters always valid");
    let boundaries = DomainBoundaries::temporal(model.setup_initial_state());
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(case.t_total, n_steps);
    let result = solver.solve(&scenario, &config).expect("solver failed");
    let profiles = outlet_profiles(&result, case.n_points, n_species);
    (result.time_points, profiles)
}

/// Sweeps `MULTIPLIERS` for one case, returning one JSON object per
/// resolution with `n_steps`, `delta_t_s`, and the worst-case (max over
/// species) $R_{sf}$(Euler, RK4) at that resolution.
fn sweep_case(case: &SweepCase) -> Vec<Value> {
    MULTIPLIERS
        .iter()
        .map(|&mult| {
            let n_steps = case.n_steps_baseline * mult;
            let dt = case.t_total / n_steps as f64;

            let (t_euler, c_euler) = run(case, n_steps, &EulerSolver::new());
            let (t_rk4, c_rk4) = run(case, n_steps, &RK4Solver::new());

            let rsf_max = (0..case.species.len())
                .map(|i| rsf(&t_euler, &c_euler[i], &t_rk4, &c_rk4[i]))
                .fold(f64::MIN, f64::max);

            println!(
                "  {:<24} n_steps={:>7}  (×{:>2})  Δt={:>10.6} s  Rsf(Euler,RK4)={:.5}",
                case.name, n_steps, mult, dt, rsf_max
            );

            json!({
                "multiplier": mult,
                "n_steps": n_steps,
                "delta_t_s": dt,
                "rsf_euler_vs_rk4": rsf_max,
            })
        })
        .collect()
}

// =================================================================================================
// Entry point
// =================================================================================================

fn main() -> Result<(), JsonError> {
    let cases = [
        SweepCase::ascorbic_erythorbic(),
        SweepCase::glucose_fructose_linear(),
    ];

    println!(
        "Running Δt convergence sweep — {} case(s)...\n",
        cases.len()
    );

    let mut report = Map::new();
    for case in &cases {
        println!("{}:", case.name);
        let points = sweep_case(case);
        report.insert(case.name.to_string(), Value::Array(points));
        println!();
    }

    let path = std::env::temp_dir().join("stiffness_convergence.json");
    to_json(&report, path.to_str().unwrap())?;
    println!("Report written to {}", path.display());

    Ok(())
}
