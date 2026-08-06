//! On-demand scientific validation report (issue #44).
//!
//! Runs the reference cases from `validation/reference.rs` and produces a
//! structured JSON summary — simulated vs. reference peak positions, the
//! $R_{sf}$ dissimilarity between the Euler and RK4 solvers, and the solver
//! parameters used (Δt, Δz, CFL) — for scientific review and publication.
//!
//! Complements the automated CI tests in `validation/main.rs` (which assert
//! pass/fail against fixed tolerances) with a human-readable artifact meant
//! for inspection rather than assertion.
//!
//! # Why the reference values and $R_{sf}$ are duplicated here
//!
//! `validation/` is a standalone `[[test]]` crate (declared in `Cargo.toml`),
//! not part of the public `chrom-rs` library — examples cannot depend on it.
//! The physical case parameters (mirroring [`validation::reference`]) and the
//! surface-resolution criterion (mirroring [`validation::dissimilarity`]) are
//! therefore reproduced locally below, kept intentionally minimal. See
//! `validation/reference.rs` and `validation/dissimilarity.rs` for the
//! authoritative, test-asserted versions and their full literature citations.
//!
//! # Usage
//!
//! ```text
//! cargo run --example validation_report
//! ```
//!
//! Writes a JSON report to a temporary file (path printed on completion) and
//! prints a summary table to stdout.

use chrom_rs::models::{LangmuirMulti, LangmuirSingle, SpeciesParams, TemporalInjection};
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

/// First moment (retention time) of an outlet concentration profile.
fn first_moment(times: &[f64], concs: &[f64]) -> f64 {
    let area = trapezoid(times, concs);
    let num: Vec<f64> = times.iter().zip(concs).map(|(t, c)| t * c).collect();
    trapezoid(times, &num) / area
}

/// Peak position (time of maximum) and amplitude of an outlet profile.
fn peak(times: &[f64], concs: &[f64]) -> (f64, f64) {
    let (idx, &c) = concs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    (times[idx], c)
}

// =================================================================================================
// Reference cases — minimal local copy of `validation/reference.rs`
// =================================================================================================

/// One species entry within a validated case (single- or multi-species).
struct SpeciesCase {
    name: &'static str,
    lambda: f64,
    langmuir_k: f64,
    port_number: u32,
    /// Analytical dilute-limit retention time [s]
    t_retention: f64,
}

/// A physical validation case, single- or multi-species.
struct Case {
    name: &'static str,
    column_length: f64,
    porosity: f64,
    velocity: f64,
    n_points: usize,
    c0: f64,
    t_total: f64,
    n_steps: usize,
    species: Vec<SpeciesCase>,
}

impl Case {
    fn t_inj(&self) -> f64 {
        2.0 * self.t_total / self.n_steps as f64
    }

    fn dz(&self) -> f64 {
        self.column_length / self.n_points as f64
    }

    fn dt(&self) -> f64 {
        self.t_total / self.n_steps as f64
    }

    /// CFL number $u_e \Delta t / \Delta z$, $u_e = u / \varepsilon$.
    fn cfl(&self) -> f64 {
        (self.velocity / self.porosity) * self.dt() / self.dz()
    }

    /// Case A — TFA, linear regime. Mirrors `ReferenceCase::linear_tfa()`.
    fn tfa_linear() -> Self {
        Self {
            name: "tfa_linear",
            column_length: 0.3,
            porosity: 0.4,
            velocity: 1.0e-3,
            n_points: 100,
            c0: 1.0e-3,
            t_total: 900.0,
            n_steps: 4500,
            species: vec![SpeciesCase {
                name: "TFA",
                lambda: 1.0,
                langmuir_k: 0.5,
                port_number: 6,
                t_retention: 624.0,
            }],
        }
    }

    /// Case D — Ascorbic/Erythorbic, competitive Langmuir. Mirrors
    /// `MultiSpeciesCase::ascorbic_erythorbic()`.
    fn ascorbic_erythorbic() -> Self {
        Self {
            name: "ascorbic_erythorbic",
            column_length: 0.25,
            porosity: 0.4,
            velocity: 1.0e-3,
            n_points: 100,
            c0: 1.0e-3,
            t_total: 800.0,
            n_steps: 4000,
            species: vec![
                SpeciesCase {
                    name: "Ascorbic",
                    lambda: 1.0,
                    langmuir_k: 1.1,
                    port_number: 2,
                    t_retention: 448.0,
                },
                SpeciesCase {
                    name: "Erythorbic",
                    lambda: 1.0,
                    langmuir_k: 1.7,
                    port_number: 2,
                    t_retention: 556.0,
                },
            ],
        }
    }

    /// Case E — Glucose/Fructose, linear regime. Mirrors
    /// `MultiSpeciesCase::glucose_fructose_linear()`.
    fn glucose_fructose_linear() -> Self {
        let porosity = 0.4;
        Self {
            name: "glucose_fructose_linear",
            column_length: 0.3,
            porosity,
            velocity: 1.0e-3,
            n_points: 100,
            c0: 1.0e-3,
            t_total: 400.0,
            n_steps: 2000,
            species: vec![
                SpeciesCase {
                    name: "Glucose",
                    lambda: 0.0,
                    langmuir_k: 0.27 / (1.0 - porosity),
                    port_number: 1,
                    t_retention: 168.6,
                },
                SpeciesCase {
                    name: "Fructose",
                    lambda: 0.0,
                    langmuir_k: 0.46 / (1.0 - porosity),
                    port_number: 1,
                    t_retention: 202.8,
                },
            ],
        }
    }
}

// =================================================================================================
// Simulation runners
// =================================================================================================

/// Extract per-species $C_{outlet}(t)$; species 0 for a single-species result.
fn outlet_profiles(result: &SimulationResult, n_points: usize, n_species: usize) -> Vec<Vec<f64>> {
    let mut per_species: Vec<Vec<f64>> = vec![Vec::new(); n_species];
    for state in &result.state_trajectory {
        match state.get(PhysicalQuantity::Concentration) {
            Some(PhysicalData::Vector(v)) => per_species[0].push(v[n_points - 1]),
            Some(PhysicalData::Matrix(m)) => {
                for s in 0..n_species {
                    per_species[s].push(m[(n_points - 1, s)]);
                }
            }
            _ => {}
        }
    }
    per_species
}

/// Run `case` with the given solver, returning `(time_points, [species][time_step])`.
fn run(case: &Case, solver: &dyn Solver) -> (Vec<f64>, Vec<Vec<f64>>) {
    let t_inj = case.t_inj();
    let config = SolverConfiguration::time_evolution(case.t_total, case.n_steps);

    let (result, n_species) = if case.species.len() == 1 {
        let sp = &case.species[0];
        let model = LangmuirSingle::new(
            sp.lambda,
            sp.langmuir_k,
            sp.port_number as f64,
            case.porosity,
            case.velocity,
            case.column_length,
            case.n_points,
            TemporalInjection::rectangle(0.0, t_inj, case.c0),
        );
        let boundaries = DomainBoundaries::temporal(model.setup_initial_state());
        let scenario = Scenario::new(Box::new(model), boundaries);
        (solver.solve(&scenario, &config).expect("solver failed"), 1)
    } else {
        let species: Vec<SpeciesParams> = case
            .species
            .iter()
            .map(|sp| {
                SpeciesParams::new(
                    sp.name,
                    sp.lambda,
                    sp.langmuir_k,
                    sp.port_number,
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
        .expect("invalid case parameters");
        let boundaries = DomainBoundaries::temporal(model.setup_initial_state());
        let scenario = Scenario::new(Box::new(model), boundaries);
        (
            solver.solve(&scenario, &config).expect("solver failed"),
            n_species,
        )
    };

    let profiles = outlet_profiles(&result, case.n_points, n_species);
    (result.time_points, profiles)
}

// =================================================================================================
// Report generation
// =================================================================================================

/// Build the JSON report entry for one case.
fn report_case(case: &Case) -> Value {
    let (t_rk4, c_rk4) = run(case, &RK4Solver::new());
    let (t_euler, c_euler) = run(case, &EulerSolver::new());

    let species: Vec<Value> = case
        .species
        .iter()
        .enumerate()
        .map(|(i, sp)| {
            let (peak_time, peak_amplitude) = peak(&t_rk4, &c_rk4[i]);
            let t_r_sim = first_moment(&t_rk4, &c_rk4[i]);
            let error_pct = (t_r_sim - sp.t_retention).abs() / sp.t_retention * 100.0;
            json!({
                "name": sp.name,
                "peak_position_s": peak_time,
                "peak_amplitude_mol_l": peak_amplitude,
                "retention_time_sim_s": t_r_sim,
                "retention_time_reference_s": sp.t_retention,
                "retention_time_error_pct": error_pct,
            })
        })
        .collect();

    // Rsf(Euler, RK4) per species, then case-level max (worst-case divergence).
    let rsf_per_species: Vec<f64> = (0..case.species.len())
        .map(|i| rsf(&t_euler, &c_euler[i], &t_rk4, &c_rk4[i]))
        .collect();
    let rsf_max = rsf_per_species.iter().cloned().fold(f64::MIN, f64::max);

    json!({
        "case": case.name,
        "species": species,
        "rsf_euler_vs_rk4": rsf_max,
        "solver_parameters": {
            "delta_t_s": case.dt(),
            "delta_z_m": case.dz(),
            "cfl": case.cfl(),
            "n_points": case.n_points,
            "n_steps": case.n_steps,
        },
    })
}

fn main() -> Result<(), JsonError> {
    let cases = [
        Case::tfa_linear(),
        Case::ascorbic_erythorbic(),
        Case::glucose_fructose_linear(),
    ];

    println!("Running validation report — {} case(s)...\n", cases.len());

    let mut report_cases = Vec::with_capacity(cases.len());
    for case in &cases {
        print!("  {:<24} ... ", case.name);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let entry = report_case(case);
        println!(
            "Rsf(Euler,RK4) = {:.4}",
            entry["rsf_euler_vs_rk4"].as_f64().unwrap()
        );
        report_cases.push(entry);
    }

    let report: Map<String, Value> = json!({ "cases": report_cases })
        .as_object()
        .cloned()
        .unwrap();

    let path = std::env::temp_dir().join("validation_report.json");
    to_json(&report, path.to_str().unwrap())?;

    println!("\n=== Summary ===");
    for entry in &report_cases {
        println!("\n{}", entry["case"].as_str().unwrap());
        for sp in entry["species"].as_array().unwrap() {
            println!(
                "  {:<10} tR sim = {:>7.1} s   tR ref = {:>7.1} s   error = {:.3} %",
                sp["name"].as_str().unwrap(),
                sp["retention_time_sim_s"].as_f64().unwrap(),
                sp["retention_time_reference_s"].as_f64().unwrap(),
                sp["retention_time_error_pct"].as_f64().unwrap(),
            );
        }
        let params = &entry["solver_parameters"];
        println!(
            "  Δt = {:.4} s, Δz = {:.6} m, CFL = {:.4}",
            params["delta_t_s"].as_f64().unwrap(),
            params["delta_z_m"].as_f64().unwrap(),
            params["cfl"].as_f64().unwrap(),
        );
    }

    println!("\nReport written to {:?}", path);
    Ok(())
}
