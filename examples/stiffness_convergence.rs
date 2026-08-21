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

//! # Ajouts du 2026-08-17 / *2026-08-17 additions*
//!
//! Deux ajouts additifs, sans changer le format existant (`rsf_euler_vs_rk4`
//! reste présent et inchangé — `tools/plot_stiffness_convergence.rs`
//! continue de fonctionner sans modification) :
//!
//! 1. **Temps de résolution par solveur** (`euler_ms`, `rk4_ms`), même
//!    convention `std::time::Instant` que `examples/validation_report.rs`.
//!    Permet de croiser précision (Rsf) et coût (temps) sur le même
//!    balayage, plutôt que dans deux exemples séparés.
//! 2. **Rsf auto-cohérent par solveur** (`rsf_euler_self_vs_next`,
//!    `rsf_rk4_self_vs_next`) : Rsf entre la résolution courante et la
//!    résolution suivante (plus fine) du même solveur, calculé séparément
//!    pour Euler et RK4 — pas seulement Euler contre RK4 à résolution
//!    égale. Répond à la question « RK4 a-t-il besoin de moins de pas
//!    qu'Euler pour un résultat équivalent ? » : `rsf_euler_vs_rk4` seul ne
//!    peut pas y répondre (les deux solveurs peuvent s'accorder tout en
//!    étant l'un et l'autre loin de convergence) ; comparer chaque solveur
//!    à sa propre résolution suivante montre à quelle résolution *chacun*
//!    s'est stabilisé, indépendamment de l'autre. `None` sur la dernière
//!    résolution du balayage (pas de résolution plus fine pour comparer).
//!
//! Un troisième cas, `erythorbic_alone`, a été ajouté aux deux cas
//! existants — voir la note sur `SweepCase::erythorbic_alone` pour la
//! justification (c'est l'espèce déjà validée la plus rétentive du dépôt,
//! pas besoin d'inventer une isotherme synthétique).
//! *A third case, `erythorbic_alone`, was added alongside the two existing
//! ones — see the note on `SweepCase::erythorbic_alone` for the rationale
//! (it is the most retentive already-validated species in the repository,
//! no need to invent a synthetic isotherm).*

//! # Correction du 2026-08-18 / *2026-08-18 correction*
//!
//! `euler_ms`/`rk4_ms` (ajout du 2026-08-17 ci-dessus) étaient mesurés en un
//! seul `Instant::now()` par résolution — une erreur : contrairement à Rsf
//! (fonction pure du résultat du solveur, réellement déterministe), le temps
//! mesuré est du chronométrage mur-à-mur ordinaire, sujet au même bruit
//! d'ordonnancement/thermique documenté ailleurs dans cette session (10 à
//! 59% de dérive observée sur les bancs Criterion selon les conditions). Un
//! seul run n'a pas plus de légitimité ici que dans un banc Criterion.
//! Chaque résolution est donc désormais chronométrée sur `N_TIMING_RUNS`
//! passages indépendants ; le JSON expose `euler_ms_mean`/`_min`/`_max` (et
//! l'équivalent RK4) au lieu d'un unique `euler_ms`/`rk4_ms`. Les profils de
//! concentration utilisés pour Rsf viennent d'un seul de ces passages (le
//! solveur étant déterministe, tous produisent des trajectoires
//! bit-identiques — seul le temps varie).
//! *`euler_ms`/`rk4_ms` (2026-08-17 addition above) were measured with a
//! single `Instant::now()` per resolution — an error: unlike Rsf (a pure
//! function of the solver's output, genuinely deterministic), the measured
//! time is ordinary wall-clock timing, subject to the same scheduling/
//! thermal noise documented elsewhere in this session (10 to 59% drift
//! observed on the Criterion benches depending on conditions). A single run
//! has no more legitimacy here than in a Criterion bench. Each resolution is
//! therefore now timed over `N_TIMING_RUNS` independent passes; the JSON
//! exposes `euler_ms_mean`/`_min`/`_max` (and the RK4 equivalent) instead of
//! a single `euler_ms`/`rk4_ms`. The concentration profiles used for Rsf
//! come from just one of these passes (the solver being deterministic, all
//! produce bit-identical trajectories — only the timing varies).*

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

    /// **Erythorbic seule** (ajout 2026-08-17) — pas une nouvelle isotherme
    /// inventée, mais l'espèce déjà validée dans
    /// `MultiSpeciesCase::ascorbic_erythorbic` (`validation/reference.rs`),
    /// isolée. C'est l'isotherme la plus rétentive déjà validée dans le
    /// dépôt : $K_a^0 = \lambda + \bar{N}\tilde{K} = 1.0 + 1.2 \times 1.7 =
    /// 3.04$ (avec $\bar N=(1-\varepsilon)N=1.2$), au-dessus de TFA (2.8),
    /// Ascorbique seule (2.32), et très au-dessus de Glucose/Fructose
    /// (0.27/0.46). Colonne et discrétisation reprises telles quelles
    /// d'`ascorbic_erythorbic` pour rester comparable.
    /// *Not a newly invented isotherm, but the species already validated in
    /// `MultiSpeciesCase::ascorbic_erythorbic` (`validation/reference.rs`),
    /// isolated. It is the most retentive already-validated isotherm in the
    /// repository: $K_a^0 = \lambda + \bar{N}\tilde{K} = 1.0 + 1.2 \times
    /// 1.7 = 3.04$ (with $\bar N=(1-\varepsilon)N=1.2$), above TFA (2.8),
    /// Ascorbic alone (2.32), and well above Glucose/Fructose (0.27/0.46).
    /// Column and discretisation reused as-is from `ascorbic_erythorbic` to
    /// stay comparable.*
    fn erythorbic_alone() -> Self {
        Self {
            name: "erythorbic_alone",
            column_length: 0.25,
            porosity: 0.4,
            velocity: 1.0e-3,
            n_points: 100,
            c0: 1.0e-3,
            t_total: 800.0,
            n_steps_baseline: 4000,
            species: vec![("Erythorbic", 1.0, 1.7, 2)],
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

/// Nombre de passages chronométrés indépendants par résolution/solveur.
/// Le solveur étant déterministe, un seul passage suffirait pour la
/// trajectoire (Rsf) — mais pas pour le temps, qui reste sujet au bruit
/// d'ordonnancement/thermique documenté dans la correction du 2026-08-18
/// ci-dessus. 5 passages : compromis entre légitimité statistique minimale
/// et coût (chaque résolution est déjà relancée pour chaque multiplicateur
/// × chaque cas × chaque solveur).
/// *Number of independent timed passes per resolution/solver. The solver
/// being deterministic, a single pass would suffice for the trajectory
/// (Rsf) — but not for timing, still subject to the scheduling/thermal
/// noise documented in the 2026-08-18 correction above. 5 passes: a
/// compromise between minimal statistical legitimacy and cost (each
/// resolution is already re-run per multiplier × case × solver).*
const N_TIMING_RUNS: usize = 5;

/// Chronomètre `solver.solve()` sur `N_TIMING_RUNS` passages indépendants,
/// retournant `(mean_ms, min_ms, max_ms)` et le profil du dernier passage
/// (trajectoire bit-identique sur tous les passages, seul le temps varie).
/// *Times `solver.solve()` over `N_TIMING_RUNS` independent passes,
/// returning `(mean_ms, min_ms, max_ms)` and the last pass's profile
/// (bit-identical trajectory across all passes, only timing varies).*
fn timed_run(
    case: &SweepCase,
    n_steps: usize,
    solver: &dyn Solver,
) -> (Vec<f64>, Vec<Vec<f64>>, f64, f64, f64) {
    let mut times_ms = Vec::with_capacity(N_TIMING_RUNS);
    let mut last_profile: Option<(Vec<f64>, Vec<Vec<f64>>)> = None;

    for _ in 0..N_TIMING_RUNS {
        let start = std::time::Instant::now();
        let (t, c) = run(case, n_steps, solver);
        times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        last_profile = Some((t, c));
    }

    let mean_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (t, c) = last_profile.expect("N_TIMING_RUNS >= 1");
    (t, c, mean_ms, min_ms, max_ms)
}

/// Sweeps `MULTIPLIERS` for one case, returning one JSON object per
/// resolution with `n_steps`, `delta_t_s`, `rsf_euler_vs_rk4` (unchanged
/// from the original format), plus per-solver timing (mean/min/max over
/// `N_TIMING_RUNS` passes — see the 2026-08-18 correction note) and
/// self-consecutive Rsf.
fn sweep_case(case: &SweepCase) -> Vec<Value> {
    // Résout à chaque multiplicateur, en conservant le profil et le temps
    // de CHAQUE solveur pour permettre la comparaison auto-cohérente avec
    // le multiplicateur suivant (plus fin).
    // Solves at each multiplier, keeping EACH solver's profile and timing
    // to allow the self-consistency comparison against the next (finer)
    // multiplier.
    struct Resolution {
        mult: usize,
        n_steps: usize,
        dt: f64,
        t_euler: Vec<f64>,
        c_euler: Vec<Vec<f64>>,
        euler_ms_mean: f64,
        euler_ms_min: f64,
        euler_ms_max: f64,
        t_rk4: Vec<f64>,
        c_rk4: Vec<Vec<f64>>,
        rk4_ms_mean: f64,
        rk4_ms_min: f64,
        rk4_ms_max: f64,
    }

    let resolutions: Vec<Resolution> = MULTIPLIERS
        .iter()
        .map(|&mult| {
            let n_steps = case.n_steps_baseline * mult;
            let dt = case.t_total / n_steps as f64;

            let (t_euler, c_euler, euler_ms_mean, euler_ms_min, euler_ms_max) =
                timed_run(case, n_steps, &EulerSolver::new());
            let (t_rk4, c_rk4, rk4_ms_mean, rk4_ms_min, rk4_ms_max) =
                timed_run(case, n_steps, &RK4Solver::new());

            Resolution {
                mult,
                n_steps,
                dt,
                t_euler,
                c_euler,
                euler_ms_mean,
                euler_ms_min,
                euler_ms_max,
                t_rk4,
                c_rk4,
                rk4_ms_mean,
                rk4_ms_min,
                rk4_ms_max,
            }
        })
        .collect();

    resolutions
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let rsf_max = (0..case.species.len())
                .map(|s| rsf(&r.t_euler, &r.c_euler[s], &r.t_rk4, &r.c_rk4[s]))
                .fold(f64::MIN, f64::max);

            // Auto-cohérence par solveur contre la résolution suivante
            // (plus fine) — None sur le dernier point du balayage.
            // Self-consistency per solver against the next (finer)
            // resolution — None on the sweep's last point.
            let (rsf_euler_self, rsf_rk4_self) = match resolutions.get(i + 1) {
                Some(next) => {
                    let e = (0..case.species.len())
                        .map(|s| rsf(&r.t_euler, &r.c_euler[s], &next.t_euler, &next.c_euler[s]))
                        .fold(f64::MIN, f64::max);
                    let k = (0..case.species.len())
                        .map(|s| rsf(&r.t_rk4, &r.c_rk4[s], &next.t_rk4, &next.c_rk4[s]))
                        .fold(f64::MIN, f64::max);
                    (Some(e), Some(k))
                }
                None => (None, None),
            };

            println!(
                "  {:<24} n_steps={:>7}  (×{:>2})  Δt={:>10.6} s  \
                 Rsf(Euler,RK4)={:.5}  Euler={:>8.2} ms [{:.2}-{:.2}]  RK4={:>8.2} ms [{:.2}-{:.2}]  \
                 self(E)={}  self(RK4)={}",
                case.name,
                r.n_steps,
                r.mult,
                r.dt,
                rsf_max,
                r.euler_ms_mean,
                r.euler_ms_min,
                r.euler_ms_max,
                r.rk4_ms_mean,
                r.rk4_ms_min,
                r.rk4_ms_max,
                rsf_euler_self
                    .map(|v| format!("{v:.5}"))
                    .unwrap_or_else(|| "—".into()),
                rsf_rk4_self
                    .map(|v| format!("{v:.5}"))
                    .unwrap_or_else(|| "—".into()),
            );

            json!({
                "multiplier": r.mult,
                "n_steps": r.n_steps,
                "delta_t_s": r.dt,
                "rsf_euler_vs_rk4": rsf_max,
                "euler_ms_mean": r.euler_ms_mean,
                "euler_ms_min": r.euler_ms_min,
                "euler_ms_max": r.euler_ms_max,
                "rk4_ms_mean": r.rk4_ms_mean,
                "rk4_ms_min": r.rk4_ms_min,
                "rk4_ms_max": r.rk4_ms_max,
                "timing_runs": N_TIMING_RUNS,
                "rsf_euler_self_vs_next": rsf_euler_self,
                "rsf_rk4_self_vs_next": rsf_rk4_self,
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
        SweepCase::erythorbic_alone(),
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
