//! Balayages spatial et temporel découplés — isole ce que
//! `stiffness_convergence.rs` ne peut pas isoler.
//! *Decoupled spatial and temporal sweeps — isolates what
//! `stiffness_convergence.rs` cannot isolate.*
//!
//! # Pourquoi cet exemple / *Why this example*
//!
//! `stiffness_convergence.rs` balaie `n_steps` (Δt) à `n_points` **fixe**
//! (100). Résultat observé (session du 2026-08-20) : l'ordre empirique de
//! RK4 est erratique (0,13 à 1,96 selon les paires de points) parce que
//! son erreur temporelle devient négligeable devant l'erreur spatiale
//! *fixe* dès les premiers raffinements testés — on mesure alors du bruit
//! autour d'un plancher spatial, pas la décroissance en $\Delta t^4$
//! attendue. Le même problème existe en miroir : on n'a jamais mesuré
//! l'ordre spatial pur, puisque `n_points` n'a jamais varié.
//!
//! Cet exemple sépare les deux axes en deux balayages indépendants :
//!
//! 1. **Spatial pur** : `n_steps` fixé très fin (le plus fin déjà testé
//!    dans `stiffness_convergence.rs`, `n_steps_baseline × 50`), `n_points`
//!    variable (`SPATIAL_MULTIPLIERS`). L'erreur temporelle devient
//!    négligeable, ce qui devrait laisser apparaître l'ordre spatial du
//!    schéma amont (attendu : 1).
//! 2. **Temporel, plancher repoussé** : `n_points` fixé à la résolution la
//!    plus fine du balayage spatial ci-dessus, `n_steps` variable (mêmes
//!    multiplicateurs que `stiffness_convergence.rs`, pour rester
//!    comparable). Le plancher spatial est repoussé assez bas pour laisser
//!    (en principe) la décroissance en $\Delta t^4$ de RK4 se manifester
//!    avant de le rencontrer.
//!
//! Même limite de coût que `cfl_stability_scan.rs` : passages uniques,
//! aucun échantillonnage Criterion — mais `n_points × n_steps` grandit
//! vite (le balayage temporel tourne à 8× la résolution spatiale de base,
//! donc ~8× le coût par pas de `stiffness_convergence.rs` à `n_steps`
//! égal). Pas de session de plusieurs heures pour autant : contrairement à
//! `stiffness_convergence.rs`, cet exemple ne chronomètre rien (pas de
//! `N_TIMING_RUNS`, pas de passages répétés pour moyenner le bruit
//! d'exécution) — un seul passage déterministe par point suffit puisque
//! seul l'ordre de convergence (fonction pure du résultat, pas du temps
//! d'exécution) est mesuré ici.
//!
//! # Ce que cet exemple ne fait pas
//!
//! Il ne mesure aucun temps de résolution — ce n'est pas son objet (voir
//! `stiffness_convergence.rs` et `bench_cfl_stability` pour ça). Il ne
//! remplace pas non plus `stiffness_convergence.rs`, qui reste la
//! référence pour Rsf(Euler, RK4) à `n_points` fixe.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example spatial_temporal_convergence
//! ```
//!
//! Écrit un rapport JSON dans le répertoire temporaire système
//! (`spatial_temporal_convergence.json`), même convention que
//! `stiffness_convergence.rs`.

use chrom_rs::models::{LangmuirMulti, SpeciesParams, TemporalInjection};
use chrom_rs::output::export::{JsonError, to_json};
use chrom_rs::physics::{PhysicalData, PhysicalModel, PhysicalQuantity};
use chrom_rs::solver::{
    DomainBoundaries, EulerSolver, RK4Solver, Scenario, SimulationResult, Solver,
    SolverConfiguration,
};
use serde_json::{Map, Value, json};

// =================================================================================================
// Rsf — copie exacte de examples/stiffness_convergence.rs (voir son en-tête
// pour la justification de la duplication : validation/ est un crate de
// test [[test]] séparé, inaccessible depuis examples/).
// Rsf — exact copy of examples/stiffness_convergence.rs (see its header
// for why this is duplicated: validation/ is a separate [[test]] crate,
// unreachable from examples/).
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
fn rsf(label: &str, t1: &[f64], c1: &[f64], t2: &[f64], c2: &[f64]) -> f64 {
    // Diagnostic ciblé (2026-08-22) : le solveur renvoie des valeurs finies
    // (vérifié séparément par le contrôle NaN sur c_euler/c_rk4 avant
    // l'appel à rsf), mais self(E)/self(RK4) ressortent NaN pour les cas à
    // 2 espèces à n_points=800. Suspect : l'aire sous la courbe
    // (trapezoid) utilisée par normalize() tombe à zéro (ou négative, ou
    // NaN) pour une des deux courbes, produisant du NaN par division —
    // sans qu'aucune valeur individuelle ne soit elle-même NaN/Inf.
    // *Targeted diagnostic (2026-08-22): the solver returns finite values
    // (checked separately by the NaN check on c_euler/c_rk4 before calling
    // rsf), but self(E)/self(RK4) come out NaN for the 2-species cases at
    // n_points=800. Suspect: the area under the curve (trapezoid) used by
    // normalize() drops to zero (or negative, or NaN) for one of the two
    // curves, producing NaN through division — without any individual
    // value itself being NaN/Inf.*
    let area1 = trapezoid(t1, c1);
    let area2 = trapezoid(t2, c2);
    // Formulation explicite plutôt que `!(area > 0.0)` : ce dernier
    // déclenche clippy::neg_cmp_op_on_partial_ord (la négation d'une
    // comparaison sur un type partiellement ordonné comme f64/NaN est
    // jugée fragile, même si le résultat est correct ici). area1/area2
    // sont "mauvaises" si NaN OU non strictement positives.
    // Explicit formulation instead of `!(area > 0.0)`: the latter trips
    // clippy::neg_cmp_op_on_partial_ord (negating a comparison on a
    // partially-ordered type like f64/NaN is considered fragile, even
    // though the result is correct here). area1/area2 are "bad" if NaN
    // OR not strictly positive.
    let area1_bad = area1.is_nan() || area1 <= 0.0;
    let area2_bad = area2.is_nan() || area2 <= 0.0;
    if area1_bad || area2_bad {
        eprintln!(
            "    !!! rsf[{label}]: aire non-positive détectée — area1={area1:.6e} area2={area2:.6e} \
             (len c1={}, c2={}, min(c1)={:.6e}, max(c1)={:.6e}, min(c2)={:.6e}, max(c2)={:.6e})",
            c1.len(),
            c2.len(),
            c1.iter().cloned().fold(f64::INFINITY, f64::min),
            c1.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            c2.iter().cloned().fold(f64::INFINITY, f64::min),
            c2.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
    }

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
// Cas de référence — mêmes cas que stiffness_convergence.rs, `n_points`
// désormais paramétrable par appel plutôt que figé dans la structure.
// Reference cases — same cases as stiffness_convergence.rs, `n_points`
// now parameterized per call instead of fixed in the struct.
// =================================================================================================

struct SweepCase {
    name: &'static str,
    column_length: f64,
    porosity: f64,
    velocity: f64,
    /// `n_points` de référence — utilisé uniquement pour calculer
    /// `t_inj_fixed()` (indépendant de la résolution spatiale en réalité,
    /// mais on garde le même point de référence temporel que
    /// `stiffness_convergence.rs` pour rester comparable). La résolution
    /// spatiale RÉELLEMENT utilisée dans chaque run est un paramètre de
    /// `run()`, pas ce champ.
    /// Reference `n_points` — only used to compute `t_inj_fixed()`
    /// (actually independent of spatial resolution, but kept as the same
    /// temporal reference point as `stiffness_convergence.rs` to stay
    /// comparable). The spatial resolution ACTUALLY used in each run is a
    /// parameter of `run()`, not this field.
    n_points_baseline: usize,
    c0: f64,
    t_total: f64,
    n_steps_baseline: usize,
    species: Vec<(&'static str, f64, f64, u32)>,
}

impl SweepCase {
    fn t_inj_fixed(&self) -> f64 {
        2.0 * self.t_total / self.n_steps_baseline as f64
    }

    fn ascorbic_erythorbic() -> Self {
        Self {
            name: "ascorbic_erythorbic",
            column_length: 0.25,
            porosity: 0.4,
            velocity: 1.0e-3,
            n_points_baseline: 100,
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
            n_points_baseline: 100,
            c0: 1.0e-3,
            t_total: 400.0,
            n_steps_baseline: 2000,
            species: vec![
                ("Glucose", 0.0, 0.45, 1),
                ("Fructose", 0.0, 0.7666666666666667, 1),
            ],
        }
    }

    fn erythorbic_alone() -> Self {
        Self {
            name: "erythorbic_alone",
            column_length: 0.25,
            porosity: 0.4,
            velocity: 1.0e-3,
            n_points_baseline: 100,
            c0: 1.0e-3,
            t_total: 800.0,
            n_steps_baseline: 4000,
            species: vec![("Erythorbic", 1.0, 1.7, 2)],
        }
    }
}

/// Multiplicateurs de `n_points_baseline` pour le balayage spatial pur —
/// 100 → 200 → 400 → 800.
/// `n_points_baseline` multipliers for the spatial-only sweep — 100 → 200
/// → 400 → 800.
const SPATIAL_MULTIPLIERS: [usize; 4] = [1, 2, 4, 8];

/// Multiplicateurs de `n_steps_baseline` pour le balayage temporel —
/// mêmes valeurs que `stiffness_convergence.rs`, pour rester comparable.
/// `n_steps_baseline` multipliers for the temporal sweep — same values as
/// `stiffness_convergence.rs`, to stay comparable.
const TEMPORAL_MULTIPLIERS: [usize; 6] = [1, 2, 5, 10, 25, 50];

/// Multiplicateur de `n_steps_baseline` utilisé comme Δt fixe (très fin)
/// pendant le balayage spatial pur — le plus fin déjà testé dans
/// `stiffness_convergence.rs`, pour que l'erreur temporelle y soit
/// négligeable devant l'erreur spatiale qu'on veut isoler.
/// `n_steps_baseline` multiplier used as the fixed (very fine) Δt during
/// the spatial-only sweep — the finest already tested in
/// `stiffness_convergence.rs`, so temporal error stays negligible next to
/// the spatial error we want to isolate.
const FIXED_N_STEPS_MULTIPLIER: usize = 50;

// =================================================================================================
// Simulation — identique à stiffness_convergence.rs, `n_points` en
// paramètre explicite au lieu d'être lu depuis `case`.
// Simulation — identical to stiffness_convergence.rs, `n_points` an
// explicit parameter instead of read from `case`.
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

fn run(
    case: &SweepCase,
    n_points: usize,
    n_steps: usize,
    solver: &dyn Solver,
) -> (Vec<f64>, Vec<Vec<f64>>) {
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
        n_points,
        case.porosity,
        case.velocity,
        case.column_length,
    )
    .expect("reference case parameters always valid");
    let boundaries = DomainBoundaries::temporal(model.setup_initial_state());
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(case.t_total, n_steps);
    let result = solver.solve(&scenario, &config).expect("solver failed");
    let profiles = outlet_profiles(&result, n_points, n_species);
    (result.time_points, profiles)
}

// =================================================================================================
// Auto-convergence sur un axe unique — factorisée pour servir aux deux
// balayages (seul ce qui varie entre les deux, `n_points` ou `n_steps`,
// diffère à l'appel).
// Self-convergence on a single axis — factored to serve both sweeps
// (only what varies between them, `n_points` or `n_steps`, differs at the
// call site).
// =================================================================================================

struct Point {
    /// Valeur du paramètre balayé à ce point (n_points ou n_steps, selon
    /// l'axe). *Swept parameter's value at this point (n_points or
    /// n_steps, depending on the axis).*
    swept_value: usize,
    n_points: usize,
    n_steps: usize,
    t_euler: Vec<f64>,
    c_euler: Vec<Vec<f64>>,
    t_rk4: Vec<f64>,
    c_rk4: Vec<Vec<f64>>,
}

/// `fold(f64::MIN, f64::max)` avale silencieusement un NaN : `f64::max`
/// renvoie l'opérande non-NaN, donc la graine `f64::MIN` survit et
/// s'affiche comme une valeur normale au lieu de signaler l'échec (bug
/// trouvé sur les données du 2026-08-22 : lignes affichant
/// `-1.797...e308`, exactement `f64::MIN` en notation décimale complète).
/// Cette fonction propage le NaN au lieu de l'ignorer.
/// *`fold(f64::MIN, f64::max)` silently swallows a NaN: `f64::max`
/// returns the non-NaN operand, so the `f64::MIN` seed survives and
/// prints as a normal value instead of signaling failure (bug found on
/// the 2026-08-22 data: rows showing `-1.797...e308`, exactly `f64::MIN`
/// in full decimal notation). This function propagates NaN instead of
/// ignoring it.*
fn max_or_nan(iter: impl Iterator<Item = f64>) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for v in iter {
        if v.is_nan() {
            return f64::NAN;
        }
        if v > max {
            max = v;
        }
    }
    max
}

fn sweep_axis(
    case: &SweepCase,
    swept_values: &[usize],
    n_points_for: impl Fn(usize) -> usize,
    n_steps_for: impl Fn(usize) -> usize,
) -> Vec<Value> {
    let points: Vec<Point> = swept_values
        .iter()
        .map(|&swept_value| {
            let n_points = n_points_for(swept_value);
            let n_steps = n_steps_for(swept_value);
            let (t_euler, c_euler) = run(case, n_points, n_steps, &EulerSolver::new());
            let (t_rk4, c_rk4) = run(case, n_points, n_steps, &RK4Solver::new());

            // Contrôle direct, indépendant de toute comparaison avec le
            // point suivant — isole la dégénérescence au point exact
            // plutôt que de la découvrir seulement via rsf(400, 800).
            // Direct check, independent of any comparison with the next
            // point — isolates the degeneracy to the exact point rather
            // than only discovering it via rsf(400, 800).
            let euler_nan = c_euler.iter().flatten().any(|v| !v.is_finite());
            let rk4_nan = c_rk4.iter().flatten().any(|v| !v.is_finite());
            if euler_nan || rk4_nan {
                eprintln!(
                    "  !!! {} n_points={} n_species={} n_steps={} : NaN/Inf détecté (euler={} rk4={}) — seuil n_points*n_species={} ({} le seuil 999)",
                    case.name, n_points, case.species.len(), n_steps, euler_nan, rk4_nan,
                    n_points * case.species.len(),
                    if n_points * case.species.len() > 999 { "au-dessus de" } else { "sous" },
                );
            }
            Point {
                swept_value,
                n_points,
                n_steps,
                t_euler,
                c_euler,
                t_rk4,
                c_rk4,
            }
        })
        .collect();

    points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (rsf_euler_self, rsf_rk4_self) = match points.get(i + 1) {
                Some(next) => {
                    let e = max_or_nan((0..case.species.len()).map(|s| {
                        let label = format!(
                            "{}/euler/espece{s}/pts{}vs{}/steps{}vs{}",
                            case.name, p.n_points, next.n_points, p.n_steps, next.n_steps
                        );
                        rsf(
                            &label,
                            &p.t_euler,
                            &p.c_euler[s],
                            &next.t_euler,
                            &next.c_euler[s],
                        )
                    }));
                    let k = max_or_nan((0..case.species.len()).map(|s| {
                        let label = format!(
                            "{}/rk4/espece{s}/pts{}vs{}/steps{}vs{}",
                            case.name, p.n_points, next.n_points, p.n_steps, next.n_steps
                        );
                        rsf(&label, &p.t_rk4, &p.c_rk4[s], &next.t_rk4, &next.c_rk4[s])
                    }));
                    (Some(e), Some(k))
                }
                None => (None, None),
            };

            println!(
                "  {:<24} n_points={:>5}  n_steps={:>7}  self(E)={}  self(RK4)={}",
                case.name,
                p.n_points,
                p.n_steps,
                rsf_euler_self
                    .map(|v| format!("{v:.6}"))
                    .unwrap_or_else(|| "—".into()),
                rsf_rk4_self
                    .map(|v| format!("{v:.6}"))
                    .unwrap_or_else(|| "—".into()),
            );

            json!({
                "swept_value": p.swept_value,
                "n_points": p.n_points,
                "n_steps": p.n_steps,
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

    let mut report = Map::new();

    println!("=== Balayage spatial pur (Δt fixe, très fin) ===\n");
    let mut spatial_report = Map::new();
    for case in &cases {
        println!("{}:", case.name);
        let fixed_n_steps = case.n_steps_baseline * FIXED_N_STEPS_MULTIPLIER;
        let points = sweep_axis(
            case,
            &SPATIAL_MULTIPLIERS,
            |mult| case.n_points_baseline * mult,
            |_mult| fixed_n_steps,
        );
        spatial_report.insert(case.name.to_string(), Value::Array(points));
        println!();
    }
    report.insert("spatial_sweep".to_string(), Value::Object(spatial_report));

    println!(
        "=== Balayage temporel (n_points fixe, résolution spatiale la plus fine ci-dessus) ===\n"
    );
    let mut temporal_report = Map::new();
    for case in &cases {
        println!("{}:", case.name);
        let fixed_n_points = case.n_points_baseline * *SPATIAL_MULTIPLIERS.last().unwrap();
        let points = sweep_axis(
            case,
            &TEMPORAL_MULTIPLIERS,
            |_mult| fixed_n_points,
            |mult| case.n_steps_baseline * mult,
        );
        temporal_report.insert(case.name.to_string(), Value::Array(points));
        println!();
    }
    report.insert("temporal_sweep".to_string(), Value::Object(temporal_report));

    let path = std::env::temp_dir().join("spatial_temporal_convergence.json");
    to_json(&report, path.to_str().unwrap())?;
    println!("Report written to {}", path.display());

    Ok(())
}
