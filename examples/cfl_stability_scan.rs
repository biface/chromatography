//! Balayage rapide de stabilité CFL — diagnostic enrichi, hors Criterion.
//! *Fast CFL stability scan — enriched diagnostic, outside Criterion.*
//!
//! # Pourquoi cet exemple / *Why this example*
//!
//! `benches/langmuir_performance.rs::bench_cfl_stability` teste déjà 7
//! valeurs de CFL (0.3 à 3.0), mais deux limites empêchaient de conclure
//! sur la vraie frontière de stabilité :
//!
//! 1. Le sondage embarqué (`is_numerically_stable`) ne teste que NaN/Inf —
//!    un schéma qui a déjà commencé à osciller ou à produire des
//!    concentrations négatives ressort encore `stable=true` tant que les
//!    valeurs restent finies.
//! 2. Trouver la vraie limite en repoussant la grille CFL dans
//!    `bench_cfl_stability` coûte cher : chaque valeur ajoutée, c'est
//!    2 solveurs × 50 échantillons × 3 runs en plus dans une session
//!    Criterion de plusieurs heures — pour une question qui ne nécessite
//!    qu'un seul solve par point.
//!
//! Cet exemple répond aux deux : un diagnostic de stabilité plus riche que
//! NaN/Inf (négativité, oscillations parasites, queue non résorbée), sur
//! un seul passage par valeur de CFL (pas de mesure de temps, donc pas de
//! `warm_up`/`sample_size` Criterion) — de quoi explorer une grille CFL
//! large en quelques secondes plutôt qu'en rallongeant une session de 4h.
//!
//! *`benches/langmuir_performance.rs::bench_cfl_stability` already tests 7
//! CFL values (0.3 to 3.0), but two limitations prevented drawing a
//! conclusion on the actual stability boundary:*
//!
//! *1. The embedded probe (`is_numerically_stable`) only tests for
//!    NaN/Inf — a scheme that has already started oscillating or
//!    producing negative concentrations still reports `stable=true` as
//!    long as the values stay finite.*
//! *2. Finding the real limit by extending the CFL grid inside
//!    `bench_cfl_stability` is expensive: every added value means
//!    2 solvers × 50 samples × 3 runs more inside a multi-hour Criterion
//!    session — for a question that only needs one solve per point.*
//!
//! *This example addresses both: a stability diagnostic richer than
//! NaN/Inf (negativity, spurious oscillations, unresolved tail), on a
//! single pass per CFL value (no timing, hence no Criterion
//! `warm_up`/`sample_size`) — enough to explore a wide CFL grid in
//! seconds instead of extending a 4-hour session.*
//!
//! # Ce que le diagnostic ne remplace pas / *What this diagnostic does not replace*
//!
//! Les seuils ci-dessous (`OSCILLATION_THRESHOLD`, `TAIL_RATIO_THRESHOLD`)
//! sont des heuristiques de repérage, pas une preuve d'instabilité au sens
//! mathématique — ils signalent des candidats à regarder de plus près (à
//! l'œil, ou via `tools/plot_*`), pas un verdict définitif.
//! *The thresholds below (`OSCILLATION_THRESHOLD`, `TAIL_RATIO_THRESHOLD`)
//! are flagging heuristics, not a mathematical proof of instability — they
//! point at candidates worth a closer look (visually, or via
//! `tools/plot_*`), not a final verdict.*
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example cfl_stability_scan
//! ```
//!
//! Écrit aussi un rapport JSON dans le répertoire temporaire système
//! (chemin affiché à la fin), même convention que
//! `examples/stiffness_convergence.rs`. Lu ensuite par
//! `tools/plot_cfl_stability.rs` :
//!
//! ```text
//! cargo run --release --bin plot_cfl_stability
//! ```
//! *Also writes a JSON report to the system temp directory (path printed
//! at the end), same convention as `examples/stiffness_convergence.rs`.
//! Read afterwards by `tools/plot_cfl_stability.rs`.*

use nalgebra::DVector;
use serde_json::{Map, Value, json};

use chrom_rs::models::{LangmuirSingle, TemporalInjection};
use chrom_rs::output::export::{JsonError, to_json};
use chrom_rs::physics::{PhysicalData, PhysicalModel, PhysicalQuantity, PhysicalState};
use chrom_rs::solver::{
    DomainBoundaries, EulerSolver, RK4Solver, Scenario, SimulationResult, Solver,
    SolverConfiguration,
};

// =================================================================================================
// Paramètres TFA — dupliqués depuis benches/langmuir_performance.rs
// TFA parameters — duplicated from benches/langmuir_performance.rs
// =================================================================================================
//
// Duplication délibérée : benches/ est un binaire séparé, les constantes et
// fonctions utilitaires qui y sont définies (fn, pas pub) ne sont pas
// accessibles depuis examples/. Même choix que examples/stiffness_convergence.rs
// et examples/validation_report.rs pour leurs propres dépendances dupliquées.
// *Deliberate duplication: benches/ is a separate binary — the constants and
// helper functions defined there (fn, not pub) aren't reachable from
// examples/. Same choice as examples/stiffness_convergence.rs and
// examples/validation_report.rs for their own duplicated dependencies.*

const LAMBDA: f64 = 1.2;
const LANGMUIR_K: f64 = 0.4;
const PORT_NUMBER: f64 = 2.0;
const POROSITY: f64 = 0.4;
const VELOCITY: f64 = 0.001;
const COLUMN_LENGTH: f64 = 0.25;

const F_E: f64 = (1.0 - POROSITY) / POROSITY;
const U_E: f64 = VELOCITY / POROSITY;

/// N̄ = (1 − ε) · N — voir la note de correction sur `U_EFF_C0` juste en
/// dessous.
/// *N̄ = (1 − ε) · N — see the correction note on `U_EFF_C0` just below.*
const N_BAR: f64 = (1.0 - POROSITY) * PORT_NUMBER;

/// Correction 2026-08-2X : utilisait `PORT_NUMBER` (N) brut au lieu de
/// `N_BAR` (N̄=(1-ε)N) — écart de 14% sur σ(0) (0,25 vs 0,2841 correct),
/// copié depuis `benches/langmuir_performance.rs` avant correction. Voir
/// le commentaire détaillé sur la constante `U_EFF_C0` de ce fichier pour
/// l'historique complet. N'affectait aucune simulation, seulement le
/// label CFL utilisé pour construire `n_steps` (~0,88× le label affiché
/// avant correction).
/// *2026-08-2X correction: used raw `PORT_NUMBER` (N) instead of `N_BAR`
/// (N̄=(1-ε)N) — a 14% discrepancy on σ(0) (0.25 vs the correct 0.2841),
/// copied from `benches/langmuir_performance.rs` before the fix. See that
/// file's `U_EFF_C0` constant for the full history. Affected no
/// simulation, only the CFL label used to build `n_steps` (~0.88× the
/// displayed label before the fix).*
const U_EFF_C0: f64 = U_E / (1.0 + F_E * (LAMBDA + N_BAR * LANGMUIR_K));

const N_POINTS_REF: usize = 100;
const TOTAL_TIME: f64 = 600.0;

fn cfl_to_nsteps(cfl: f64, n_points: usize, total_time: f64) -> usize {
    let dz = COLUMN_LENGTH / n_points as f64;
    let dt_max = cfl * dz / U_EFF_C0;
    (total_time / dt_max).ceil() as usize
}

fn tfa_single(n_points: usize) -> LangmuirSingle {
    LangmuirSingle::new(
        LAMBDA,
        LANGMUIR_K,
        PORT_NUMBER,
        POROSITY,
        VELOCITY,
        COLUMN_LENGTH,
        n_points,
        TemporalInjection::dirac(0.0, 1e-3),
    )
}

fn build_scenario(model: LangmuirSingle, n_steps: usize) -> (Scenario, SolverConfiguration) {
    let initial_state = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial_state);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(TOTAL_TIME, n_steps);
    (scenario, config)
}

// =================================================================================================
// Diagnostic de stabilité enrichi
// Enriched stability diagnostic
// =================================================================================================

/// Nombre de changements de signe de la dérivée spatiale première, au-delà
/// duquel le profil final est considéré suspect (dents de scie plutôt
/// qu'un pic chromatographique lisse à 1-2 extrema).
/// *Number of sign changes in the first spatial derivative beyond which
/// the final profile is flagged as suspect (sawtooth pattern rather than a
/// smooth chromatographic peak with 1-2 extrema).*
const OSCILLATION_THRESHOLD: usize = 4;

/// Ratio ‖C(T)‖₂ / max_t ‖C(t)‖₂ au-delà duquel la queue de la simulation
/// est considérée non résorbée (le pic d'élution devrait redescendre près
/// de zéro avant la fin de la fenêtre de simulation TOTAL_TIME=600s).
/// *Ratio ‖C(T)‖₂ / max_t ‖C(t)‖₂ beyond which the simulation's tail is
/// considered unresolved (the elution peak should have decayed back
/// toward zero before the end of the TOTAL_TIME=600s simulation window).*
const TAIL_RATIO_THRESHOLD: f64 = 0.10;

/// Seuil *relatif* (pas absolu) en-deçà duquel un minimum négatif est
/// ignoré comme bruit d'arrondi flottant plutôt que signalé comme
/// instabilité.
///
/// # Pourquoi relatif et pas `min < 0.0` tout court
/// *Why relative, not plain `min < 0.0`*
///
/// Premier balayage (2026-08-20, cf. run utilisateur) : CFL=1.2/rk4 donnait
/// `min=-7.6e-68` face à `max=2.0e-13` — un négatif 55 ordres de grandeur
/// plus petit que l'échelle du signal, donc du bruit d'arrondi sur une
/// queue déjà quasi nulle, pas une instabilité. Un seuil absolu le
/// signalait à tort. `min` n'est retenu comme négativité réelle que s'il
/// dépasse `NEGATIVE_RELATIVE_TOLERANCE` fois l'échelle du profil
/// (`max(|min|, |max|)`) — voir CFL=1.5/rk4 (`min=-1.03`, `max=0.041`,
/// négatif dominant l'échelle : instabilité réelle, toujours signalée).
/// *First sweep (2026-08-20, per user run): CFL=1.2/rk4 gave
/// `min=-7.6e-68` against `max=2.0e-13` — a negative 55 orders of
/// magnitude smaller than the signal's scale, i.e. rounding noise on an
/// already near-zero tail, not instability. A plain absolute threshold
/// wrongly flagged it. `min` only counts as genuine negativity once it
/// exceeds `NEGATIVE_RELATIVE_TOLERANCE` times the profile's scale
/// (`max(|min|, |max|)`) — see CFL=1.5/rk4 (`min=-1.03`, `max=0.041`,
/// negative dominates the scale: genuine instability, still flagged).*
const NEGATIVE_RELATIVE_TOLERANCE: f64 = 1e-9;

#[derive(Debug)]
struct StabilitySignature {
    all_finite: bool,
    final_min: f64,
    final_max: f64,
    has_negative: bool,
    oscillation_count: usize,
    peak_norm: f64,
    tail_norm: f64,
    tail_ratio: f64,
}

impl StabilitySignature {
    /// Verdict humainement lisible — voir le module doc pour les limites
    /// de cette classification (heuristique, pas une preuve).
    /// *Human-readable verdict — see the module doc for this
    /// classification's limits (heuristic, not a proof).*
    fn verdict(&self) -> &'static str {
        if !self.all_finite {
            "NaN/Inf"
        } else if self.has_negative {
            "SUSPECT (concentration négative)"
        } else if self.oscillation_count > OSCILLATION_THRESHOLD {
            "SUSPECT (oscillations)"
        } else if self.tail_ratio > TAIL_RATIO_THRESHOLD {
            "SUSPECT (queue non résorbée)"
        } else {
            "OK"
        }
    }
}

/// Extrait le vecteur de concentration d'un [`PhysicalState`] mono-espèce.
/// *Extracts the concentration vector from a single-species [`PhysicalState`].*
///
/// Renvoie `None` si la quantité est absente ou n'est pas un [`PhysicalData::Vector`]
/// (ne devrait pas arriver pour [`LangmuirSingle`], mais on ne veut pas paniquer
/// sur un cas mal formé).
/// *Returns `None` if the quantity is absent or not a [`PhysicalData::Vector`]
/// (shouldn't happen for [`LangmuirSingle`], but we don't want to panic on a
/// malformed case.)*
fn concentration_vector(state: &PhysicalState) -> Option<&DVector<f64>> {
    match state.get(PhysicalQuantity::Concentration)? {
        PhysicalData::Vector(v) => Some(v),
        _ => None,
    }
}

fn analyze(result: &SimulationResult) -> StabilitySignature {
    // Fini/NaN sur TOUTE la trajectoire, pas seulement l'état final — plus
    // strict que is_numerically_stable (qui ne regarde que final_state).
    // Finite/NaN over the WHOLE trajectory, not just the final state —
    // stricter than is_numerically_stable (which only looks at final_state).
    let all_finite = result.state_trajectory.iter().all(|state| {
        concentration_vector(state)
            .map(|v| v.iter().all(|x| x.is_finite()))
            .unwrap_or(false)
    });

    let final_vec = concentration_vector(&result.final_state);

    let (final_min, final_max, has_negative, oscillation_count) = match final_vec {
        Some(v) if all_finite => {
            let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            // Échelle du profil pour juger si `min` est un négatif réel ou
            // du bruit d'arrondi flottant (voir la doc de
            // NEGATIVE_RELATIVE_TOLERANCE).
            // Profile scale used to judge whether `min` is genuine
            // negativity or floating-point rounding noise (see
            // NEGATIVE_RELATIVE_TOLERANCE's doc comment).
            let scale = min.abs().max(max.abs()).max(f64::MIN_POSITIVE);
            let has_neg = min < -NEGATIVE_RELATIVE_TOLERANCE * scale;

            // Changements de signe de la dérivée première spatiale
            // (nombre d'extrema locaux dans le profil final).
            // Sign changes of the first spatial derivative (number of
            // local extrema in the final profile).
            let diffs: Vec<f64> = v.as_slice().windows(2).map(|w| w[1] - w[0]).collect();
            let sign_changes = diffs.windows(2).filter(|w| w[0] * w[1] < 0.0).count();

            (min, max, has_neg, sign_changes)
        }
        _ => (f64::NAN, f64::NAN, false, 0),
    };

    let peak_norm = if all_finite {
        result
            .state_trajectory
            .iter()
            .filter_map(concentration_vector)
            .map(|v| v.norm())
            .fold(0.0_f64, f64::max)
    } else {
        f64::NAN
    };

    let tail_norm = if all_finite {
        final_vec.map(|v| v.norm()).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };

    let tail_ratio = if peak_norm > 0.0 {
        tail_norm / peak_norm
    } else {
        0.0
    };

    StabilitySignature {
        all_finite,
        final_min,
        final_max,
        has_negative,
        oscillation_count,
        peak_norm,
        tail_norm,
        tail_ratio,
    }
}

// =================================================================================================
// Balayage
// Sweep
// =================================================================================================

#[derive(Clone, Copy)]
enum SolverKind {
    Euler,
    Rk4,
}

impl SolverKind {
    const fn name(self) -> &'static str {
        match self {
            SolverKind::Euler => "euler",
            SolverKind::Rk4 => "rk4",
        }
    }

    fn run(self, cfl: f64, n_steps: usize) -> SimulationResult {
        let (scenario, config) = build_scenario(tfa_single(N_POINTS_REF), n_steps);
        let outcome = match self {
            SolverKind::Euler => EulerSolver::new().solve(&scenario, &config),
            SolverKind::Rk4 => RK4Solver::new().solve(&scenario, &config),
        };
        outcome.unwrap_or_else(|e| {
            panic!(
                "Échec inattendu du solveur {} (cfl={cfl}) : {e:?} / \
                 Unexpected {} solver failure (cfl={cfl}): {e:?}",
                self.name(),
                self.name()
            )
        })
    }
}

fn main() -> Result<(), JsonError> {
    // Pas d'appel à chrom_rs::output::register_fonts() ici : cet exemple ne
    // produit aucun SVG (register_fonts n'est utile qu'au rendu
    // plotters/html_reports, fait par tools/plot_cfl_stability.rs séparément)
    // — contrairement aux fonctions de benches/langmuir_performance.rs qui
    // l'appellent systématiquement.
    // No call to chrom_rs::output::register_fonts() here: this example
    // produces no SVG itself (register_fonts is only needed for
    // plotters/html_reports rendering, done separately by
    // tools/plot_cfl_stability.rs) — unlike the functions in
    // benches/langmuir_performance.rs, which call it unconditionally.

    // Grille existante (0.3 à 3.0, cf. bench_cfl_stability) prolongée bien
    // au-delà pour trouver la vraie limite — sans le coût Criterion.
    // Existing grid (0.3 to 3.0, cf. bench_cfl_stability) extended well
    // beyond it to find the actual limit — without the Criterion cost.
    let cfl_targets: &[f64] = &[
        0.3, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0,
    ];

    println!(
        "{:>6} {:>9} {:>7} {:>12} {:>12} {:>6} {:>12} {:>12} {:>11}  verdict",
        "CFL", "n_steps", "solveur", "min", "max", "oscil", "peak_norm", "tail_norm", "tail_ratio"
    );

    let mut any_suspect = false;
    let mut rows: Vec<Value> = Vec::with_capacity(cfl_targets.len() * 2);

    for &cfl in cfl_targets {
        let n_steps = cfl_to_nsteps(cfl, N_POINTS_REF, TOTAL_TIME);

        for solver in [SolverKind::Euler, SolverKind::Rk4] {
            let result = solver.run(cfl, n_steps);
            let sig = analyze(&result);
            let verdict = sig.verdict();
            if verdict != "OK" {
                any_suspect = true;
            }

            println!(
                "{:>6.1} {:>9} {:>7} {:>12.4e} {:>12.4e} {:>6} {:>12.4e} {:>12.4e} {:>11.4}  {}",
                cfl,
                n_steps,
                solver.name(),
                sig.final_min,
                sig.final_max,
                sig.oscillation_count,
                sig.peak_norm,
                sig.tail_norm,
                sig.tail_ratio,
                verdict,
            );

            rows.push(json!({
                "cfl": cfl,
                "n_steps": n_steps,
                "solver": solver.name(),
                "all_finite": sig.all_finite,
                "final_min": sig.final_min,
                "final_max": sig.final_max,
                "oscillation_count": sig.oscillation_count,
                "peak_norm": sig.peak_norm,
                "tail_norm": sig.tail_norm,
                "tail_ratio": sig.tail_ratio,
                "verdict": verdict,
            }));
        }
    }

    if any_suspect {
        println!(
            "\n⚠ Au moins un cas signalé au-delà du simple NaN/Inf — voir la colonne verdict.\n\
             ⚠ At least one case flagged beyond plain NaN/Inf — see the verdict column."
        );
    } else {
        println!(
            "\nAucun cas suspect sur la grille testée (même critère élargi) — la limite \
             de stabilité, si elle existe pour ce cas TFA, est au-delà de CFL={}.\n\
             No suspect case on the tested grid (even with the widened criterion) — the \
             stability limit, if it exists for this TFA case, lies beyond CFL={}.",
            cfl_targets.last().unwrap(),
            cfl_targets.last().unwrap()
        );
    }

    let mut report = Map::new();
    report.insert("rows".to_string(), Value::Array(rows));
    report.insert(
        "oscillation_threshold".to_string(),
        json!(OSCILLATION_THRESHOLD),
    );
    report.insert(
        "tail_ratio_threshold".to_string(),
        json!(TAIL_RATIO_THRESHOLD),
    );
    report.insert(
        "negative_relative_tolerance".to_string(),
        json!(NEGATIVE_RELATIVE_TOLERANCE),
    );

    let path = std::env::temp_dir().join("cfl_stability_scan.json");
    to_json(&report, path.to_str().unwrap())?;
    println!("\nReport written to {}", path.display());

    Ok(())
}
