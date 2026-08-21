//! Comptage d'instructions déterministe (Iai-Callgrind), ajout 2026-08-16.
//! *Deterministic instruction counting (Iai-Callgrind), added 2026-08-16.*
//!
//! # Pourquoi / *Why*
//!
//! Toutes les mesures de ce fichier et de `langmuir_performance.rs`/
//! `solver_performance.rs` sont en temps réel (Criterion, wall-clock) — sujettes
//! à la dérive documentée dans l'article de performance (section méthode/§4).
//! Iai-Callgrind compte les instructions exécutées (Callgrind), pas le temps :
//! résultat déterministe, un seul run suffit, aucune sensibilité au bruit
//! d'ordonnancement ou à la dérive thermique. Sert de recoupement indépendant
//! sur 3 points précis plutôt que de remplacer les mesures Criterion existantes.
//! *All measurements in this file and in `langmuir_performance.rs`/
//! `solver_performance.rs` are wall-clock (Criterion) — subject to the drift
//! documented in the performance article's methodology section (§4).
//! Iai-Callgrind counts executed instructions (Callgrind), not time:
//! deterministic result, a single run suffices, no sensitivity to scheduling
//! noise or thermal drift. Serves as an independent cross-check on 3 specific
//! points rather than replacing the existing Criterion measurements.*
//!
//! # Points choisis / *Chosen points*
//!
//! 1. `tfa_single` — cas de référence CFL=0,15 (même config que `bench_cfl_stability`)
//! 2. `tfa_2species_pre_threshold` — n_points=100 (série, ops=200 < 999)
//! 3. `tfa_2species_post_threshold` — n_points=600 (parallèle, ops=1200 ≥ 999) —
//!    recoupe directement la sonde RK4 ajoutée à `bench_parallelism_threshold`
//!    (§D des bancs complémentaires) avec une mesure indépendante du wall-clock.
//!
//! # Installation / *Setup*
//!
//! Nécessite `valgrind` installé sur la machine et l'ajout suivant à
//! `Cargo.toml` (la crate `chrom-rs` elle-même, pas ce fichier) — voir le
//! `Cargo.toml` mis à jour du 2026-08-17 pour la version déjà appliquée :
//! *Requires `valgrind` installed on the machine, and the following
//! addition to `Cargo.toml` (the `chrom-rs` crate itself, not this file) —
//! see the 2026-08-17 updated `Cargo.toml` for the already-applied version:*
//!
//! ```toml
//! [dev-dependencies]
//! iai-callgrind = "0.14"
//!
//! [[bench]]
//! name = "iai_solver_comparison"
//! harness = false
//!
//! # Requis : `bench` hérite de `release` (strip = true), qui casse la
//! # localisation des fonctions par Valgrind sans ce correctif — vérifié
//! # contre la doc officielle d'iai-callgrind (Prerequisites), pas supposé.
//! # Required: `bench` inherits from `release` (strip = true), which breaks
//! # Valgrind's function lookup without this fix — verified against
//! # iai-callgrind's own documentation (Prerequisites), not assumed.
//! [profile.bench]
//! inherits = "release"
//! debug = true
//! strip = false
//! ```
//!
//! # Exécution / *Run*
//!
//! ```bash
//! cargo bench --bench iai_solver_comparison
//! ```
//!
//! Un seul run — pas de 3 runs, pas de sleep : c'est tout l'intérêt de cette
//! mesure par rapport aux bancs Criterion du reste du fichier.
//! *A single run — no 3 runs, no sleep: that is the whole point of this
//! measurement relative to the Criterion benches elsewhere in this file.*

use chrom_rs::models::{LangmuirMulti, LangmuirSingle, SpeciesParams, TemporalInjection};
use chrom_rs::physics::PhysicalModel;
use chrom_rs::solver::{
    DomainBoundaries, EulerSolver, RK4Solver, Scenario, Solver, SolverConfiguration,
};
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

// Constantes TFA dupliquées localement — voir la même remarque dans
// examples/alloc_profiling.rs (fonctions privées à langmuir_performance.rs,
// non exportées par la crate).
const LAMBDA: f64 = 1.2;
const LANGMUIR_K: f64 = 0.4;
const PORT_NUMBER: f64 = 2.0;
const PORT_NUMBER_U32: u32 = 2;
const POROSITY: f64 = 0.4;
const VELOCITY: f64 = 0.001;
const COLUMN_LENGTH: f64 = 0.25;
const N_STEPS_REF: usize = 1000; // CFL≈0,15 sur nz=100

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

fn tfa_2species(n_points: usize) -> LangmuirMulti {
    let make_sp = |name: &str| {
        SpeciesParams::new(
            name,
            LAMBDA,
            LANGMUIR_K,
            PORT_NUMBER_U32,
            TemporalInjection::dirac(0.0, 1e-3),
        )
    };
    LangmuirMulti::new(
        vec![make_sp("TFA_A"), make_sp("TFA_B")],
        n_points,
        POROSITY,
        VELOCITY,
        COLUMN_LENGTH,
    )
    .expect("Paramètres TFA toujours valides")
}

// ── Point 1 : cas de référence mono-espèce ─────────────────────────────────

#[library_benchmark]
fn iai_euler_tfa_single() {
    let model = tfa_single(100);
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(600.0, N_STEPS_REF);
    black_box(EulerSolver::new().solve(&scenario, &config).unwrap());
}

#[library_benchmark]
fn iai_rk4_tfa_single() {
    let model = tfa_single(100);
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(600.0, N_STEPS_REF);
    black_box(RK4Solver::new().solve(&scenario, &config).unwrap());
}

// ── Point 2 : 2 espèces, n_points=100 (série, sous le seuil) ───────────────

#[library_benchmark]
fn iai_euler_2species_pre_threshold() {
    let model = tfa_2species(100);
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(600.0, 301); // n_steps mesuré au même point, session 2026-08-16
    black_box(EulerSolver::new().solve(&scenario, &config).unwrap());
}

#[library_benchmark]
fn iai_rk4_2species_pre_threshold() {
    let model = tfa_2species(100);
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(600.0, 301);
    black_box(RK4Solver::new().solve(&scenario, &config).unwrap());
}

// ── Point 3 : 2 espèces, n_points=600 (parallèle, au-dessus du seuil) ──────

#[library_benchmark]
fn iai_euler_2species_post_threshold() {
    let model = tfa_2species(600);
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(600.0, 1800); // n_steps mesuré au même point
    black_box(EulerSolver::new().solve(&scenario, &config).unwrap());
}

#[library_benchmark]
fn iai_rk4_2species_post_threshold() {
    let model = tfa_2species(600);
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(600.0, 1800);
    black_box(RK4Solver::new().solve(&scenario, &config).unwrap());
}

library_benchmark_group!(
    name = solver_comparison;
    benchmarks =
        iai_euler_tfa_single,
        iai_rk4_tfa_single,
        iai_euler_2species_pre_threshold,
        iai_rk4_2species_pre_threshold,
        iai_euler_2species_post_threshold,
        iai_rk4_2species_post_threshold,
);

main!(library_benchmark_groups = solver_comparison);
