//! Comptage d'allocations Euler vs RK4 (ajout 2026-08-16).
//! *Allocation counting, Euler vs RK4 (added 2026-08-16).*
//!
//! # Objectif / *Objective*
//!
//! `jacobian()` alloue une `DMatrix` par appel (voir `bench_isotherm_evaluation_cost`
//! dans `langmuir_performance.rs`, issue #55). RK4 fait 4 appels à `compute_physics`
//! par pas contre 1 pour Euler — l'hypothèse à vérifier est que le nombre
//! d'allocations suit ce rapport 4:1 exactement, indépendamment du temps (qui, lui,
//! peut être affecté par le cache, la taille des allocations, etc.). Ceci ne
//! nécessite aucune modification de `LangmuirMulti`/`jacobian` : on compte les
//! allocations au niveau du process, pas au niveau de la fonction.
//! *`jacobian()` allocates a `DMatrix` per call (see `bench_isotherm_evaluation_cost`
//! in `langmuir_performance.rs`, issue #55). RK4 makes 4 calls to `compute_physics`
//! per step vs 1 for Euler — the hypothesis to check is that the allocation count
//! follows this 4:1 ratio exactly, independently of time (which can be affected by
//! cache, allocation size, etc.). This requires no change to
//! `LangmuirMulti`/`jacobian`: allocations are counted at the process level, not the
//! function level.*
//!
//! # Limite / *Limitation*
//!
//! Compte TOUTES les allocations du process pendant l'appel à `solve()`, pas
//! seulement celles de `jacobian()` — inclut l'allocation de l'état, de la
//! trajectoire stockée, etc. Le ratio Euler/RK4 reste interprétable (ces autres
//! allocations sont approximativement indépendantes du solveur), mais le nombre
//! absolu par appel n'isole pas `jacobian()` seul.
//! *Counts ALL process allocations during the `solve()` call, not just
//! `jacobian()`'s — includes state allocation, stored trajectory, etc. The
//! Euler/RK4 ratio remains interpretable (these other allocations are
//! approximately solver-independent), but the absolute count per call does not
//! isolate `jacobian()` alone.*
//!
//! # Exécution / *Run*
//!
//! ```bash
//! cargo run --release --example alloc_profiling
//! ```
//!
//! Un seul run suffit : le comptage d'allocations est déterministe (pas de bruit
//! de mesure temporelle), pas besoin de 3 runs ni de sleep inter-groupe.
//! *A single run is enough: allocation counting is deterministic (no timing
//! noise), no need for 3 runs or an inter-group sleep.*

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

use chrom_rs::models::{LangmuirMulti, SpeciesParams, TemporalInjection};
use chrom_rs::physics::PhysicalModel;
use chrom_rs::solver::{
    DomainBoundaries, EulerSolver, RK4Solver, Scenario, Solver, SolverConfiguration,
};

// Constantes TFA dupliquées localement — cet exemple est un binaire indépendant,
// `tfa_multi_2species` etc. sont privées à `langmuir_performance.rs` et ne sont pas
// exportées par la crate.
// TFA constants duplicated locally — this example is an independent binary,
// `tfa_multi_2species` etc. are private to `langmuir_performance.rs` and not
// exported by the crate.
const LAMBDA: f64 = 1.2;
const LANGMUIR_K: f64 = 0.4;
const PORT_NUMBER: u32 = 2;
const POROSITY: f64 = 0.4;
const VELOCITY: f64 = 0.001;
const COLUMN_LENGTH: f64 = 0.25;
const N_POINTS: usize = 100;
const N_STEPS: usize = 1000; // ≈ CFL 0.15 sur la grille TFA de référence

fn build_tfa_2species() -> LangmuirMulti {
    let make_sp = |name: &str| {
        SpeciesParams::new(
            name,
            LAMBDA,
            LANGMUIR_K,
            PORT_NUMBER,
            TemporalInjection::dirac(0.0, 1e-3),
        )
    };
    LangmuirMulti::new(
        vec![make_sp("TFA_A"), make_sp("TFA_B")],
        N_POINTS,
        POROSITY,
        VELOCITY,
        COLUMN_LENGTH,
    )
    .expect("Paramètres TFA toujours valides")
}

fn count_allocs_for<F: FnOnce()>(f: F) -> usize {
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    f();
    ALLOC_COUNT.load(Ordering::Relaxed) - before
}

fn main() {
    chrom_rs::output::register_fonts();

    // ── Euler ────────────────────────────────────────────────────────────
    let model = build_tfa_2species();
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(600.0, N_STEPS);
    let solver = EulerSolver::new();

    let euler_allocs = count_allocs_for(|| {
        solver
            .solve(&scenario, &config)
            .expect("solve Euler échoué");
    });

    // ── RK4 ──────────────────────────────────────────────────────────────
    let model = build_tfa_2species();
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(Box::new(model), boundaries);
    let config = SolverConfiguration::time_evolution(600.0, N_STEPS);
    let solver = RK4Solver::new();

    let rk4_allocs = count_allocs_for(|| {
        solver.solve(&scenario, &config).expect("solve RK4 échoué");
    });

    let ratio = rk4_allocs as f64 / euler_allocs as f64;

    println!("[alloc_profiling] n_points={N_POINTS} n_steps={N_STEPS} (TFA, 2 espèces)");
    println!("[alloc_profiling] Euler : {euler_allocs} allocations");
    println!("[alloc_profiling] RK4   : {rk4_allocs} allocations");
    println!(
        "[alloc_profiling] Ratio RK4/Euler : {ratio:.2} (hypothèse : 4,0 si l'allocation domine)"
    );
}
