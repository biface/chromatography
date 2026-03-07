//! Benchmarks de performance pour les modèles Langmuir
//! *Performance benchmarks for Langmuir models*
//!
//! # Objectif / *Objective*
//!
//! Ce fichier mesure cinq dimensions indépendantes des performances de
//! `chrom-rs` :
//! *This file measures five independent performance dimensions of `chrom-rs`:*
//!
//! 1. **`bench_cfl_stability`** — Stabilité numérique en fonction du CFL
//!    (Euler et RK4).
//!    *Numerical stability as a function of CFL (Euler and RK4).*
//! 2. **`bench_single_vs_multi_1species`** — Surcoût de [`LangmuirMulti`]
//!    sur un problème mono-espèce (Euler et RK4).
//!    *Overhead of [`LangmuirMulti`] on a single-species problem (Euler and RK4).*
//! 3. **`bench_multi_species_scaling`** — Scalabilité O(n³) de l'inversion
//!    jacobienne pour n_species ∈ {1, 2, 5, 10, 20, 50} (Euler uniquement).
//!    *O(n³) scalability of Jacobian inversion for n_species ∈ {1,2,5,10,20,50} (Euler only).*
//! 4. **`bench_parallelism_threshold`** — Déclenchement du parallélisme rayon
//!    autour du seuil n_points × n_species ≥ 1000.
//!    *Rayon parallelism trigger around the threshold n_points × n_species ≥ 1000.*
//! 5. **`bench_species_response_curve`** — Courbe de réponse complète de
//!    n_species = 2 à 100, **Euler et RK4**, avec n_points = 100 fixé.
//!    *Full response curve from n_species = 2 to 100, **Euler and RK4**, with n_points = 100 fixed.*
//!    Ce groupe croise simultanément :
//!    *This group simultaneously observes:*
//!    - la loi d'échelle O(n³) de l'inversion LU,
//!      *the O(n³) scaling law of LU inversion,*
//!    - le seuil de parallélisme (franchi à n_species = 10 avec n_points = 100),
//!      *the parallelism threshold (crossed at n_species = 10 with n_points = 100),*
//!    - le surcoût RK4 vs Euler sur la plage complète.
//!      *the RK4 vs Euler overhead over the full range.*
//!
//! # Paramètres physiques / *Physical parameters*
//!
//! Tous les groupes sauf les groupes 3 et 5 utilisent les paramètres TFA
//! (validés scientifiquement — Nicoud 2015, Fig. 4).  Les groupes 3 et 5
//! génèrent des paramètres aléatoires reproductibles (seed = 42).
//! *All groups except 3 and 5 use TFA parameters (scientifically validated —
//! Nicoud 2015, Fig. 4). Groups 3 and 5 generate reproducible random parameters (seed = 42).*
//!
//! # Estimation du temps de calcul (groupe 5) / *Compute-time estimation (group 5)*
//!
//! Avant chaque exécution Criterion, `bench_species_response_curve` imprime
//! sur `stderr` un tableau de pré-analyse :
//! *Before each Criterion run, `bench_species_response_curve` prints a
//! pre-analysis table to `stderr`:*
//!
//! ```text
//! [species_curve] n_sp=  2  ops=  200  regime=serial     n_steps= 2400  ratio_O3=  1.00x
//! [species_curve] n_sp= 10  ops= 1000  regime=PARALLEL   n_steps= 2400  ratio_O3=125.00x
//! [species_curve] n_sp=100  ops=10000  regime=PARALLEL   n_steps= 2400  ratio_O3=125000.00x
//! ```
//!
//! La colonne `ratio_O3` est le rapport théorique `(n / n_ref)³` par rapport
//! à n_species = 2.  Si les temps mesurés suivent ce ratio, la complexité
//! O(n³) est confirmée.
//! *The `ratio_O3` column is the theoretical ratio `(n / n_ref)³` relative to
//! n_species = 2.  If measured times follow this ratio, O(n³) complexity is confirmed.*
//!
//! # Exécution / *Running*
//!
//! ```bash
//! cargo bench --bench langmuir_performance
//! # Rapport HTML : target/criterion/
//! # HTML report:   target/criterion/
//! ```
//!
//! # Note sur la détection NaN/Inf / *Note on NaN/Inf detection*
//!
//! Pour les CFL ≥ 1.0, le solveur Euler produit typiquement des NaN.
//! La fonction [`is_numerically_stable`] inspecte l'état final après chaque
//! itération sonde et affiche un avertissement — sans interrompre la mesure
//! du temps (afin que Criterion enregistre même les cas instables).
//! *For CFL ≥ 1.0 the Euler solver typically produces NaN values.
//! The [`is_numerically_stable`] function inspects the final state after each
//! probe iteration and prints a warning — without interrupting the timing
//! (so that Criterion records even unstable cases).*

use std::time::Duration;
use std::hint::black_box;

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode,
};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

use chrom_rs::models::{LangmuirMulti, LangmuirSingle, SpeciesParams, TemporalInjection};
use chrom_rs::physics::{PhysicalData, PhysicalModel, PhysicalQuantity};
use chrom_rs::solver::{
    DomainBoundaries, EulerSolver, RK4Solver, Scenario, SimulationResult,
    SolverConfiguration, Solver,
};

// =================================================================================================
// Constantes physiques TFA partagées
// Shared TFA physical constants
// =================================================================================================

// Paramètres TFA — validés scientifiquement (Nicoud 2015, Fig. 4)
// TFA parameters — scientifically validated (Nicoud 2015, Fig. 4)
//
// Ces valeurs définissent la simulation de référence utilisée dans les groupes 1, 2 et 4.
// These values define the reference simulation used in groups 1, 2 and 4.

/// λ : ordonnée à l'origine de l'isotherm linéaire modifiée \[dimensionless\]
/// *λ : intercept of the modified linear isotherm \[dimensionless\]*
const LAMBDA: f64 = 1.2;

/// K̃ : constante d'équilibre de Langmuir \[L/mol\]
/// *K̃ : Langmuir equilibrium constant \[L/mol\]*
const LANGMUIR_K: f64 = 0.4;

/// N : nombre de sites d'adsorption \[mol/L\]
/// *N : number of adsorption sites \[mol/L\]*
const PORT_NUMBER: f64 = 2.0;

/// ε : porosité extra-granulaire \[dimensionless\]
/// *ε : extra-granular porosity \[dimensionless\]*
const POROSITY: f64 = 0.4;

/// u : vitesse superficielle \[m/s\]
/// *u : superficial velocity \[m/s\]*
const VELOCITY: f64 = 0.001;

/// L : longueur de la colonne \[m\]
/// *L : column length \[m\]*
const COLUMN_LENGTH: f64 = 0.25;

// Valeurs dérivées — calculées une fois pour éviter toute répétition dans les boucles internes.
// Derived values — computed once to avoid redundant arithmetic inside inner loops.
// Les constantes Rust permettent l'évaluation à la compilation (const eval).
// Rust constants enable compile-time evaluation (const eval).

/// F_e = (1 − ε) / ε = 1.5 \[dimensionless\]
///
/// Rapport de phase extra-granulaire.
/// *Extra-granular phase ratio.*
const F_E: f64 = (1.0 - POROSITY) / POROSITY; // 1.5

/// u_e = u / ε = 0.0025 m/s
///
/// Vitesse interstitielle.
/// *Interstitial velocity.*
const U_E: f64 = VELOCITY / POROSITY; // 0.0025

/// u_eff à C=0 = u_e / (1 + F_e · (λ + N·K̃)) ≈ 0.000625 m/s
/// *u_eff at C=0 = u_e / (1 + F_e · (λ + N·K̃)) ≈ 0.000625 m/s*
///
/// Vitesse effective du front de concentration à concentration nulle.
/// *Effective velocity of the concentration front at zero concentration.*
/// C'est la vitesse maximale du front → détermine le CFL le plus contraignant.
/// *This is the maximum front velocity → sets the most restrictive CFL constraint.*
///
/// # Calcul détaillé / *Detailed calculation*
///
/// dérivée isotherm à C=0 : λ + N·K̃ = 1.2 + 2.0 × 0.4 = 2.0
/// *isotherm derivative at C=0 : λ + N·K̃ = 1.2 + 2.0 × 0.4 = 2.0*
/// σ(0) = 1 / (1 + F_e × 2.0) = 1 / 4.0 = 0.25
/// u_eff = σ(0) × u_e = 0.25 × 0.0025 = 0.000625 m/s
const U_EFF_C0: f64 = U_E / (1.0 + F_E * (LAMBDA + PORT_NUMBER * LANGMUIR_K));

// Dimensions du domaine de référence (groupes 1 et 2)
// Reference domain dimensions (groups 1 and 2)
const N_POINTS_REF: usize = 100;
const TOTAL_TIME: f64 = 600.0;
const N_STEPS_REF: usize = 1000;

// =================================================================================================
// Fonctions utilitaires
// Utility functions
// =================================================================================================

/// Convertit un nombre de Courant–Friedrichs–Lewy (CFL) cible en nombre de pas de temps.
/// *Converts a target Courant–Friedrichs–Lewy (CFL) number into a number of time steps.*
///
/// # Formule / *Formula*
///
/// $$N_t = \left\lceil \frac{T}{dt_{max}} \right\rceil \quad
/// \text{avec / with} \quad dt_{max} = \frac{\mathrm{CFL} \cdot \Delta z}{u_{eff}}$$
///
/// # Arguments
///
/// * `cfl`         — CFL cible \[dimensionless\] / *target CFL \[dimensionless\]*
/// * `n_points`    — Nombre de points spatiaux N_z / *number of spatial points N_z*
/// * `total_time`  — Durée totale de simulation \[s\] / *total simulation time \[s\]*
///
/// # Exemple / *Example*
///
/// ```
/// // CFL=0.15 sur la grille TFA de référence (nz=100, T=600 s)
/// // CFL=0.15 on the TFA reference grid (nz=100, T=600 s)
/// // → dt_max = 0.15 × 0.0025 / 0.000625 = 0.6 s → N_t = 1 000
/// let n = cfl_to_nsteps(0.15, 100, 600.0);
/// assert_eq!(n, 1000);
/// ```
fn cfl_to_nsteps(cfl: f64, n_points: usize, total_time: f64) -> usize {
    // Δz = L / N_z  (convention volumes finis : cellules de largeur égale)
    // Δz = L / N_z  (finite volume convention: equal-width cells)
    let dz = COLUMN_LENGTH / n_points as f64;

    // Pas de temps maximal pour le CFL demandé
    // Maximum time step for the requested CFL
    let dt_max = cfl * dz / U_EFF_C0;

    // On arrondit au supérieur pour ne jamais dépasser le CFL cible
    // Ceiling rounding ensures we never exceed the target CFL
    (total_time / dt_max).ceil() as usize
}

// -------------------------------------------------------------------------------------------------
// Groupe 3 — paramètres aléatoires multi-espèces
// Group 3 — random multi-species parameters
// -------------------------------------------------------------------------------------------------

/// Paramètres physiques d'un ensemble d'espèces générés aléatoirement
/// *Physical parameters for a randomly generated set of species*
///
/// Utilisé exclusivement dans [`bench_multi_species_scaling`] et
/// [`bench_species_response_curve`] pour garantir la reproductibilité
/// (seed fixe = 42) tout en couvrant un espace large de configurations.
/// *Used exclusively in [`bench_multi_species_scaling`] and
/// [`bench_species_response_curve`] to guarantee reproducibility (fixed seed = 42)
/// while covering a broad parameter space.*
struct MultiParams {
    /// Valeurs λ_i pour chaque espèce / *λ_i values for each species*
    lambdas: Vec<f64>,
    /// Valeurs K̃_i pour chaque espèce / *K̃_i values for each species*
    ks: Vec<f64>,
    /// Valeurs N_i pour chaque espèce (en f64 pour le calcul CFL)
    /// *N_i values for each species (as f64 for CFL computation)*
    ns: Vec<f64>,
}

/// Génère des paramètres aléatoires reproductibles pour n_species espèces
/// *Generates reproducible random parameters for n_species species*
///
/// Les plages de valeurs sont physiquement raisonnables :
/// *Value ranges are physically reasonable:*
/// - λ ∈ \[0.8, 1.5\] — rétention linéaire modérée / *moderate linear retention*
/// - K̃ ∈ \[0.1, 0.8\] — affinité faible à modérée / *low to moderate affinity*
/// - N ∈ \[1.0, 3.0\] — capacité d'adsorption typique / *typical adsorption capacity*
///
/// # Arguments
///
/// * `n_species` — Nombre d'espèces à générer / *number of species to generate*
/// * `seed`      — Graine RNG pour la reproductibilité (utiliser 42) /
///                 *RNG seed for reproducibility (use 42)*
///
/// # Reproductibilité stricte / *Strict reproducibility*
///
/// `SmallRng::seed_from_u64` garantit la même séquence sur toutes les
/// plateformes, quel que soit l'endianness ou la version de l'OS.
/// *`SmallRng::seed_from_u64` guarantees the same sequence on all platforms,
/// regardless of endianness or OS version.*
fn generate_multi_params(n_species: usize, seed: u64) -> MultiParams {
    let mut rng = SmallRng::seed_from_u64(seed);

    let lambdas = (0..n_species).map(|_| rng.random_range(0.8..1.5)).collect();
    let ks      = (0..n_species).map(|_| rng.random_range(0.1..0.8)).collect();
    let ns      = (0..n_species).map(|_| rng.random_range(1.0..3.0)).collect();

    MultiParams { lambdas, ks, ns }
}

/// Calcule le nombre de pas de temps qui garantit CFL ≤ 0.5 pour tous les
/// paramètres aléatoires d'un système multi-espèces
/// *Computes the number of time steps that guarantees CFL ≤ 0.5 for all
/// random parameters of a multi-species system*
///
/// # Raisonnement / *Reasoning*
///
/// La vitesse effective maximale correspond à la dérivée isotherm *minimale*
/// (moins de rétention → front plus rapide).  On prend donc le minimum de
/// (λ_i + N_i · K̃_i) sur toutes les espèces pour trouver l'espèce la plus
/// mobile, puis on dimensionne le pas de temps en conséquence.
/// *The maximum effective velocity corresponds to the *minimum* isotherm
/// derivative (less retention → faster front).  We therefore take the minimum
/// of (λ_i + N_i · K̃_i) over all species to find the most mobile one, then
/// size the time step accordingly.*
///
/// # CFL conservatif = 0.5 / *Conservative CFL = 0.5*
///
/// Un CFL de 0.5 laisse une marge confortable vis-à-vis du seuil d'instabilité
/// (CFL = 1 pour l'Euler upwind) tout en évitant un sur-raffinement inutile.
/// *A CFL of 0.5 provides a comfortable margin below the instability threshold
/// (CFL = 1 for upwind Euler) while avoiding unnecessary over-refinement.*
fn safe_nsteps_for_multi(params: &MultiParams, n_points: usize, total_time: f64) -> usize {
    // Dérivée isotherm minimale = espèce la plus mobile
    // Minimum isotherm derivative = most mobile species
    let deriv_min = params
        .lambdas.iter()
        .zip(&params.ns)
        .zip(&params.ks)
        .map(|((l, n), k)| l + n * k)
        .fold(f64::INFINITY, f64::min);

    // Vitesse effective maximale du système
    // Maximum effective velocity of the system
    let u_eff_max = U_E / (1.0 + F_E * deriv_min);

    let dz = COLUMN_LENGTH / n_points as f64;

    // Pas de temps maximal pour CFL = 0.5
    // Maximum time step for CFL = 0.5
    let dt_max = 0.5 * dz / u_eff_max;

    (total_time / dt_max).ceil() as usize
}

// -------------------------------------------------------------------------------------------------
// Utilitaires de vérification numérique
// Numerical sanity-check utilities
// -------------------------------------------------------------------------------------------------

/// Vérifie l'absence de NaN ou d'infini dans un résultat de simulation
/// *Checks that a simulation result contains no NaN or infinite values*
///
/// Renvoie `true` si l'état final est numériquement sain, `false` sinon.
/// *Returns `true` if the final state is numerically sound, `false` otherwise.*
///
/// # Usage dans les benchmarks / *Usage in benchmarks*
///
/// Appelée *une fois* avant la boucle de mesure pour afficher l'état de
/// stabilité sans perturber le chronométrage.
/// *Called *once* before the measurement loop to display stability status
/// without biasing the timing.*
fn is_numerically_stable(result: &SimulationResult) -> bool {
    // Récupération du champ de concentration dans l'état final
    // Retrieve the concentration field from the final state
    let Some(data) = result.final_state.get(PhysicalQuantity::Concentration) else {
        return false;
    };

    // Parcours exhaustif selon la forme des données (scalaire, vecteur ou matrice)
    // Exhaustive check based on data shape (scalar, vector or matrix)
    match data {
        PhysicalData::Scalar(x) => x.is_finite(),
        PhysicalData::Vector(v) => v.iter().all(|x| x.is_finite()),
        PhysicalData::Matrix(m) => m.iter().all(|x| x.is_finite()),
        PhysicalData::Array(a)  => a.iter().all(|x| x.is_finite()),
    }
}

// -------------------------------------------------------------------------------------------------
// Constructeurs TFA réutilisables
// Reusable TFA constructors
// -------------------------------------------------------------------------------------------------

/// Construit un [`LangmuirSingle`] avec les paramètres TFA
/// *Builds a [`LangmuirSingle`] using TFA parameters*
///
/// Injection Dirac à t=0 — le type d'injection n'a pas d'importance pour
/// les mesures de performance pure (pas d'évaluation de profil coûteuse).
/// *Dirac injection at t=0 — the injection type is irrelevant for pure
/// performance measurement (no expensive profile evaluation).*
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

/// Construit un [`LangmuirMulti`] à 1 espèce avec les paramètres TFA
/// *Builds a single-species [`LangmuirMulti`] using TFA parameters*
///
/// Équivalent physique de [`tfa_single`], mais implémenté via le code
/// multi-espèces (inversion jacobienne 1×1).
/// *Physical equivalent of [`tfa_single`], but routed through the multi-species
/// code path (1×1 Jacobian inversion).*
fn tfa_multi_1species(n_points: usize) -> LangmuirMulti {
    let sp = SpeciesParams::new(
        "TFA",
        LAMBDA,
        LANGMUIR_K,
        PORT_NUMBER as u32, // u32 requis par SpeciesParams / required by SpeciesParams
        TemporalInjection::dirac(0.0, 1e-3),
    );
    LangmuirMulti::new(vec![sp], n_points, POROSITY, VELOCITY, COLUMN_LENGTH)
        .expect("Paramètres TFA toujours valides / TFA parameters always valid")
}

/// Construit un [`LangmuirMulti`] à 2 espèces TFA identiques
/// *Builds a two-species [`LangmuirMulti`] with identical TFA parameters*
///
/// Utilisé dans [`bench_parallelism_threshold`] pour tester le seuil de
/// parallélisme.  Les deux espèces partagent les paramètres TFA, ce qui
/// garantit un comportement physique reproductible.
/// *Used in [`bench_parallelism_threshold`] to probe the parallelism threshold.
/// Both species share the TFA parameters, guaranteeing reproducible behaviour.*
fn tfa_multi_2species(n_points: usize) -> LangmuirMulti {
    let make_sp = |name: &str| SpeciesParams::new(
        name,
        LAMBDA,
        LANGMUIR_K,
        PORT_NUMBER as u32,
        TemporalInjection::dirac(0.0, 1e-3),
    );
    LangmuirMulti::new(
        vec![make_sp("TFA_A"), make_sp("TFA_B")],
        n_points, POROSITY, VELOCITY, COLUMN_LENGTH,
    )
    .expect("Paramètres TFA toujours valides / TFA parameters always valid")
}

/// Encapsule un modèle dans un [`Scenario`] + [`SolverConfiguration`] prêts à l'emploi
/// *Wraps a model into a ready-to-use [`Scenario`] + [`SolverConfiguration`]*
///
/// # Générique / *Generic*
///
/// Accepte tout type qui implémente [`PhysicalModel`] + `'static` + `Send`.
/// Cela couvre à la fois [`LangmuirSingle`] et [`LangmuirMulti`].
/// *Accepts any type implementing [`PhysicalModel`] + `'static` + `Send`.
/// This covers both [`LangmuirSingle`] and [`LangmuirMulti`].*
fn build_scenario<M>(model: M, n_steps: usize) -> (Scenario, SolverConfiguration)
where
    M: PhysicalModel + 'static,
{
    let initial_state = model.setup_initial_state();
    let boundaries    = DomainBoundaries::temporal(initial_state);
    let scenario      = Scenario::new(Box::new(model), boundaries);
    let config        = SolverConfiguration::time_evolution(TOTAL_TIME, n_steps);
    (scenario, config)
}

/// Construit un [`LangmuirMulti`] à partir d'un [`MultiParams`] aléatoire
/// *Builds a [`LangmuirMulti`] from a random [`MultiParams`]*
///
/// Factorisation utilisée par les groupes 3 et 5 pour éviter de dupliquer
/// la conversion `Vec<SpeciesParams>` dans chaque `b.iter()`.
/// *Factored out for groups 3 and 5 to avoid duplicating the `Vec<SpeciesParams>`
/// conversion inside every `b.iter()` body.*
///
/// # Arguments
///
/// * `params`   — Paramètres physiques (lambdas, ks, ns) /
///                *physical parameters (lambdas, ks, ns)*
/// * `n_points` — Nombre de points spatiaux / *number of spatial points*
///
/// # Panics
///
/// Panique si les paramètres sont physiquement invalides (ne peut pas arriver
/// avec les sorties de [`generate_multi_params`] dans leurs plages documentées).
/// *Panics if parameters are physically invalid — cannot happen with outputs of
/// [`generate_multi_params`] within their documented ranges.*
fn build_multi_from_params(params: &MultiParams, n_points: usize) -> LangmuirMulti {
    let n_species = params.lambdas.len();

    let species: Vec<SpeciesParams> = (0..n_species)
        .map(|i| {
            SpeciesParams::new(
                format!("SP{i}"),
                params.lambdas[i],
                params.ks[i],
                // port_number est u32 dans SpeciesParams ; on arrondit la valeur
                // f64 en [1.0, 3.0) → toujours 1 ou 2, jamais 0.
                // port_number is u32 in SpeciesParams; the f64 in [1.0, 3.0)
                // is rounded → always 1 or 2, never 0.
                params.ns[i].round() as u32,
                // Injection Dirac à t=0 : neutre pour la performance pure
                // Dirac injection at t=0: neutral for pure performance measurement
                TemporalInjection::dirac(0.0, 1e-3),
            )
        })
        .collect();

    LangmuirMulti::new(species, n_points, POROSITY, VELOCITY, COLUMN_LENGTH)
        .expect("Paramètres générés dans les plages valides / \
                 Parameters generated within valid ranges")
}

/// Calcule le ratio théorique O(n³) par rapport à une valeur de référence
/// *Computes the theoretical O(n³) ratio relative to a reference value*
///
/// Permet de prédire à l'avance le facteur d'aggravation du coût d'inversion
/// LU lorsque le nombre d'espèces passe de `n_ref` à `n`.
/// *Allows predicting in advance the LU inversion cost penalty factor when the
/// number of species grows from `n_ref` to `n`.*
///
/// # Arguments
///
/// * `n`     — Nombre d'espèces cible / *target number of species*
/// * `n_ref` — Nombre d'espèces de référence (généralement le premier point de grille) /
///             *reference number of species (typically the first grid point)*
///
/// # Exemple / *Example*
///
/// ```
/// // De 2 à 10 espèces : facteur O(n³) attendu = (10/2)³ = 125
/// // From 2 to 10 species: expected O(n³) factor = (10/2)³ = 125
/// assert_eq!(cubic_ratio(10, 2), 125.0);
/// ```
fn cubic_ratio(n: usize, n_ref: usize) -> f64 {
    // (n / n_ref)³ — coût algébrique de l'inversion LU
    // (n / n_ref)³ — algebraic cost of LU inversion
    let ratio = n as f64 / n_ref as f64;
    ratio * ratio * ratio
}

// =================================================================================================
// Groupe 1 — bench_cfl_stability
// Group  1 — bench_cfl_stability
// =================================================================================================

/// Observe la stabilité numérique en fonction du CFL (Euler et RK4)
/// *Observes numerical stability as a function of CFL (Euler and RK4)*
///
/// # Principe / *Principle*
///
/// Pour un domaine fixe (nz=100, L=0.25 m, T=600 s), on fait varier le
/// nombre de pas de temps pour cibler différentes valeurs du CFL.  Un CFL
/// supérieur à 1 entraîne une instabilité numérique détectée par NaN/Inf.
/// *For a fixed domain (nz=100, L=0.25 m, T=600 s), the number of time steps
/// is varied to target different CFL values.  A CFL above 1 causes numerical
/// instability detected by NaN/Inf in the final state.*
///
/// | CFL cible / *Target CFL* | Comportement / *Behaviour*                  |
/// |--------------------------|---------------------------------------------|
/// | 0.3                      | Stable et précis / *Stable and accurate*    |
/// | 0.8                      | Stable, proche limite / *Stable, near limit*|
/// | 1.0                      | Limite exacte / *Exact limit (unstable)*    |
/// | 1.2                      | Instable / *Unstable (NaN/Inf expected)*    |
///
/// # Configuration Criterion
///
/// Groupe rapide (mono-espèce, nz=100) : `measurement_time=10 s`,
/// `sample_size=50`, `warm_up=3 s`.
/// *Fast group (single-species, nz=100): `measurement_time=10 s`,
/// `sample_size=50`, `warm_up=3 s`.*
fn bench_cfl_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_cfl_stability");

    // Configuration Criterion pour groupe rapide
    // Criterion configuration for a fast group
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    // CFL cibles à tester — couvre régime stable, limite et régime instable
    // Target CFL values — covers stable regime, limit, and unstable regime
    let cfl_targets: &[f64] = &[0.3, 0.8, 1.0, 1.2];

    for &cfl in cfl_targets {
        // Nombre de pas de temps correspondant à ce CFL
        // Number of time steps corresponding to this CFL
        let n_steps = cfl_to_nsteps(cfl, N_POINTS_REF, TOTAL_TIME);

        // Vérification de stabilité *avant* la boucle de mesure afin de
        // renseigner l'opérateur sans perturber le chronométrage.
        // Stability check *before* the measurement loop so as to inform the
        // operator without biasing the timing.
        let probe_model = tfa_single(N_POINTS_REF);
        let (probe_scenario, probe_config) = build_scenario(probe_model, n_steps);
        let probe_result = EulerSolver::new()
            .solve(&probe_scenario, &probe_config)
            .unwrap_or_else(|_| panic!(
                "Échec inattendu de la simulation sonde (cfl={cfl}) / \
                 Unexpected probe simulation failure (cfl={cfl})"
            ));
        eprintln!(
            "[cfl_stability] Euler CFL={cfl:.1} → n_steps={n_steps} → stable={}",
            is_numerically_stable(&probe_result)
        );

        // ── Euler ──────────────────────────────────────────────────────────
        let id = BenchmarkId::new("euler", format!("cfl_{cfl}"));
        group.bench_with_input(id, &n_steps, |b, &ns| {
            b.iter(|| {
                let model = tfa_single(N_POINTS_REF);
                let (scenario, config) = build_scenario(model, ns);
                // black_box évite que le compilateur n'élimine le résultat
                // comme code mort (optimisation agressive).
                // black_box prevents the compiler from eliminating the result
                // as dead code (aggressive optimisation).
                black_box(
                    EulerSolver::new()
                        .solve(&scenario, &config)
                        .ok(), // Err toléré sur cas instables / Err tolerated for unstable cases
                )
            })
        });

        // ── RK4 ────────────────────────────────────────────────────────────
        // Le même CFL est appliqué à RK4 pour comparer la stabilité à coût
        // temporel équivalent.  RK4 dispose d'un domaine de stabilité plus
        // large que l'Euler, ce qui peut le rendre robuste là où l'Euler diverge.
        // The same CFL is applied to RK4 to compare stability at equivalent
        // computational cost.  RK4 has a larger stability domain than Euler,
        // which may keep it stable where Euler diverges.
        let id = BenchmarkId::new("rk4", format!("cfl_{cfl}"));
        group.bench_with_input(id, &n_steps, |b, &ns| {
            b.iter(|| {
                let model = tfa_single(N_POINTS_REF);
                let (scenario, config) = build_scenario(model, ns);
                black_box(RK4Solver::new().solve(&scenario, &config).ok())
            })
        });
    }

    group.finish();
}

// =================================================================================================
// Groupe 2 — bench_single_vs_multi_1species
// Group  2 — bench_single_vs_multi_1species
// =================================================================================================

/// Compare LangmuirSingle et LangmuirMulti sur un problème physiquement identique à 1 espèce
/// *Compares LangmuirSingle and LangmuirMulti on a physically identical single-species problem*
///
/// # Objectif / *Objective*
///
/// Quantifier le surcoût de la généralité : [`LangmuirMulti`] alloue une
/// `DMatrix` 1×1 et effectue une factorisation LU à chaque pas de temps,
/// là où [`LangmuirSingle`] calcule une simple dérivée scalaire.
/// *Quantify the cost of generality: [`LangmuirMulti`] allocates a 1×1
/// `DMatrix` and performs an LU factorisation at every time step, whereas
/// [`LangmuirSingle`] computes a plain scalar derivative.*
///
/// # Résultat attendu / *Expected result*
///
/// Le ratio Single/Multi donne le coût marginal de la généralité pour les
/// cas mono-espèce.  Ce chiffre oriente le conseil utilisateur : préférer
/// [`LangmuirSingle`] dès que le problème ne requiert qu'une seule espèce.
/// *The Single/Multi ratio gives the marginal cost of generality for single-species
/// cases.  This figure guides user advice: prefer [`LangmuirSingle`] whenever
/// the problem involves only one species.*
///
/// # Configuration
///
/// TFA, nz=100, n_steps=1000 (CFL ≈ 0.15), Euler + RK4.
fn bench_single_vs_multi_1species(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_single_vs_multi_1species");

    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    // ── LangmuirSingle — Euler ─────────────────────────────────────────────
    group.bench_function("single/euler", |b| {
        b.iter(|| {
            let model = tfa_single(N_POINTS_REF);
            let (scenario, config) = build_scenario(model, N_STEPS_REF);
            black_box(EulerSolver::new().solve(&scenario, &config).ok())
        })
    });

    // ── LangmuirSingle — RK4 ──────────────────────────────────────────────
    group.bench_function("single/rk4", |b| {
        b.iter(|| {
            let model = tfa_single(N_POINTS_REF);
            let (scenario, config) = build_scenario(model, N_STEPS_REF);
            black_box(RK4Solver::new().solve(&scenario, &config).ok())
        })
    });

    // ── LangmuirMulti (1 espèce) — Euler ──────────────────────────────────
    // Même physique que Single, mais implémenté via le chemin multi-espèces
    // (allocation DMatrix + inversion LU 1×1).
    // Same physics as Single, but routed through the multi-species code path
    // (DMatrix allocation + 1×1 LU inversion).
    group.bench_function("multi_1sp/euler", |b| {
        b.iter(|| {
            let model = tfa_multi_1species(N_POINTS_REF);
            let (scenario, config) = build_scenario(model, N_STEPS_REF);
            black_box(EulerSolver::new().solve(&scenario, &config).ok())
        })
    });

    // ── LangmuirMulti (1 espèce) — RK4 ────────────────────────────────────
    group.bench_function("multi_1sp/rk4", |b| {
        b.iter(|| {
            let model = tfa_multi_1species(N_POINTS_REF);
            let (scenario, config) = build_scenario(model, N_STEPS_REF);
            black_box(RK4Solver::new().solve(&scenario, &config).ok())
        })
    });

    group.finish();
}

// =================================================================================================
// Groupe 3 — bench_multi_species_scaling
// Group  3 — bench_multi_species_scaling
// =================================================================================================

/// Courbe de scalabilité de l'inversion jacobienne en O(n³)
/// *Scalability curve of Jacobian inversion in O(n³)*
///
/// # Objectif / *Objective*
///
/// L'inversion d'une matrice n×n par décomposition LU est en O(n³).
/// Ce benchmark vérifie que le temps de calcul par pas de temps scale
/// effectivement selon cette loi lorsque n_species varie de 1 à 50.
/// *Inverting an n×n matrix by LU decomposition is O(n³).
/// This benchmark verifies that the compute time per time step actually scales
/// according to this law as n_species varies from 1 to 50.*
///
/// # Paramètres aléatoires reproductibles (seed = 42)
/// *Reproducible random parameters (seed = 42)*
///
/// La seed fixe garantit la reproductibilité stricte entre exécutions.
/// *The fixed seed guarantees strict reproducibility across runs.*
///
/// # n_steps adaptatif / *Adaptive n_steps*
///
/// `safe_nsteps_for_multi` garantit CFL ≤ 0.5 quelle que soit la vitesse
/// effective maximale du système aléatoire.
/// *`safe_nsteps_for_multi` guarantees CFL ≤ 0.5 regardless of the maximum
/// effective velocity of the random system.*
///
/// # Solveur / *Solver*
///
/// Euler uniquement — on isole l'effet du nombre d'espèces, pas du solveur.
/// *Euler only — we isolate the effect of the number of species, not the solver.*
///
/// # Configuration Criterion
///
/// Groupe lent (n_species=50, nz=100) : `measurement_time=20 s`,
/// `sample_size=20`, `SamplingMode::Flat`, `warm_up=5 s`.
/// *Slow group (n_species=50, nz=100): `measurement_time=20 s`,
/// `sample_size=20`, `SamplingMode::Flat`, `warm_up=5 s`.*
fn bench_multi_species_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_multi_species_scaling");

    // Configuration pour groupe potentiellement lent (n_species=50)
    // Configuration for a potentially slow group (n_species=50)
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(5));

    // Plage n_species : de 1 à 50 pour observer la courbure O(n³)
    // n_species range: from 1 to 50 to observe O(n³) curvature
    let n_species_list: &[usize] = &[1, 2, 5, 10, 20, 50];

    for &n_species in n_species_list {
        let params  = generate_multi_params(n_species, 42);
        let n_steps = safe_nsteps_for_multi(&params, N_POINTS_REF, TOTAL_TIME);

        eprintln!("[multi_scaling] n_species={n_species} → n_steps={n_steps}");

        let id = BenchmarkId::new("euler", format!("n_species_{n_species}"));

        group.bench_with_input(id, &n_species, |b, &n_sp| {
            b.iter(|| {
                // Re-génération à chaque itération : la construction est O(n_species),
                // négligeable face au corps de simulation O(n_steps × n_points × n³).
                // Re-generation at each iteration: construction is O(n_species),
                // negligible compared to the simulation body O(n_steps × n_points × n³).
                let p = generate_multi_params(n_sp, 42);

                // Construction des SpeciesParams depuis les vecteurs générés
                // Build SpeciesParams from the generated vectors
                let species: Vec<SpeciesParams> = (0..n_sp)
                    .map(|i| SpeciesParams::new(
                        format!("SP{i}"),
                        p.lambdas[i],
                        p.ks[i],
                        p.ns[i].round() as u32, // u32 requis ; arrondi / required; rounded
                        TemporalInjection::dirac(0.0, 1e-3),
                    ))
                    .collect();

                let model = LangmuirMulti::new(
                    species, N_POINTS_REF, POROSITY, VELOCITY, COLUMN_LENGTH,
                )
                .expect("Paramètres aléatoires dans les plages valides / \
                         Random parameters within valid ranges");

                let (scenario, config) = build_scenario(model, n_steps);
                black_box(EulerSolver::new().solve(&scenario, &config).ok())
            })
        });
    }

    group.finish();
}

// =================================================================================================
// Groupe 4 — bench_parallelism_threshold
// Group  4 — bench_parallelism_threshold
// =================================================================================================

/// Observe le déclenchement du parallélisme rayon autour du seuil
/// n_points × n_species ≥ 1000
/// *Observes the triggering of Rayon parallelism around the threshold
/// n_points × n_species ≥ 1000*
///
/// # Seuil de parallélisme / *Parallelism threshold*
///
/// ```text
/// parallel_threshold() == 999
/// → ops = n_points × n_species ≥ 1000  ⟹  rayon prend le relais
/// → ops = n_points × n_species ≥ 1000  ⟹  rayon takes over
/// ```
///
/// # Scan de n_points / *n_points scan*
///
/// n_species est fixé à 2 (espèces TFA identiques).  On fait varier n_points
/// pour faire traverser le seuil.
/// *n_species is fixed at 2 (identical TFA species).  n_points is varied to
/// cross the threshold.*
///
/// | n_points | ops = n_points × 2 | Régime / *Regime*       |
/// |----------|--------------------|-------------------------|
/// | 499      | 998                | Série / *Serial*        |
/// | 500      | 1000               | Parallèle / *Parallel* ←|
/// | 5000     | 10 000             | Parallèle / *Parallel*  |
///
/// # Résultat attendu / *Expected result*
///
/// Deux segments linéaires avec un saut initial au passage du seuil (coût
/// de spawn du pool rayon).
/// *Two linear segments with an initial jump at the threshold crossing (Rayon
/// thread-pool spawn cost).*
///
/// # n_steps adaptatif / *Adaptive n_steps*
///
/// dz diminue quand n_points augmente → dt_max = 0.5 × dz / u_eff.
/// *dz shrinks as n_points grows → dt_max = 0.5 × dz / u_eff.*
///
/// # Configuration Criterion
///
/// Groupe lent (n_points=5000) : `measurement_time=20 s`, `sample_size=20`,
/// `SamplingMode::Flat`, `warm_up=5 s`.
/// *Slow group (n_points=5000): `measurement_time=20 s`, `sample_size=20`,
/// `SamplingMode::Flat`, `warm_up=5 s`.*
fn bench_parallelism_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_parallelism_threshold");

    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(5));

    // Scan autour du seuil 1000 (= 999 + 1)
    // Scan around threshold 1000 (= 999 + 1)
    // Les valeurs 499/500/501 sont critiques pour observer la cassure.
    // Values 499/500/501 are critical to capture the breakpoint.
    // Grille densifiée en trois zones / Densified grid in three zones:
    //   Zone A — Série (50→499)  : quasi-logarithmique pour régression O(n²)
    //   Zone B — Cassure (480→520): grain fin ±20 autour du seuil
    //   Zone C — Parallèle (520→5000) : couverture régulière
    let n_points_list: &[usize] = &[
        // Zone A : série quasi-logarithmique / Serial quasi-log
        50, 75, 100, 150, 200, 300, 350, 400,
        // Zone B : grain fin autour du seuil / Fine grain around threshold
        480, 490, 495, 498, 499,
        500,  // ← seuil exact ops=1000 / exact threshold
        501, 502, 505, 510, 520,
        // Zone C : parallèle régulier / Regular parallel
        600, 700, 800, 1000, 1500, 2000, 3000, 5000,
    ];

    // n_species fixé à 2 pour ce groupe
    // n_species fixed at 2 for this group
    const N_SPECIES: usize = 2;

    for &n_points in n_points_list {
        let ops = n_points * N_SPECIES;

        // Calcul adaptatif du pas de temps pour maintenir la stabilité sur
        // toute la plage de n_points (dz ∝ 1/n_points).
        // Adaptive time step to maintain stability across the full n_points
        // range (dz ∝ 1/n_points).
        let dz = COLUMN_LENGTH / n_points as f64;

        // u_eff pour 2 espèces TFA identiques = même valeur que 1 espèce
        // u_eff for 2 identical TFA species = same value as 1 species
        let n_steps = {
            let dt_max = 0.5 * dz / U_EFF_C0;
            (TOTAL_TIME / dt_max).ceil() as usize
        };

        eprintln!(
            "[parallelism] n_points={n_points} → ops={ops} → n_steps={n_steps} → régime={}",
            if ops >= 1000 { "parallèle/parallel" } else { "série/serial" }
        );

        let id = BenchmarkId::new("euler", format!("npts_{n_points}"));

        group.bench_with_input(id, &n_points, |b, &npts| {
            b.iter(|| {
                let model = tfa_multi_2species(npts);
                let (scenario, config) = build_scenario(model, n_steps);
                black_box(EulerSolver::new().solve(&scenario, &config).ok())
            })
        });
    }

    group.finish();
}

// =================================================================================================
// Groupe 5 — bench_species_response_curve
// Group  5 — bench_species_response_curve
// =================================================================================================

/// Courbe de réponse complète : temps de calcul en fonction de n_species (2–100)
/// *Full response curve: compute time as a function of n_species (2–100)*
///
/// # Objectif triple / *Three-fold objective*
///
/// Ce groupe est le seul à observer simultanément trois effets :
/// *This group is the only one to simultaneously observe three effects:*
///
/// ## 1. Loi d'échelle O(n³) de l'inversion LU
/// ## *1. O(n³) scaling law of LU inversion*
///
/// À chaque pas de temps et pour chaque point spatial, le solveur inverse la
/// matrice `(I + F_e · M)` de taille `n_species × n_species`.  La
/// factorisation LU est en O(n³) : le tracé log-log doit montrer une droite
/// de pente 3.
/// *At every time step and every spatial point, the solver inverts the
/// `(I + F_e · M)` matrix of size `n_species × n_species`.  LU factorisation
/// is O(n³): the log-log plot must show a straight line of slope 3.*
///
/// Le tableau de pré-analyse (`stderr`) affiche le ratio théorique
/// `(n / n_ref)³` avec `n_ref = 2` pour chaque point de grille.
/// *The pre-analysis table (`stderr`) shows the theoretical ratio `(n / n_ref)³`
/// with `n_ref = 2` for each grid point.*
///
/// ## 2. Seuil de parallélisme rayon (n_points = 100 fixé)
/// ## *2. Rayon parallelism threshold (n_points = 100 fixed)*
///
/// Avec `n_points = 100`, la condition `n_points × n_species ≥ 1000`
/// est satisfaite à partir de **n_species = 10**.  Une *rupture de pente*
/// est visible à ce point : le parallélisme absorbe une partie du coût O(n³)
/// mais introduit un saut fixe dû au spawn du pool de threads.
/// *With `n_points = 100`, the condition `n_points × n_species ≥ 1000`
/// is met starting at **n_species = 10**.  A *slope break* is visible at
/// this point: parallelism absorbs part of the O(n³) cost but introduces a
/// fixed jump from the Rayon thread-pool spawn.*
///
/// ```text
/// n_species =  9 → ops =  900 → série     (pas de thread spawn / no thread spawn)
/// n_species = 10 → ops = 1000 → PARALLÈLE ← cassure / breakpoint expected
/// ```
///
/// ## 3. Surcoût RK4 vs Euler / *3. RK4 overhead vs Euler*
///
/// RK4 effectue **4 appels à `compute_physics`** par pas de temps contre 1
/// pour Euler.  On s'attend donc à :
/// *RK4 makes **4 calls to `compute_physics`** per time step vs 1 for Euler.
/// We therefore expect:*
/// - Rapport RK4/Euler ≈ 4× sur toute la plage de n_species.
///   *RK4/Euler ratio ≈ 4× over the full n_species range.*
/// - Les deux courbes **parallèles** en log-log (même pente O(n³)).
///   *Both curves **parallel** in log-log (same O(n³) slope).*
/// - La cassure du parallélisme présente *sur les deux courbes* au même n_species.
///   *Parallelism breakpoint present *on both curves* at the same n_species.*
///
/// Si le rapport s'éloigne de 4, cela signale des coûts cachés (allocations,
/// copies d'état) qui mériteraient une optimisation.
/// *If the ratio deviates significantly from 4, it signals hidden costs
/// (allocations, state copies) that would warrant optimisation.*
///
/// # Configuration
///
/// - `n_points = 100` (fixé / *fixed*) — choix délibéré pour croiser le seuil à n=10
///   *deliberate choice to cross the threshold at n=10*
/// - `n_steps` adaptatif / *adaptive* — CFL ≤ 0.5 garanti via `safe_nsteps_for_multi`
///   *CFL ≤ 0.5 guaranteed via `safe_nsteps_for_multi`*
/// - Paramètres aléatoires, seed = 42 (identiques au groupe 3)
///   *Random parameters, seed = 42 (identical to group 3)*
/// - Solveurs / *Solvers*: Euler + RK4
///
/// # Lecture des résultats Criterion / *Reading Criterion results*
///
/// Les rapports HTML dans `target/criterion/bench_species_response_curve/`
/// contiennent les séries `euler/n_sp_N` et `rk4/n_sp_N`.
/// Tracer log(temps) vs log(n_species) permet de vérifier la pente O(n³)
/// et la cassure à n=10.
/// *HTML reports in `target/criterion/bench_species_response_curve/` contain
/// the `euler/n_sp_N` and `rk4/n_sp_N` series.
/// Plotting log(time) vs log(n_species) allows visual verification of the
/// O(n³) slope and the breakpoint at n=10.*
///
/// # Pourquoi trois groupes ? / *Why three groups?*
///
/// Criterion 0.8 impose `sample_size >= 10` (assertion interne).
/// `measurement_time` est le budget **par point individuel**, pas pour le groupe
/// entier. Avec `SamplingMode::Flat`, Criterion exécute exactement `sample_size`
/// itérations quel que soit le temps écoulé.
///
/// *Criterion 0.8 enforces `sample_size >= 10` (internal assertion).
/// `measurement_time` is the budget **per individual point**, not the whole group.
/// With `SamplingMode::Flat`, Criterion runs exactly `sample_size` iterations
/// regardless of elapsed time.*
///
/// Budgets estimés / *Estimated budgets*:
///
/// | Groupe / *Group*       | n_species      | sample_size | Budget estimé |
/// |------------------------|----------------|-------------|---------------|
/// | `_small`               | 2 → 30         | 20          | ~15 min       |
/// | `_medium`              | 50, 75         | 10          | ~1 h          |
/// | `_xl`  (nuit / night)  | 100 (Euler)    | 10          | ~2 h          |
///
/// RK4 à n=100 (~2 300 s/iter × 10 = 6 h) est volontairement exclu — à lancer
/// manuellement avec un filtre si besoin.
/// *RK4 at n=100 (~2,300 s/iter × 10 = 6 h) is intentionally excluded — run
/// manually with a filter if needed.*
///
/// Exécution recommandée / *Recommended execution*:
///
/// ```bash
/// # Rapide (quelques minutes) / Fast (a few minutes)
/// cargo bench --bench langmuir_performance -- bench_species_response_curve_small
///
/// # Moyen (prévoir ~1h) / Medium (~1 h)
/// cargo bench --bench langmuir_performance -- bench_species_response_curve_medium
///
/// # Lent, lancer le soir / Slow, run overnight
/// cargo bench --bench langmuir_performance -- bench_species_response_curve_xl
///
/// # RK4 n=100 uniquement si machine dédiée / RK4 n=100 only on dedicated machine
/// cargo bench --bench langmuir_performance -- "bench_species_response_curve_xl/rk4/n_sp_100"
/// ```
fn bench_species_response_curve(c: &mut Criterion) {
    // Ce point d'entrée délègue aux trois sous-groupes.
    // La séparation permet de changer sample_size indépendamment par tranche.
    // This entry point delegates to the three sub-groups.
    // The split allows changing sample_size independently per slice.
    bench_species_response_curve_small(c);
    bench_species_response_curve_medium(c);
    bench_species_response_curve_xl(c);
}

/// Groupe rapide — n_species ∈ {2..30}, `sample_size=20`
/// *Fast group — n_species ∈ {2..30}, `sample_size=20`*
fn bench_species_response_curve_small(c: &mut Criterion) {
    // Grille densifiée en trois zones / Densified grid in three zones:
    //
    // Zone A — Série (2→9) : chaque entier pour une régression O(n³) précise
    //   Zone A — Serial (2→9): every integer for a precise O(n³) regression
    //
    // Zone B — Grain fin autour du seuil n=10 (8,9 déjà dans la zone A)
    //   Zone B — Fine grain around threshold n=10 (8,9 already in zone A)
    //
    // Zone C — Parallèle (12→30) : intermédiaires logarithmiques n=18, n=25
    //   Zone C — Parallel (12→30): logarithmic intermediates n=18, n=25
    run_species_curve_group(c, "bench_species_response_curve_small",
        &[
            // Zone A : série, chaque entier / Serial, every integer
            2, 3, 4, 5, 6, 7, 8, 9,
            // Zone B : grain fin seuil (10, 11, 12) / Fine grain threshold
            10, 11, 12,
            // Zone C : parallèle log-espacé / Log-spaced parallel
            15, 18, 20, 25, 30,
        ],
        20, 5,
        true,  // inclure RK4 / include RK4
    );
}

/// Groupe moyen — n_species ∈ {50, 75}, `sample_size=10`
/// *Medium group — n_species ∈ {50, 75}, `sample_size=10`*
///
/// Budgets : Euler≈15 min, RK4≈40 min par espèce.
/// *Budgets: Euler≈15 min, RK4≈40 min per species.*
fn bench_species_response_curve_medium(c: &mut Criterion) {
    run_species_curve_group(c, "bench_species_response_curve_medium",
        &[50, 75],
        10, 1,
        true,  // inclure RK4 / include RK4
    );
}

/// Groupe lourd — n_species=100, Euler uniquement, `sample_size=10`
/// *Heavy group — n_species=100, Euler only, `sample_size=10`*
///
/// Euler à n=100 : ~586 s/iter × 10 ≈ 1 h 37.
/// RK4   à n=100 : ~2 300 s/iter × 10 ≈ 6 h 24 → exclu par défaut.
///
/// *Euler at n=100: ~586 s/iter × 10 ≈ 1 h 37.*
/// *RK4   at n=100: ~2,300 s/iter × 10 ≈ 6 h 24 → excluded by default.*
///
/// Pour lancer RK4 explicitement / *To run RK4 explicitly*:
/// ```bash
/// cargo bench --bench langmuir_performance -- "bench_species_response_curve_xl/rk4"
/// ```
fn bench_species_response_curve_xl(c: &mut Criterion) {
    run_species_curve_group(c, "bench_species_response_curve_xl",
        &[100],
        10, 1,
        false, // RK4 exclu par défaut / RK4 excluded by default
    );
}

/// Boucle Euler (+ RK4 optionnel) partagée par les trois groupes
/// *Euler (+ optional RK4) loop shared by all three groups*
///
/// # Arguments
///
/// * `include_rk4` — si `false`, seul Euler est mesuré (groupes très lents)
///   *if `false`, only Euler is measured (very slow groups)*
fn run_species_curve_group(
    c: &mut Criterion,
    group_name: &str,
    n_species_list: &[usize],
    sample_size: usize,
    warm_up_secs: u64,
    include_rk4: bool,
) {
    let mut group = c.benchmark_group(group_name);

    // SamplingMode::Flat : exactement `sample_size` itérations, sans adaptation.
    // Sans Flat, le mode Auto peut lancer des milliers de samples sur les petits points.
    // SamplingMode::Flat: exactly `sample_size` iterations, no auto-scaling.
    // Without Flat, Auto mode may run thousands of samples on small points.
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(sample_size);   // ≥ 10 obligatoire depuis Criterion 0.8
    group.warm_up_time(Duration::from_secs(warm_up_secs));

    const N_POINTS: usize = 100;
    const N_REF:    usize = 2;

    // ── Pré-analyse stderr ─────────────────────────────────────────────────
    eprintln!(
        "
[{group_name}] Pré-analyse / Pre-analysis          — n_points={N_POINTS}, sample_size={sample_size}, rk4={include_rk4}
{:-<80}", ""
    );
    eprintln!("{:<12} {:<8} {:<20} {:<12} {:<14}", "n_species", "ops", "régime/regime", "n_steps", "ratio_O3(×)");
    eprintln!("{:-<80}", "");
    for &n_sp in n_species_list {
        let ops     = N_POINTS * n_sp;
        let regime  = if ops >= 1000 { "PARALLÈLE/PARALLEL" } else { "série/serial" };
        let params  = generate_multi_params(n_sp, 42);
        let n_steps = safe_nsteps_for_multi(&params, N_POINTS, TOTAL_TIME);
        let ratio   = cubic_ratio(n_sp, N_REF);
        eprintln!("{:<12} {:<8} {:<20} {:<12} {:<14.1}", n_sp, ops, regime, n_steps, ratio);
    }
    eprintln!("{:-<80}
", "");

    // ── Boucle de benchmarks ───────────────────────────────────────────────
    for &n_sp in n_species_list {
        let params  = generate_multi_params(n_sp, 42);
        let n_steps = safe_nsteps_for_multi(&params, N_POINTS, TOTAL_TIME);

        // ── Euler ──────────────────────────────────────────────────────────
        let euler_id = BenchmarkId::new("euler", format!("n_sp_{n_sp}"));
        group.bench_with_input(euler_id, &n_sp, |b, &n| {
            b.iter(|| {
                let p     = generate_multi_params(n, 42);
                let model = build_multi_from_params(&p, N_POINTS);
                let (scenario, config) = build_scenario(model, n_steps);
                black_box(EulerSolver::new().solve(&scenario, &config).ok())
            })
        });

        // ── RK4 (conditionnel) ─────────────────────────────────────────────
        // RK4 est exclu pour les groupes dont le coût dépasse ~6 h.
        // RK4 is excluded for groups whose cost exceeds ~6 h.
        if include_rk4 {
            let rk4_id = BenchmarkId::new("rk4", format!("n_sp_{n_sp}"));
            group.bench_with_input(rk4_id, &n_sp, |b, &n| {
                b.iter(|| {
                    let p     = generate_multi_params(n, 42);
                    let model = build_multi_from_params(&p, N_POINTS);
                    let (scenario, config) = build_scenario(model, n_steps);
                    black_box(RK4Solver::new().solve(&scenario, &config).ok())
                })
            });
        }
    }

    group.finish();
}

// Fonction fantôme — ne compile pas si appelée, sert uniquement de
// documentation pour le cas RK4/n=100.
// Ghost function — does not compile if called, serves only as
// documentation for the RK4/n=100 case.
#[allow(dead_code)]
fn _note_rk4_n100() {
    // Pour lancer RK4 à n=100 explicitement :
    // To run RK4 at n=100 explicitly:
    //   cargo bench --bench langmuir_performance -- "bench_species_response_curve_xl/rk4/n_sp_100"
    // Budget estimé / Estimated budget : ~6 h 24 (10 samples × ~2 300 s)
}


// =================================================================================================
// Enregistrement Criterion
// Criterion registration
// =================================================================================================

criterion_group!(
    langmuir_benches,
    bench_cfl_stability,
    bench_single_vs_multi_1species,
    bench_multi_species_scaling,
    bench_parallelism_threshold,
    bench_species_response_curve,
);

criterion_main!(langmuir_benches);

// =================================================================================================
// Tests unitaires des fonctions utilitaires
// Unit tests for utility functions
//
// Les benchmarks Criterion ne sont pas testables par `cargo test`, mais les
// fonctions utilitaires (cfl_to_nsteps, generate_multi_params, …) le sont.
// Criterion benchmarks cannot be tested with `cargo test`, but the utility
// functions (cfl_to_nsteps, generate_multi_params, …) can and should be.
//
// Ce module garantit une couverture 80–90 % des chemins de code non-Criterion.
// This module ensures 80–90% coverage of non-Criterion code paths.
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── cfl_to_nsteps ─────────────────────────────────────────────────────────

    /// Vérifie le cas de référence documenté dans le briefing
    /// *Verifies the reference case documented in the briefing*
    ///
    /// CFL=0.15, nz=100, T=600 s → dt_max=0.6 s → N_t=1000
    #[test]
    fn test_cfl_to_nsteps_reference_case() {
        // Δz = 0.25/100 = 0.0025 m
        // dt_max = 0.15 × 0.0025 / 0.000625 = 0.6 s
        // n_steps = ceil(600/0.6) = 1000
        let n = cfl_to_nsteps(0.15, 100, 600.0);
        assert_eq!(n, 1000,
            "Cas de référence CFL=0.15 doit donner N_t=1000 / \
             Reference case CFL=0.15 must give N_t=1000");
    }

    /// Un CFL plus grand → pas de temps plus grand → *moins* de pas
    /// *A larger CFL → larger time step → *fewer* steps*
    #[test]
    fn test_cfl_larger_means_fewer_steps() {
        let n_small_cfl = cfl_to_nsteps(0.3, 100, 600.0);
        let n_large_cfl = cfl_to_nsteps(0.8, 100, 600.0);
        assert!(n_small_cfl > n_large_cfl,
            "CFL plus petit doit donner plus de pas / \
             Smaller CFL must give more time steps");
    }

    /// Le nombre de pas est strictement positif pour tout CFL positif
    /// *Step count is strictly positive for any positive CFL*
    #[test]
    fn test_cfl_positive_nsteps() {
        let n = cfl_to_nsteps(0.3, 100, 600.0);
        assert!(n > 0,
            "Le nombre de pas doit être strictement positif / \
             Step count must be strictly positive");
    }

    /// Plus de points spatiaux → Δz plus petit → plus de pas pour le même CFL
    /// *More spatial points → smaller Δz → more steps for the same CFL*
    #[test]
    fn test_more_points_means_more_steps() {
        let n_coarse = cfl_to_nsteps(0.5, 50, 600.0);
        let n_fine   = cfl_to_nsteps(0.5, 200, 600.0);
        assert!(n_fine > n_coarse,
            "Grille plus fine doit requérir plus de pas pour le même CFL / \
             Finer grid must require more steps for the same CFL");
    }

    // ── generate_multi_params ─────────────────────────────────────────────────

    /// La taille des vecteurs retournés doit correspondre à n_species
    /// *The size of the returned vectors must match n_species*
    #[test]
    fn test_generate_multi_params_length() {
        let p = generate_multi_params(5, 42);
        assert_eq!(p.lambdas.len(), 5);
        assert_eq!(p.ks.len(), 5);
        assert_eq!(p.ns.len(), 5);
    }

    /// Reproductibilité stricte : deux appels avec la même seed → mêmes valeurs
    /// *Strict reproducibility: two calls with the same seed → identical values*
    #[test]
    fn test_generate_multi_params_reproducible() {
        let p1 = generate_multi_params(10, 42);
        let p2 = generate_multi_params(10, 42);
        assert_eq!(p1.lambdas, p2.lambdas,
            "Lambdas doivent être identiques / Lambdas must be identical");
        assert_eq!(p1.ks, p2.ks,
            "Ks doivent être identiques / Ks must be identical");
        assert_eq!(p1.ns, p2.ns,
            "Ns doivent être identiques / Ns must be identical");
    }

    /// Seeds différentes → résultats différents
    /// *Different seeds → different results*
    #[test]
    fn test_generate_multi_params_different_seeds() {
        let p1 = generate_multi_params(5, 42);
        let p2 = generate_multi_params(5, 99);
        // Il est astronomiquement improbable que les 5 lambdas soient identiques
        // It is astronomically unlikely that all 5 lambdas are identical
        assert_ne!(p1.lambdas, p2.lambdas,
            "Seeds différentes → valeurs différentes / Different seeds → different values");
    }

    /// Les valeurs générées sont dans les plages physiques attendues
    /// *Generated values are within expected physical ranges*
    #[test]
    fn test_generate_multi_params_ranges() {
        let p = generate_multi_params(20, 42);
        for &l in &p.lambdas { assert!(l >= 0.8 && l < 1.5, "λ hors plage/out of range: {l}"); }
        for &k in &p.ks      { assert!(k >= 0.1 && k < 0.8, "K̃ hors plage/out of range: {k}"); }
        for &n in &p.ns      { assert!(n >= 1.0 && n < 3.0, "N hors plage/out of range: {n}"); }
    }

    // ── safe_nsteps_for_multi ─────────────────────────────────────────────────

    /// Le résultat doit être strictement positif
    /// *The result must be strictly positive*
    #[test]
    fn test_safe_nsteps_positive() {
        let p = generate_multi_params(3, 42);
        let n = safe_nsteps_for_multi(&p, 100, 600.0);
        assert!(n > 0);
    }

    /// Grille plus fine → CFL plus contraignant → plus de pas de temps
    /// *Finer grid → more restrictive CFL → more time steps*
    #[test]
    fn test_safe_nsteps_finer_grid_more_steps() {
        let p        = generate_multi_params(3, 42);
        let n_coarse = safe_nsteps_for_multi(&p, 50, 600.0);
        let n_fine   = safe_nsteps_for_multi(&p, 200, 600.0);
        assert!(n_fine > n_coarse,
            "Grille plus fine doit requérir plus de pas / \
             Finer grid must require more time steps");
    }

    // ── Constantes dérivées / Derived constants ───────────────────────────────

    /// Vérifie les valeurs dérivées documentées dans le briefing
    /// *Verifies the derived values documented in the briefing*
    #[test]
    fn test_derived_constants() {
        // F_E = (1-0.4)/0.4 = 1.5
        assert!((F_E - 1.5).abs() < 1e-12,
            "F_E doit être 1.5 / F_E must be 1.5");

        // U_E = 0.001/0.4 = 0.0025
        assert!((U_E - 0.0025).abs() < 1e-12,
            "U_E doit être 0.0025 m/s / U_E must be 0.0025 m/s");

        // U_EFF_C0 = 0.0025 / (1 + 1.5 × 2.0) = 0.000625
        assert!((U_EFF_C0 - 0.000625).abs() < 1e-12,
            "U_EFF_C0 doit être 0.000625 m/s / U_EFF_C0 must be 0.000625 m/s");
    }

    // ── Constructeurs TFA / TFA constructors ──────────────────────────────────

    #[test]
    fn test_tfa_single_builds_correctly() {
        let model = tfa_single(100);
        assert_eq!(model.spatial_points(), 100);
        assert!((model.length() - COLUMN_LENGTH).abs() < 1e-12);
    }

    #[test]
    fn test_tfa_multi_1species_builds_correctly() {
        let model = tfa_multi_1species(100);
        assert_eq!(model.n_species(), 1);
        assert_eq!(model.points(), 100);
    }

    #[test]
    fn test_tfa_multi_2species_builds_correctly() {
        let model = tfa_multi_2species(100);
        assert_eq!(model.n_species(), 2);
        assert_eq!(model.points(), 100);
    }

    // ── build_multi_from_params ───────────────────────────────────────────────

    /// La fonction doit produire un modèle avec le bon nombre d'espèces
    /// *The function must produce a model with the correct number of species*
    #[test]
    fn test_build_multi_from_params_species_count() {
        let p     = generate_multi_params(7, 42);
        let model = build_multi_from_params(&p, 50);
        assert_eq!(model.n_species(), 7);
    }

    /// Le nombre de points spatiaux doit être transmis fidèlement
    /// *The number of spatial points must be passed through faithfully*
    #[test]
    fn test_build_multi_from_params_n_points() {
        let p     = generate_multi_params(3, 42);
        let model = build_multi_from_params(&p, 150);
        assert_eq!(model.points(), 150);
    }

    /// Deux appels avec la même seed doivent produire des modèles de structure identique
    /// *Two calls with the same seed must produce models with identical structure*
    #[test]
    fn test_build_multi_from_params_reproducible() {
        let p1 = generate_multi_params(5, 42);
        let p2 = generate_multi_params(5, 42);
        let m1 = build_multi_from_params(&p1, 100);
        let m2 = build_multi_from_params(&p2, 100);
        // On ne peut pas comparer les modèles directement, mais leur structure doit être identique.
        // We cannot compare models directly, but their structure must be identical.
        assert_eq!(m1.n_species(), m2.n_species());
        assert_eq!(m1.points(), m2.points());
    }

    // ── cubic_ratio ───────────────────────────────────────────────────────────

    /// (2/2)³ = 1 — référence par rapport à elle-même / *reference against itself*
    #[test]
    fn test_cubic_ratio_same_value_is_one() {
        assert!((cubic_ratio(2, 2) - 1.0).abs() < 1e-12);
    }

    /// (10/2)³ = 125 — cas documenté dans le module doc / *case documented in module doc*
    #[test]
    fn test_cubic_ratio_2_to_10() {
        assert!((cubic_ratio(10, 2) - 125.0).abs() < 1e-12,
            "Ratio O(n³) de 2→10 doit être 125 / O(n³) ratio from 2→10 must be 125");
    }

    /// (100/2)³ = 125 000 — plafond de la courbe de réponse / *ceiling of the response curve*
    #[test]
    fn test_cubic_ratio_2_to_100() {
        assert!((cubic_ratio(100, 2) - 125_000.0).abs() < 1e-6);
    }

    /// Le ratio est strictement croissant avec n
    /// *The ratio is strictly increasing with n*
    #[test]
    fn test_cubic_ratio_is_monotonic() {
        let r1 = cubic_ratio(10, 2);
        let r2 = cubic_ratio(20, 2);
        let r3 = cubic_ratio(50, 2);
        assert!(r1 < r2 && r2 < r3,
            "cubic_ratio doit être strictement croissant / \
             cubic_ratio must be strictly increasing");
    }

    // ── Vérification du seuil de parallélisme pour n_points=100 ──────────────
    // ── Parallelism threshold verification for n_points=100 ──────────────────

    /// Avec n_points=100, le seuil ops≥1000 doit être franchi exactement à
    /// n_species=10, pas avant.
    /// *With n_points=100, the ops≥1000 threshold must be crossed exactly at
    /// n_species=10, not before.*
    #[test]
    fn test_parallelism_threshold_crossed_at_n_species_10() {
        let threshold = chrom_rs::solver::parallel_threshold(); // 999
        let n_points  = 100_usize;

        // n_species=9 → ops=900 → sous le seuil / below the threshold
        assert!(n_points * 9  <= threshold,
            "n_sp=9  doit être sous le seuil / must be below threshold");
        // n_species=10 → ops=1000 → au-dessus du seuil / above the threshold
        assert!(n_points * 10 >  threshold,
            "n_sp=10 doit dépasser le seuil / must exceed threshold");
    }

    /// Une simulation TFA stable (CFL=0.15) doit produire un état final fini
    /// *A stable TFA simulation (CFL=0.15) must produce a finite final state*
    #[test]
    fn test_is_numerically_stable_on_stable_simulation() {
        let model = tfa_single(100);
        let (scenario, config) = build_scenario(model, 1000);
        let result = EulerSolver::new()
            .solve(&scenario, &config)
            .expect("Simulation de référence stable / Stable reference simulation");
        assert!(is_numerically_stable(&result),
            "État final de la simulation TFA doit être numériquement stable / \
             TFA simulation final state must be numerically stable");
    }
}
