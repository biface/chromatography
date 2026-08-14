# chrom-rs

[![Crates.io](https://img.shields.io/crates/v/chrom-rs.svg)](https://crates.io/crates/chrom-rs)
[![Docs.rs](https://docs.rs/chrom-rs/badge.svg)](https://docs.rs/chrom-rs)
[![CI](https://github.com/biface/chromatography/actions/workflows/ci.yml/badge.svg)](https://github.com/biface/chromatography/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#licence)

Un framework Rust pour simuler la chromatographie en phase liquide. `chrom-rs` modélise le transport et l'adsorption 
d'espèces chimiques dans une colonne chromatographique actuellement à l'aide d'isothermes de Langmuir et de solveurs
numériques.

La version anglaise de ce README est disponible : [README.md](README.md).

---

## Fonctionnalités

- **Simulation mono-espèce** via `LangmuirSingle` — dérivée scalaire, optimisée pour l'étude de composés purs
- **Adsorption compétitive multi-espèces** via `LangmuirMulti` — jacobien complet, inversion LU, O(n³) en nombre d'espèces
- **Solveurs numériques** : Euler explicite (ordre 1) et Runge-Kutta 4 (ordre 4)
- **Profils d'injection** : Dirac, Gaussien, Rectangle, ou closure personnalisée
- **Interface par fichiers de configuration** : trois fichiers YAML/JSON indépendants (`model.yml`, `scenario.yml`, `solver.yml`)
- **Sorties** : export CSV, export JSON, chromatogrammes via `plotters`
- **CLI** : `chrom-rs run` (alias `simulate`, `solve`) et `chrom-rs check`, propulsées par `dynamic-cli`

---

## Installation

```toml
[dependencies]
chrom-rs = "0.5"
```

---

## Démarrage rapide

```rust
use chrom_rs::{
    models::{LangmuirSingle, TemporalInjection},
    solver::{DomainBoundaries, RK4Solver, Scenario, Solver, SolverConfiguration},
};

// Modèle physique — TFA sur colonne C18
let injection = TemporalInjection::gaussian(10.0, 3.0, 0.1);
let model = Box::new(LangmuirSingle::new(
    1.2,    // λ  — terme de rétention linéaire
    0.4,    // K̃  — constante d'équilibre de Langmuir [L/mol]
    2.0,    // N  — capacité d'adsorption
    0.4,    // ε  — porosité de la colonne
    0.001,  // u  — vitesse superficielle [m/s]
    0.25,   // L  — longueur de colonne [m]
    100,    // nz — points spatiaux
    injection,
));

// Scénario et solveur
let initial = model.setup_initial_state();
let boundaries = DomainBoundaries::temporal(initial);
let scenario = Scenario::new(model, boundaries);
let config = SolverConfiguration::time_evolution(600.0, 10_000);

let result = RK4Solver::new().solve(&scenario, &config).unwrap();
println!("{} points temporels simulés", result.time_points.len());
```

---

## Interface par fichiers de configuration

Les simulations peuvent être pilotées par trois fichiers YAML sans écrire de code Rust :

```bash
chrom-rs run \
  --model    examples/config/tfa/model.yml \
  --scenario examples/config/tfa/scenario_gaussian.yml \
  --solver   examples/config/tfa/solver_rk4.yml \
  --output-csv  result.csv \
  --output-plot result.png \
  --export-json result.json
```

Les trois fichiers d'entrée et chaque artefact de sortie peuvent aussi
être fournis via une syntaxe répétable `--source`/`--output` — équivalente,
et combinable avec les options ci-dessus (jamais les deux pour le même
rôle dans la même invocation) :

```bash
chrom-rs solve \
  --source model    file=examples/config/tfa/model.yml \
  --source scenario file=examples/config/tfa/scenario_gaussian.yml \
  --source solver   file=examples/config/tfa/solver_rk4.yml \
  --output csv  file=result.csv \
  --output png  file=result.png \
  --output json file=result.json
```

Avant de lancer un calcul, `chrom-rs check` valide un jeu de fichiers de
configuration — ou, sans aucun `--source`, liste simplement les fichiers
de type config présents dans `--project-dir` :

```bash
chrom-rs check \
  --source model    file=examples/config/tfa/model.yml \
  --source scenario file=examples/config/tfa/scenario_gaussian.yml
```

Voir `examples/config/` pour des fichiers prêts à l'emploi et `examples/tfa_from_config.rs` /
`examples/acids_from_config.rs` pour les points d'entrée Rust correspondants.

---

## Exemples

| Exemple             | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `tfa_single`        | Chromatographie TFA — injections Dirac et Gaussienne, Euler et RK4 |
| `tfa_from_config`   | Même simulation pilotée par fichiers de configuration              |
| `acids_multi`       | Acides ascorbique / érythorbique — phases solo et compétitive      |
| `acids_from_config` | Même simulation pilotée par fichiers de configuration              |

```bash
cargo run --example tfa_single
cargo run --example tfa_from_config
cargo run --example acids_multi
cargo run --example acids_from_config
```

---

## Documentation

Documentation complète de l'API : [docs.rs/chrom-rs](https://docs.rs/chrom-rs)

---

## Licence

Distribué sous la licence [Apache License, Version 2.0](LICENSE-APACHE).
