# chrom-rs

[![Crates.io](https://img.shields.io/crates/v/chrom-rs.svg)](https://crates.io/crates/chrom-rs)
[![Docs.rs](https://docs.rs/chrom-rs/badge.svg)](https://docs.rs/chrom-rs)
[![CI](https://github.com/biface/chromatography/actions/workflows/ci.yml/badge.svg)](https://github.com/biface/chromatography/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)

A Rust framework for simulating liquid-phase chromatography. `chrom-rs` models the transport and adsorption of chemical
species through a chromatographic column presently using Langmuir isotherms and numerical solvers.

A French version of this README is available: [README.fr.md](README.fr.md).

---

## Features

- **Single-species** simulation via `LangmuirSingle` — scalar derivative, optimised for pure-compound studies
- **Multi-species competitive adsorption** via `LangmuirMulti` — full Jacobian, LU inversion, O(n³) in number of species
- **Numerical solvers**: Forward Euler (order 1) and Runge-Kutta 4 (order 4)
- **Injection profiles**: Dirac, Gaussian, Rectangle, or custom closure
- **Config-file interface**: three independent YAML/JSON files (`model.yml`, `scenario.yml`, `solver.yml`)
- **Outputs**: CSV export, JSON export, chromatogram plots via `plotters`
- **CLI**: `chrom-rs run` powered by `dynamic-cli`

---

## Installation

```toml
[dependencies]
chrom-rs = "0.2"
```

---

## Quick Start

```rust
use chrom_rs::{
    models::{LangmuirSingle, TemporalInjection},
    solver::{DomainBoundaries, RK4Solver, Scenario, Solver, SolverConfiguration},
};

// Physical model — TFA on a C18 column
let injection = TemporalInjection::gaussian(10.0, 3.0, 0.1);
let model = Box::new(LangmuirSingle::new(
    1.2,    // λ  — linear retention term
    0.4,    // K̃  — Langmuir equilibrium constant [L/mol]
    2.0,    // N  — adsorption capacity
    0.4,    // ε  — column porosity
    0.001,  // u  — superficial velocity [m/s]
    0.25,   // L  — column length [m]
    100,    // nz — spatial points
    injection,
));

// Scenario and solver
let initial = model.setup_initial_state();
let boundaries = DomainBoundaries::temporal(initial);
let scenario = Scenario::new(model, boundaries);
let config = SolverConfiguration::time_evolution(600.0, 10_000);

let result = RK4Solver::new().solve(&scenario, &config).unwrap();
println!("Simulated {} time points", result.time_points.len());
```

---

## Config-file interface

Simulations can be driven by three YAML files without writing Rust code:

```bash
chrom-rs run \
  --model    examples/config/tfa/model.yml \
  --scenario examples/config/tfa/scenario_gaussian.yml \
  --solver   examples/config/tfa/solver_rk4.yml \
  --output-csv  result.csv \
  --output-plot result.png \
  --export-json result.json
```

See `examples/config/` for ready-to-use fixtures and `examples/tfa_from_config.rs` /
`examples/acids_from_config.rs` for the corresponding Rust entry points.

---

## Examples

| Example             | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| `tfa_single`        | TFA chromatography — Dirac and Gaussian injections, Euler and RK4 |
| `tfa_from_config`   | Same simulation driven by config files                            |
| `acids_multi`       | Ascorbic / Erythorbic acids — solo and competitive phases         |
| `acids_from_config` | Same simulation driven by config files                            |

```bash
cargo run --example tfa_single
cargo run --example tfa_from_config
cargo run --example acids_multi
cargo run --example acids_from_config
```

---

## Documentation

Full API documentation: [docs.rs/chrom-rs](https://docs.rs/chrom-rs)

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE).
