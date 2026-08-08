# Changelog

All notable changes to `chrom-rs` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning: [SemVer](https://semver.org/)

---

## [Unreleased]

### Added
- `--source <model|scenario|solver> file=...` and `--output <csv|svg|png|json> file=...`, repeatable options with sub-parameters (built on `dynamic-cli` 0.6.0's `repeatable`/`option_parameters` — see [dcli#21](https://github.com/biface/dcli/issues/21) for that feature's own rationale), alongside the existing legacy scalar options (`--model`, `--output-csv`, etc.) — both syntaxes work, can be mixed across different roles/outputs in the same invocation, never both for the *same* role (ambiguity error). `svg`/`png` discriminants require a matching file extension, checked before any file is written.
- New `check` command (aliases: none; `run` gains the alias `solve`): validates whichever `--source <role> file=...` are given, using the exact same `load_model`/`load_scenario`/`load_solver` functions `run` itself uses, plus a plain inventory of `--project-dir`'s config-like files. No file classification or cross-combination testing — only what's explicitly asked for is validated.
- `benches/langmuir_performance.rs` — group 6, `bench_reference_cases`: Euler vs RK4 end-to-end `Solver::solve` time on the two real multi-species cases validated in #43/#44 (Ascorbic/Erythorbic, Glucose/Fructose), as opposed to groups 3/5 which explore synthetic n_species scaling ([#55](https://github.com/biface/chromatography/issues/55))
- `examples/validation_report.rs` — per-solver wall-clock solve time (`solve_time_ms.euler`, `solve_time_ms.rk4`, `solve_time_ms.rk4_over_euler_ratio`) added to the JSON report and console summary, alongside the existing `rsf_euler_vs_rk4` ([#55](https://github.com/biface/chromatography/issues/55))
- `examples/config/ascorbic_erythorbic/`, `examples/config/glucose_fructose/` — CLI-ready config directories (`model.yml`, `scenario.yml`, `solver_euler.yml`, `solver_rk4.yml`) exactly mirroring the physical and solver parameters of `MultiSpeciesCase::ascorbic_erythorbic()`/`glucose_fructose_linear()` in `validation/reference.rs` — usable directly with `chrom-rs run --output-plot` to produce a chromatogram matching the literature-validated cases ([#55](https://github.com/biface/chromatography/issues/55))
- `tools/plot_reference_cases_timing.rs` (`[[bin]] plot_reference_cases_timing`) — grouped bar chart of Euler vs RK4 mean solve time on the two real reference cases, reading Criterion `estimates.json` from `bench_reference_cases` ([#55](https://github.com/biface/chromatography/issues/55))
- `tools/plot_cost_accuracy.rs` (`[[bin]] plot_cost_accuracy`) — grouped bar chart of Euler vs RK4 solve time annotated with $R_{sf}$(Euler, RK4) per case, reading `validation_report.json` ([#55](https://github.com/biface/chromatography/issues/55))
- `benches/langmuir_performance.rs` — group 7, `bench_isotherm_evaluation_cost`: isolates a single `PhysicalModel::compute_physics` call (no PDE solve) for both real reference cases at matched n_points/n_species, to test whether the group 6 RK4/Euler ratio difference is actually explained by isotherm-evaluation cost ([#55](https://github.com/biface/chromatography/issues/55))
- `examples/stiffness_convergence.rs` — Δt convergence sweep (baseline `n_steps` ×1/×2/×5/×10/×25/×50) for both real reference cases, tracking $R_{sf}$(Euler, RK4) at each resolution with injection duration held fixed, to test the adsorption-front-stiffness hypothesis from the wiki write-up's "Perspective" section ([#55](https://github.com/biface/chromatography/issues/55))
- `tools/plot_stiffness_convergence.rs` (`[[bin]] plot_stiffness_convergence`) — log-log plot of $R_{sf}$(Euler, RK4) vs Δt per case, reading `stiffness_convergence.json` ([#55](https://github.com/biface/chromatography/issues/55))

### Changed
- `src/cli/commands.yml` — `model`/`scenario`/`solver` on `run` moved from schema-level `required: true` to `required: false`; the "all three roles required, exactly once each" rule is now enforced in `RunHandler::execute` (`resolve_source`) rather than by `dynamic-cli` itself, since a role can now be satisfied by either syntax.
- `tools/plot_parallelism_threshold.rs` — added a marker on the n_points=100 measured point: both real reference cases (Ascorbic/Erythorbic, Glucose/Fructose) operate there (ops=200), well inside the serial regime ([#55](https://github.com/biface/chromatography/issues/55))

### Verified
- Neither real reference case crosses the Rayon parallelism threshold: n_points × n_species = 100 × 2 = 200 for both Ascorbic/Erythorbic and Glucose/Fructose, against a threshold of 999 ([#55](https://github.com/biface/chromatography/issues/55))

---

## [0.5.0] — dynamic-cli 0.6.0 Adaptation

### Changed
- `Cargo.toml` — `dynamic-cli` `0.2.0` → `0.6.0` ([#52](https://github.com/biface/chromatography/issues/52), DD-001)
- `src/cli/mod.rs` — `register_handler` → `register_sync_handler` (renamed in dcli 0.5.0; the old name is `#[deprecated]`, which `cargo clippy -- -D warnings` turns into a build error)
- `src/cli/app.rs` — `RunHandler::execute` and the internal `resolve_input_path` helper migrated from `&HashMap<String, String>` to `&ParsedArgs` (dcli 0.6.0): `args.get("x")` → `args.get_scalar("x")` for all 4 fields read directly (`project-dir`, `output-csv`, `output-plot`, `export-json`) plus the 3 fields resolved via `resolve_input_path` (`model`, `scenario`, `solver`)
- No user-facing CLI surface change — `src/cli/commands.yml` unchanged, no `repeatable` options introduced (see DD-016, deferred to v0.6.0)

### Docs
- `benches/solver_performance.rs`, `benches/langmuir_performance.rs`, `tools/plot_parallelism_threshold.rs`, `tools/plot_species_response_curve.rs` — module-level rustdoc introductions rewritten to state the run command (`cargo bench --bench <name>` and, where applicable, the companion `cargo run --bin plot_*` invocation) up front ([#54](https://github.com/biface/chromatography/issues/54))
- `benches/langmuir_performance.rs`, `tools/plot_parallelism_threshold.rs`, `tools/plot_species_response_curve.rs` — rustdoc converted from bilingual FR/EN to English only, per the project's rustdoc convention (bilingual prose belongs in `//` code comments and the wiki, not `///`/`//!` rustdoc)

---

## [0.4.1] — 2026-08-06

### Added
- `validation/reference.rs` — two multi-species reference cases for `LangmuirMulti` ([#43](https://github.com/biface/chromatography/issues/43)):
  - `MultiSpeciesCase::ascorbic_erythorbic()` — competitive Langmuir (Nicoud 2015, Figure 5): $t_{R,A} = 448$ s, $t_{R,B} = 556$ s, $\Delta t_R = 108$ s
  - `MultiSpeciesCase::glucose_fructose_linear()` — linear regime only (Nicoud 2015, §10.1.2, Eq. (10.1), linear terms): Henry constants $K_{gl} = 0.27$, $K_{fr} = 0.46$ reproduced exactly via $\lambda = 0$, $N = 1$, $\tilde K = K_H / (1-\varepsilon)$
  - `validation/main.rs` — 3 new integration tests: per-species retention time (< 1 % error) for both cases, plus retention-time-gap check for the competitive case
- `examples/validation_report.rs` — on-demand structured JSON comparison report ([#44](https://github.com/biface/chromatography/issues/44)): simulated vs. reference peak position/amplitude, $R_{sf}$(Euler, RK4), and solver parameters (Δt, Δz, CFL) for every reference case; run via `cargo run --example validation_report`

### Fixed
- `validation/reference.rs`, `validation/dissimilarity.rs` — module-level rustdoc citations corrected: Lapidus & Amundson (1952) → Aris (1959) for the analytical PD model; Felinger & Guiochon → Nicoud (2015), §7.1 for the $R_{sf}$ criterion ([#43](https://github.com/biface/chromatography/issues/43))
- `cargo clippy --all-targets -- -D warnings` cleanup: 1 deny-by-default error and ~20 warnings, none previously visible to CI (`cargo clippy` omitted `--all-targets`/`--tests` — see `.github/workflows/ci.yml` below):
  - `src/physics/traits.rs` — ambiguous `3.14` scalar in `test_outlet_data_scalar` (`clippy::approx_constant`, deny-by-default)
  - `src/models/langmuir_multi.rs` — redundant `&` in a `format!` argument
  - `src/solver/boundary.rs` — `assert_eq!(x, true)` → `assert!(x)`
  - `src/solver/mod.rs` — redundant closure around `parallel_threshold`
  - `src/physics/data.rs` — `Vec` literal never mutated → array; manual indexed loop → `zip`
  - `examples/acids_multi.rs`, `examples/acids_from_config.rs` — 9 `for s in 0..n_species` loops idiomatized to iterators (2 similar-looking loops per file left untouched: they index two/three collections independently and are not needless-range-loop candidates)
  - `examples/diffusion.rs` — `.max(0.0).min(1.0)` → `.clamp(0.0, 1.0)`
  - `benches/langmuir_performance.rs` — unused `Rng` import (superseded by `RngExt::random_range` under `rand` 0.10), unused `use super::*;` in `mod tests`, doc-list indentation (×3)
  - `benches/solver_performance.rs` — unused loop variable `label` → `_label`
  - `tools/plot_species_response_curve.rs` — compile-time assertions wrapped in `const { assert!(..) }`
  - `src/solver/methods/euler.rs`, `rk4.rs` — `EulerSolver::default()` / `RK4Solver::default()` kept as-is with a justified `#[allow(clippy::default_constructed_unit_structs)]`: these tests specifically exercise the `Default` impl
  - `tests/solver_convergence.rs` — kept the `as f64` cast clippy flagged as `unnecessary_cast` (justified `#[allow]`): removing it triggers E0689 (ambiguous numeric type) on the following `.exp()` call, since inherent-method resolution runs before the type is otherwise pinned

### CI
- `.github/workflows/ci.yml` — clippy job extended to `cargo clippy --all-targets -- -D warnings`; previously omitted `--all-targets`/`--tests`, so `#[cfg(test)]` code, `examples/`, `benches/`, `tools/`, and standalone `[[test]]` crates (including `validation/`) were never checked

### Changed
- `Cargo.toml`: version bumped from `0.4.0` to `0.4.1`

---

## [0.4.0] — 2026-06-07

### Added
- `validation/` — standalone test crate for scientific validation (DD-012, [#17](https://github.com/biface/chromatography/issues/17), [#42](https://github.com/biface/chromatography/issues/42)):
  - `validation/dissimilarity.rs` — surface resolution $R_{sf}$ (Eq. 7.1, Felinger & Guiochon): `rsf`, `normalize`, `trapezoid` functions; linear interpolation on heterogeneous time grids; 12 unit tests
  - `validation/reference.rs` — physical reference cases with analytical predictions: `ReferenceCase::linear_tfa()` (Lapidus-Amundson, $t_R = 624$ s, $\sigma = 61.4$ s) and `ReferenceCase::nonlinear_tfa()` ($C_0 = 1.0$ mol/L); `COLUMN_LENGTH`, `POROSITY`, `VELOCITY`, `N_POINTS`, `N_STEPS`, `T_TOTAL`, `T_INJ` shared constants
  - `validation/main.rs` — 7 integration tests: retention time (< 1 % error), peak width (< 10 % error), mass conservation (≥ 90 %), Langmuir peak compression (mode-based), solver consistency $R_{sf}(\text{Euler}, \text{RK4}) = 0.016 < 0.05$, plus a diagnostic test
  - Declared in `Cargo.toml` as `[[test]] name = "scientific_validation" path = "validation/main.rs"`

### Fixed
- **`LangmuirSingle::derivative_isotherm`** — the effective adsorption capacity $\bar{N} = (1-\varepsilon) \cdot N$ was incorrectly computed as raw $N$ (missing stationary phase fraction $(1-\varepsilon)$). `LangmuirMulti` stored `stationary_fraction = 1 - \varepsilon` and was unaffected. Impact: retention time under-predicted by ~46 % for the default TFA parameters. ([#42](https://github.com/biface/chromatography/issues/42))
- **`RK4Solver` — intermediate stage time evaluation** — stages k₂, k₃, k₄ all used the same `ComputeContext` at $t = n \cdot \Delta t$, making time-dependent boundary conditions (injections) incorrect at intermediate times. Each stage now receives its own context: k₁ at $t$, k₂/k₃ at $t + \Delta t/2$, k₄ at $t + \Delta t$. ([#42](https://github.com/biface/chromatography/issues/42))
- **`TemporalInjection::Dirac`** — the Gaussian approximation (fixed $\sigma = 0.1$ s) placed half its area at $t < 0$ and was incompatible with RK4 intermediate stage evaluation. Replaced by an exact discrete equality: `evaluate(t)` returns `amount` if `t == time`, `0.0` otherwise. ([#42](https://github.com/biface/chromatography/issues/42))

### Changed
- `Cargo.toml`: version bumped from `0.3.0` to `0.4.0`


---

## [0.3.0] — 2026-05-31

### Added
- `domain/` module — validated construction facade for physical equipment (DD-011, [#16](https://github.com/biface/chromatography/issues/16), [#38](https://github.com/biface/chromatography/issues/38)):
  - `Column { column_length, n_points, porosity, diameter? }` — column geometry; derived accessors `dz()`, `phase_ratio()`
  - `MobilePhase { velocity, viscosity? }` — carrier fluid; derived accessor `interstitial_velocity(porosity)`
  - `Sample { injections: HashMap<Option<String>, TemporalInjection> }` — inlet injection profiles; compatible with `PhysicalModel::set_injections`
  - `Detector { position: DetectorPosition }` — signal measurement point; `DetectorPosition::Outlet | Relative(f64) | Absolute(f64)`; accessors `absolute_position()`, `node_index()`, `validate_against_column()`
  - All types derive `Serialize + Deserialize`; validated constructors return `Result<T, XxxError>`
  - All types re-exported via `lib.rs` prelude
- `physics/context.rs` — typed compute context (DD-008, [#13](https://github.com/biface/chromatography/issues/13), [#47](https://github.com/biface/chromatography/issues/47)):
  - `ComputeContext` — infallible `time()` / `time_step()` accessors; optional `HashMap<ContextVariable, ContextValue>` for derived quantities
  - `ContextVariable` — typed enum key (`Hash + Eq`): `Time`, `TimeStep`, `SpatialGradient { dimension, component }`, `External { name }`
  - `ContextValue` — typed value enum: `Scalar(f64)`, `Boolean(bool)`, `ScalarField(DVector<f64>)`, `VectorField(DMatrix<f64>)`
  - Structurally aligned with oxiflow `ComputeContext`; convergence deferred to post-v1.0.0 (DD-014)
- `LangmuirSingle::from_domain(column, mobile_phase, lambda, k, port, injection)` — ergonomic constructor from domain objects
- `LangmuirMulti::from_domain(column, mobile_phase, species) -> Result<Self, String>` — ergonomic constructor from domain objects

### Changed
- **Breaking** — `PhysicalModel::compute_physics` signature extended with compute context:
  ```rust
  // 0.2.0 (removed)
  fn compute_physics(&self, state: &PhysicalState) -> PhysicalState;
  // 0.3.0
  fn compute_physics(&self, state: &PhysicalState, ctx: &ComputeContext) -> PhysicalState;
  ```
  Euler and RK4 solvers now build `ComputeContext::new(t, dt)` at each step; models read `ctx.time()` directly — `state.set_metadata("time", t)` removed
- **Breaking** — `LangmuirSingle` JSON/YAML keys renamed for consistency with `LangmuirMulti` and `output/`:
  - `"length"` → `"column_length"`
  - `"nz"` → `"n_points"`
- `LangmuirSingle` internal field `length` → `column_length`, `nz` → `n_points`; accessor `length()` → `column_length()`

### Fixed
- Rustdoc broken intra-doc links in `domain/mod.rs` (`Column`, `MobilePhase`, `Sample`, `Detector`) and `domain/sample.rs` (`PhysicalModel::set_injections`)
- `cargo doc --no-deps` now generates 0 warnings

---

## [0.2.0] — 2026-04-22

### Added
- GitHub Actions workflows: CI (fmt, clippy, test, doc), coverage (cargo-llvm-cov + Codecov), mirror (GitLab), release-drafter (SemVer)
- GitHub issue templates: bug, feature, maintenance, decision
- Release-drafter SemVer configuration (`release-drafter-semver-template.yml`)
- Serialize/Deserialize on all core types: `PhysicalQuantity`, `PhysicalData`, `PhysicalState`,
  `TemporalInjection`, `Scenario`, `DomainBoundaries`, `SolverType`, `SolverConfiguration`, `SimulationResult`
- `#[typetag::serde]` on `PhysicalModel` trait and all implementors
- `ndarray/serde` feature activated — `PhysicalData::Array` fully serializable
- `typetag = "0.2"`, `serde_yaml = "0.9"` added to dependencies
- `Exportable` trait (`physics/traits.rs`): `to_map` / `from_map` mapping layer between physical models and JSON
- `ExportError`: `MissingKey`, `InvalidValue`, `SpeciesCountMismatch`
- `outlet_data(quantity, trajectory, idx)`: generic outlet extractor for any `PhysicalQuantity`
- `sample_indices(total, n)`: uniform downsampling helper, first and last points always included
- `Exportable` implemented on `LangmuirSingle` and `LangmuirMulti` — named species blocks, `global` extension point
- `output/export/json.rs`: `to_json` / `from_json` — pure I/O layer, `Map<String, Value>` only, no model knowledge
- `serde_json = "1.0"` added to dependencies
- `step: Option<usize>` field on `SolverConfiguration` — trajectory subsampling for JSON export; builder `with_step(n) -> Self`
- `set_injection(&mut self, TemporalInjection)` on `LangmuirSingle`
- `set_injection_all` and `set_injection_for` on `LangmuirMulti`
- `PhysicalModel::set_injections` — single generic injection entry-point on the trait
- `config/` module: `ConfigError`, `Format`, `load_from_file<T>` generic helper
- `config/model.rs`: `load_model(path) -> Result<Box<dyn PhysicalModel>, ConfigError>`
- `config/scenario.rs`: `load_scenario(path, &mut dyn PhysicalModel) -> Result<DomainBoundaries, ConfigError>`
- `config/solver.rs`: `load_solver(path) -> Result<SolverConfig, ConfigError>`
- `cli/` module: `ChromContext` (validated `--project-dir`), `RunHandler` (full simulation pipeline), `build_app()`
- Command surface: `chrom-rs run --model --scenario --solver [--project-dir] [--output-csv] [--output-plot] [--export-json]`
- `examples/tfa_from_config.rs` — reproduces `tfa.rs` via config files; results numerically identical
- `examples/acids_from_config.rs` — reproduces `acids_multi.rs`; solo phase derives `LangmuirSingle` from `LangmuirMulti` parameters
- `examples/config/tfa/` — `model.yml`, `scenario_dirac.yml`, `scenario_gaussian.yml`, `solver_rk4.yml`, `solver_euler.yml`
- `examples/config/acids/` — `model.yml`, `scenario_gaussian.yml`, `solver_rk4.yml`, `solver_euler.yml`
- `tests/cli_integration.rs` — 8 end-to-end tests for `RunHandler::execute` (single-species, multi-species, CSV/plot/JSON, error paths)

### Fixed
- `cli/app.rs`: use `model.points()` instead of `result.time_points.len()` in `plot_chromatogram` / `plot_chromatogram_multi` — prevents matrix index out of bounds

### Changed
- Upgrade `dynamic-cli` from `0.1.1` to `0.2.0`
- Upgrade `nalgebra` from `0.33` to `0.34.2` with `serde-serialize` feature
- `PhysicalQuantity::Custom(&'static str)` → `Custom(String)`, `Copy` removed
- Fix rustdoc redirect in CI: `dynamic_cli/index.html` → `chrom_rs/index.html`
- Enable and fix all doc-tests across `models/`, `solver/`, and `output/` modules
- Add `libfontconfig1-dev` system dependency in CI jobs (required by `plotters`)
- `LangmuirMulti`: add public accessors `porosity`, `velocity`, `column_length`, `spatial_points`, `species_params`
- `LangmuirSingle` and `LangmuirMulti`: `Exportable` `to_map` / `from_map` round-trip tests added

---

## [0.1.0] — 2025-12-30

### Added
- Core physics: `LangmuirSingleModel` (scalar derivative, 10–100× faster on 1 species), `LangmuirMultiModel` (full competitive Jacobian, LU inversion, O(n³) in n_species)
- Numerical solvers: Forward Euler (order 1), Runge-Kutta 4 (order 4)
- CFL stability condition with IEEE 754 epsilon guard
- Runtime-configurable parallelism via Rayon (threshold: 999 ops, `AtomicUsize` + `ThresholdGuard` RAII)
- Benchmarks (Criterion): CFL stability, single vs multi species, multi-species scaling, parallelism threshold
- Visualization (`plotters`): chromatogram, steady-state profile, profile evolution — mono and multi-species
- CSV export (`output/` module, partial)
- KaTeX integration in rustdoc (`docs/katex-header.html`)
- Bilingual documentation convention (EN code + FR/EN examples on demand)
- Design Decisions DD-001 through DD-007 (see GitHub issues [#6](https://github.com/biface/chromatography/issues/6)–[#12](https://github.com/biface/chromatography/issues/12))
