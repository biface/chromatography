# Changelog

All notable changes to `chrom-rs` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning: [SemVer](https://semver.org/)

---

## [Unreleased]

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
