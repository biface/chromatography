# Changelog

All notable changes to `chrom-rs` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning: [SemVer](https://semver.org/)

---

## [Unreleased]

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
