# Changelog

All notable changes to `chrom-rs` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
Versioning: [SemVer](https://semver.org/)

---

## [Unreleased] — v0.2.0 · Application Layer

### Added
- GitHub Actions workflows: CI (fmt, clippy, test, doc), coverage (cargo-llvm-cov + Codecov), mirror (GitLab), release-drafter (SemVer)
- GitHub issue templates: bug, feature, maintenance, decision
- Release-drafter SemVer configuration (`release-drafter-semver-template.yml`)
- Serialize/Deserialize on all core types: `PhysicalQuantity`, `PhysicalData`, `PhysicalState`,
  `TemporalInjection`, `Scenario`, `DomainBoundaries`, `SolverType`, `SolverConfiguration`, `SimulationResult`
- `#[typetag::serde]` on `PhysicalModel` trait and all implementors
- `ndarray/serde` feature activated — `PhysicalData::Array` fully serializable
- `typetag = "0.2"`, `serde_yaml = "0.9"` added to dependencies
- `Exportable` trait (`physics/traits.rs`): `to_map` / `from_map` mapping layer between physical models and JSON — no circular dependency (option A signature)
- `ExportError`: `MissingKey`, `InvalidValue`, `SpeciesCountMismatch`
- `outlet_data(quantity, trajectory, idx)`: generic outlet extractor for any `PhysicalQuantity`
- `sample_indices(total, n)`: uniform downsampling helper, first and last points always included
- `Exportable` implemented on `LangmuirSingle` and `LangmuirMulti` — named species blocks (`species_N` + `"name"` key), `global` extension point for scalar/vector quantities
- `output/export/json.rs`: `to_json` / `from_json` — pure I/O layer, `Map<String, Value>` only, no model knowledge
- `serde_json = "1.0"` added to dependencies
- `step: Option<usize>` field on `SolverConfiguration` with `#[serde(default)]` — trajectory subsampling for JSON export (DD-010 / DD-015); builder `with_step(n) -> Self`
- `set_injection(&mut self, TemporalInjection)` on `LangmuirSingle` — replaces injection profile post-deserialisation
- `set_injection_all(&mut self, TemporalInjection)` and `set_injection_for(&mut self, &str, TemporalInjection) -> Result<(), String>` on `LangmuirMulti`
- `PhysicalModel::set_injections(&mut self, &HashMap<Option<String>, TemporalInjection>) -> Result<(), String>` — single generic injection entry-point on the trait: `None` key = default for all unlisted species, `Some(name)` key = per-species override; default no-op for models without injection
- `config/` module (DD-015): `ConfigError` (Io / UnsupportedFormat / Parse / Validation), `Format` enum, `load_from_file<T>` generic helper (format check precedes file I/O)
- `config/model.rs`: `load_model(path) -> Result<Box<dyn PhysicalModel>, ConfigError>` via typetag — injection left as `None`, applied by scenario loader
- `config/scenario.rs`: `load_scenario(path, &mut dyn PhysicalModel) -> Result<DomainBoundaries, ConfigError>` — builds `HashMap<Option<String>, TemporalInjection>` from `default_injection` and `injections[]` YAML fields, calls `set_injections` once; `initial_condition: zero` supported in v0.2.0
- `config/solver.rs`: `load_solver(path) -> Result<SolverConfig, ConfigError>` — `SolverConfig { config: SolverConfiguration, solver_name: String }`; validates `type` (RK4 / Euler), `total_time > 0`, `time_steps > 0`

### Changed
- Upgrade `dynamic-cli` dependency from `0.1.1` to `0.2.0`
- Upgrade `nalgebra` dependency from `0.33` to `0.34.2` with `serde-serialize` feature
- `PhysicalQuantity::Custom(&'static str)` → `Custom(String)`, `Copy` removed
- Fix rustdoc redirect in CI: `dynamic_cli/index.html` → `chrom_rs/index.html`
- Enable and fix all doc-tests across `models/`, `solver/`, and `output/` modules — remove `ignore` attribute, align examples with current public API
- Add `libfontconfig1-dev` system dependency in CI jobs (required by `plotters`)
- `LangmuirMulti`: add public accessors `porosity`, `velocity`, `column_length`, `spatial_points`, `species_params` — with full rustdoc (physical symbol, unit, relation to precomputed quantities)
- `LangmuirMulti` `PhysicalModel` impl methods (`points`, `name`, `description`) documented
- `LangmuirSingle` and `LangmuirMulti`: Exportable `to_map` / `from_map` round-trip tests added (closes #34)

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
