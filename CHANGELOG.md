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

### Changed
- Upgrade `dynamic-cli` dependency from `0.1.1` to `0.2.0`
- Upgrade `nalgebra` dependency from `0.33` to `0.34.2` with `serde-serialize` feature
- `PhysicalQuantity::Custom(&'static str)` → `Custom(String)`, `Copy` removed
- Fix rustdoc redirect in CI: `dynamic_cli/index.html` → `chrom_rs/index.html`
- Enable and fix all doc-tests across `models/`, `solver/`, and `output/` modules — remove `ignore` attribute, align examples with current public API
- Add `libfontconfig1-dev` system dependency in CI jobs (required by `plotters`)
- `LangmuirMulti`: add public accessors `porosity`, `velocity`, `column_length`, `spatial_points`, `species_params`

### Removed
- Untrack `langmuir_single_simple.rs` (out of scope, kept locally)

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
