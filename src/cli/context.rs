//! Execution context shared across all `chrom-rs` command handlers.
//!
//! - [`ChromContext`]: the `project_dir` invariant plus the three
//!   pending-configuration builder slots used by `config`/`build` (DD-016).
//! - [`ContextError`]: errors from [`ChromContext::set_project_dir`].
//!
//! # Path invariants
//!
//! `project_dir` is always a *safe* directory:
//! - No `..` component (prevents escaping the declared root).
//! - The directory exists and is readable by the current process.
//! - The directory is writable by the current process (output files land here).
//!
//! These invariants are enforced exclusively by
//! [`ChromContext::set_project_dir`].

use std::io;
use std::path::{Component, Path, PathBuf};

use dynamic_cli::ExecutionContext;

use crate::cli::builders::{
    InjectionBuilder, ModelBuilder, MultiModelBuilder, ScenarioBuilder, ShapeSwitch,
    SingleModelBuilder, SolverBuilder, SpeciesBuilder,
};

// ============================================================================
// ContextError
// ============================================================================

/// Errors that can occur when configuring a [`ChromContext`].
#[derive(Debug)]
pub enum ContextError {
    /// The path contains a `..` component.
    PathTraversal(PathBuf),

    /// The path does not point to an existing directory.
    NotADirectory(PathBuf),

    /// The current process lacks read or write permission on the directory.
    PermissionDenied(PathBuf, io::Error),
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextError::PathTraversal(p) => write!(
                f,
                "project-dir '{}' contains '..': path traversal is not allowed",
                p.display()
            ),
            ContextError::NotADirectory(p) => write!(
                f,
                "project-dir '{}' does not exist or is not a directory",
                p.display()
            ),
            ContextError::PermissionDenied(p, e) => write!(
                f,
                "project-dir '{}': insufficient permissions — {e}",
                p.display()
            ),
        }
    }
}

impl std::error::Error for ContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ContextError::PermissionDenied(_, e) => Some(e),
            _ => None,
        }
    }
}

// ============================================================================
// ChromContext
// ============================================================================

/// Runtime state shared across all `chrom-rs` command handlers.
///
/// `project_dir` is the directory relative to which `--model`,
/// `--scenario`, `--solver`, and all output file names are resolved.
///
/// Since v0.6.0, `ChromContext` also holds three optional builder slots —
/// `pending_model`, `pending_solver`, `pending_scenario` — one per
/// `config`/`build` target (DD-016, issue #53). Each slot is filled in
/// field by field across one or more `config` occurrences, whether given in
/// a single invocation or spread across a chained sequence (`dynamic-cli`
/// 0.9.0 command chaining). Nothing in a slot is validated until it is
/// turned into a real model/solver/scenario by `save` or `run`.
///
/// # Example
///
/// ```rust,no_run
/// use chrom_rs::cli::context::ChromContext;
///
/// let mut ctx = ChromContext::new();           // project_dir = "."
/// ctx.set_project_dir("experiments/run_01").unwrap();
/// assert_eq!(ctx.project_dir(), std::path::Path::new("experiments/run_01"));
/// ```
pub struct ChromContext {
    /// Root directory for all file-name resolution.
    ///
    /// Invariants enforced by [`set_project_dir`](Self::set_project_dir):
    /// no `..` component; existing, readable, writable directory.
    project_dir: PathBuf,

    /// Pending model under construction via `config --model ...`.
    /// `None` until the first `single`/`multi`/`species` occurrence.
    pending_model: Option<ModelBuilder>,

    /// Pending solver under construction via `config --solver ...`.
    pending_solver: Option<SolverBuilder>,

    /// Pending scenario under construction via `config --scenario ...`.
    pending_scenario: Option<ScenarioBuilder>,
}

impl ChromContext {
    /// Creates a context whose project directory is the current working
    /// directory (`.`).
    pub fn new() -> Self {
        Self {
            project_dir: PathBuf::from("."),
            pending_model: None,
            pending_solver: None,
            pending_scenario: None,
        }
    }

    /// Returns the current project directory.
    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// Returns the pending model slot, if any.
    pub fn pending_model(&self) -> Option<&ModelBuilder> {
        self.pending_model.as_ref()
    }

    /// Returns the pending solver slot, if any.
    pub fn pending_solver(&self) -> Option<&SolverBuilder> {
        self.pending_solver.as_ref()
    }

    /// Returns the pending scenario slot, if any.
    pub fn pending_scenario(&self) -> Option<&ScenarioBuilder> {
        self.pending_scenario.as_ref()
    }

    /// Merges `fields` into the pending model as a single-species build.
    ///
    /// - If the slot is empty, locks it to `Single(fields)`.
    /// - If the slot already holds `Single`, merges field by field
    ///   (last-write-wins per field, see [`SingleModelBuilder::merge`]).
    /// - If the slot holds `Multi`, resets it to `Single(fields)` and
    ///   returns [`ShapeSwitch::ToSingle`] so the caller can print a
    ///   visible warning — the previously accumulated multi-species state
    ///   (including any species list) is discarded, not merged.
    pub fn merge_model_single(&mut self, fields: SingleModelBuilder) -> Option<ShapeSwitch> {
        match &mut self.pending_model {
            None => {
                self.pending_model = Some(ModelBuilder::Single(fields));
                None
            }
            Some(ModelBuilder::Single(existing)) => {
                existing.merge(fields);
                None
            }
            Some(ModelBuilder::Multi(_)) => {
                self.pending_model = Some(ModelBuilder::Single(fields));
                Some(ShapeSwitch::ToSingle)
            }
        }
    }

    /// Merges the scalar fields of `fields` into the pending model as a
    /// multi-species build. Mirrors [`Self::merge_model_single`]'s
    /// switch/merge/lock logic, in the opposite direction. `fields.species`
    /// is ignored here — add species via [`Self::add_species`] instead.
    pub fn merge_model_multi(&mut self, fields: MultiModelBuilder) -> Option<ShapeSwitch> {
        match &mut self.pending_model {
            None => {
                self.pending_model = Some(ModelBuilder::Multi(fields));
                None
            }
            Some(ModelBuilder::Multi(existing)) => {
                existing.merge_scalars(fields);
                None
            }
            Some(ModelBuilder::Single(_)) => {
                self.pending_model = Some(ModelBuilder::Multi(fields));
                Some(ShapeSwitch::ToMulti)
            }
        }
    }

    /// Appends one species to the pending multi-species model, in call
    /// order. A `species` occurrence alone is enough to lock the slot into
    /// `Multi` — a scalar `multi` occurrence is not required first.
    ///
    /// If the slot currently holds `Single`, it is reset to an empty
    /// `Multi` (species list starting with just this one) and
    /// [`ShapeSwitch::ToMulti`] is returned as a visible-warning signal.
    pub fn add_species(&mut self, species: SpeciesBuilder) -> Option<ShapeSwitch> {
        match &mut self.pending_model {
            None => {
                let mut multi = MultiModelBuilder::default();
                multi.push_species(species);
                self.pending_model = Some(ModelBuilder::Multi(multi));
                None
            }
            Some(ModelBuilder::Multi(existing)) => {
                existing.push_species(species);
                None
            }
            Some(ModelBuilder::Single(_)) => {
                let mut multi = MultiModelBuilder::default();
                multi.push_species(species);
                self.pending_model = Some(ModelBuilder::Multi(multi));
                Some(ShapeSwitch::ToMulti)
            }
        }
    }

    /// Merges `fields` into the pending solver, field by field
    /// (last-write-wins per field). Locks the slot on first call.
    pub fn merge_solver(&mut self, fields: SolverBuilder) {
        match &mut self.pending_solver {
            Some(existing) => existing.merge(fields),
            None => self.pending_solver = Some(fields),
        }
    }

    /// Sets or merges the pending scenario's initial condition.
    pub fn set_scenario_initial_condition(&mut self, value: impl Into<String>) {
        self.pending_scenario
            .get_or_insert_with(ScenarioBuilder::default)
            .set_initial_condition(value.into());
    }

    /// Merges `injection` into the pending scenario's default injection.
    pub fn merge_scenario_default_injection(&mut self, injection: InjectionBuilder) {
        self.pending_scenario
            .get_or_insert_with(ScenarioBuilder::default)
            .merge_default_injection(injection);
    }

    /// Merges `injection` into the pending scenario's override for
    /// `species` (creating the override on first mention of that species).
    pub fn merge_scenario_species_override(&mut self, species: &str, injection: InjectionBuilder) {
        self.pending_scenario
            .get_or_insert_with(ScenarioBuilder::default)
            .merge_species_override(species, injection);
    }

    /// Sets the project directory after validating the path.
    ///
    /// # Validation
    ///
    /// 1. Rejects any path containing a `..` component.
    /// 2. Checks that the path points to an existing directory.
    /// 3. Verifies read permission by listing the directory.
    /// 4. Verifies write permission by creating and removing a probe file.
    ///
    /// # Errors
    ///
    /// - [`ContextError::PathTraversal`] if `path` contains `..`.
    /// - [`ContextError::NotADirectory`] if `path` does not exist or is a file.
    /// - [`ContextError::PermissionDenied`] if the process cannot read or
    ///   write the directory.
    pub fn set_project_dir(&mut self, path: impl Into<PathBuf>) -> Result<(), ContextError> {
        let path = path.into();

        // 1 — Reject `..` regardless of position.
        if path.components().any(|c| c == Component::ParentDir) {
            return Err(ContextError::PathTraversal(path));
        }

        // 2 — Must be an existing directory.
        if !path.is_dir() {
            return Err(ContextError::NotADirectory(path));
        }

        // 3 — Read permission.
        std::fs::read_dir(&path).map_err(|e| ContextError::PermissionDenied(path.clone(), e))?;

        // 4 — Write permission: probe with a temporary file.
        let probe = path.join(".chrom_rs_write_probe");
        std::fs::File::create(&probe)
            .map_err(|e| ContextError::PermissionDenied(path.clone(), e))?;
        let _ = std::fs::remove_file(&probe);

        self.project_dir = path;
        Ok(())
    }
}

impl Default for ChromContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext for ChromContext {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use dynamic_cli::downcast_ref;

    // ── ChromContext pending-model/solver/scenario slots (issue #67) ────────

    #[test]
    fn test_pending_model_starts_empty() {
        let ctx = ChromContext::new();
        assert!(ctx.pending_model().is_none());
        assert!(ctx.pending_solver().is_none());
        assert!(ctx.pending_scenario().is_none());
    }

    #[test]
    fn test_merge_model_single_locks_then_merges() {
        let mut ctx = ChromContext::new();

        let warning = ctx.merge_model_single(SingleModelBuilder {
            lambda: Some(1.2),
            ..Default::default()
        });
        assert!(warning.is_none());

        let warning = ctx.merge_model_single(SingleModelBuilder {
            langmuir_k: Some(0.4),
            ..Default::default()
        });
        assert!(warning.is_none());

        match ctx.pending_model() {
            Some(ModelBuilder::Single(single)) => {
                assert_eq!(single.lambda, Some(1.2));
                assert_eq!(single.langmuir_k, Some(0.4));
            }
            other => panic!("expected Single, got {other:?}"),
        }
    }

    #[test]
    fn test_add_species_alone_locks_multi_shape() {
        let mut ctx = ChromContext::new();

        let warning = ctx.add_species(SpeciesBuilder {
            name: Some("Ascorbic".to_string()),
            ..Default::default()
        });
        assert!(warning.is_none(), "no multi occurrence needed first");

        match ctx.pending_model() {
            Some(ModelBuilder::Multi(multi)) => {
                assert_eq!(multi.species.len(), 1);
                assert_eq!(multi.species[0].name.as_deref(), Some("Ascorbic"));
            }
            other => panic!("expected Multi, got {other:?}"),
        }
    }

    #[test]
    fn test_switching_single_to_multi_resets_and_warns() {
        let mut ctx = ChromContext::new();
        ctx.merge_model_single(SingleModelBuilder {
            lambda: Some(1.2),
            ..Default::default()
        });

        let warning = ctx.merge_model_multi(MultiModelBuilder {
            n_points: Some(100),
            ..Default::default()
        });
        assert_eq!(warning, Some(ShapeSwitch::ToMulti));

        match ctx.pending_model() {
            Some(ModelBuilder::Multi(multi)) => {
                assert_eq!(multi.n_points, Some(100));
            }
            other => panic!("expected Multi after switch, got {other:?}"),
        }
    }

    #[test]
    fn test_switching_multi_to_single_resets_and_warns() {
        let mut ctx = ChromContext::new();
        ctx.add_species(SpeciesBuilder {
            name: Some("A".to_string()),
            ..Default::default()
        });

        let warning = ctx.merge_model_single(SingleModelBuilder {
            lambda: Some(1.2),
            ..Default::default()
        });
        assert_eq!(warning, Some(ShapeSwitch::ToSingle));

        match ctx.pending_model() {
            Some(ModelBuilder::Single(single)) => {
                assert_eq!(single.lambda, Some(1.2));
            }
            other => panic!("expected Single after switch, got {other:?}"),
        }
    }

    #[test]
    fn test_merge_solver_accumulates_across_calls() {
        let mut ctx = ChromContext::new();
        ctx.merge_solver(SolverBuilder {
            solver_type: Some("RK4".to_string()),
            ..Default::default()
        });
        ctx.merge_solver(SolverBuilder {
            total_time: Some(600.0),
            time_steps: Some(10_000),
            ..Default::default()
        });

        let solver = ctx.pending_solver().expect("must be set");
        assert_eq!(solver.solver_type.as_deref(), Some("RK4"));
        assert_eq!(solver.total_time, Some(600.0));
        assert_eq!(solver.time_steps, Some(10_000));
    }

    #[test]
    fn test_scenario_calls_before_model_do_not_touch_pending_model() {
        let mut ctx = ChromContext::new();
        ctx.set_scenario_initial_condition("zero");
        ctx.merge_scenario_default_injection(InjectionBuilder {
            injection_type: Some("Gaussian".to_string()),
            center: Some(10.0),
            ..Default::default()
        });
        ctx.merge_scenario_species_override(
            "Erythorbic",
            InjectionBuilder {
                injection_type: Some("Dirac".to_string()),
                ..Default::default()
            },
        );

        // No pending model was ever set — scenario builds independently,
        // ordering/existence checks belong to the future `config scenario`
        // handler (#71), not to ChromContext itself.
        assert!(ctx.pending_model().is_none());

        let scenario = ctx.pending_scenario().expect("must be set");
        assert_eq!(scenario.initial_condition.as_deref(), Some("zero"));
        assert_eq!(
            scenario.default_injection.as_ref().unwrap().center,
            Some(10.0)
        );
        assert_eq!(scenario.species_overrides.len(), 1);
    }

    // ── ChromContext ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_defaults_to_current_dir() {
        let ctx = ChromContext::new();
        assert_eq!(ctx.project_dir(), Path::new("."));
    }

    #[test]
    fn test_default_equals_new() {
        let ctx = ChromContext::default();
        assert_eq!(ctx.project_dir(), Path::new("."));
    }

    #[test]
    fn test_set_project_dir_valid() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = ChromContext::new();
        ctx.set_project_dir(dir.path()).unwrap();
        assert_eq!(ctx.project_dir(), dir.path());
    }

    #[test]
    fn test_set_project_dir_rejects_parent_component() {
        let mut ctx = ChromContext::new();
        assert!(matches!(
            ctx.set_project_dir("some/../other"),
            Err(ContextError::PathTraversal(_))
        ));
    }

    #[test]
    fn test_set_project_dir_rejects_leading_parent() {
        let mut ctx = ChromContext::new();
        assert!(matches!(
            ctx.set_project_dir("../sibling"),
            Err(ContextError::PathTraversal(_))
        ));
    }

    #[test]
    fn test_set_project_dir_rejects_missing_path() {
        let mut ctx = ChromContext::new();
        assert!(matches!(
            ctx.set_project_dir("/tmp/chrom_rs_does_not_exist_xyz"),
            Err(ContextError::NotADirectory(_))
        ));
    }

    #[test]
    fn test_set_project_dir_rejects_file_path() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut ctx = ChromContext::new();
        assert!(matches!(
            ctx.set_project_dir(file.path()),
            Err(ContextError::NotADirectory(_))
        ));
    }

    // ── ExecutionContext downcast ─────────────────────────────────────────────

    #[test]
    fn test_as_any_downcast_ref() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = ChromContext::new();
        ctx.set_project_dir(dir.path()).unwrap();
        let boxed: Box<dyn ExecutionContext> = Box::new(ctx);
        let recovered = downcast_ref::<ChromContext>(boxed.as_ref()).unwrap();
        assert_eq!(recovered.project_dir(), dir.path());
    }

    #[test]
    fn test_as_any_mut_downcast_mut() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        let mut ctx = ChromContext::new();
        ctx.set_project_dir(dir1.path()).unwrap();
        let any_mut = ctx.as_any_mut();
        let recovered = any_mut.downcast_mut::<ChromContext>().unwrap();
        recovered.set_project_dir(dir2.path()).unwrap();
        assert_eq!(ctx.project_dir(), dir2.path());
    }

    // ── ContextError ─────────────────────────────────────────────────────────

    #[test]
    fn test_display_path_traversal() {
        let e = ContextError::PathTraversal(PathBuf::from("a/../b"));
        assert!(e.to_string().contains(".."));
        assert!(e.to_string().contains("path traversal"));
    }

    #[test]
    fn test_display_not_a_directory() {
        let e = ContextError::NotADirectory(PathBuf::from("/no/such/dir"));
        assert!(e.to_string().contains("no/such/dir"));
    }

    #[test]
    fn test_display_permission_denied() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = ContextError::PermissionDenied(PathBuf::from("/locked"), io_err);
        assert!(e.to_string().contains("locked"));
        assert!(e.to_string().contains("permissions"));
    }

    #[test]
    fn test_source_permission_denied_has_source() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = ContextError::PermissionDenied(PathBuf::from("/locked"), io_err);
        assert!(e.source().is_some());
    }

    #[test]
    fn test_source_path_traversal_is_none() {
        use std::error::Error;
        let e = ContextError::PathTraversal(PathBuf::from("a/b"));
        assert!(e.source().is_none());
    }
}
