//! Pending-configuration builders for the interactive `config`/`build` command.
//!
//! These types hold configuration state that is accumulated field by field
//! across one or more `config` command occurrences — whether given in a
//! single CLI invocation or spread across a chained sequence (`dynamic-cli`
//! 0.9.0's command chaining, DD-026). Every field is optional and **nothing
//! is validated at this stage**; validation happens only when the pending
//! state is turned into a real [`crate::models::LangmuirSingle`] /
//! [`crate::models::LangmuirMulti`] / [`crate::solver::SolverConfiguration`]
//! (`save` or `run`).
//!
//! Related design decision: DD-016 (issue #53). Ticket: #67.

// ============================================================================
// Model builders
// ============================================================================

/// Pending model state — either single- or multi-species shape.
///
/// Which shape is active is decided by the *first* field ever set on the
/// slot: an initial `single` occurrence locks [`ModelBuilder::Single`], an
/// initial `multi` **or** `species` occurrence locks
/// [`ModelBuilder::Multi`] (`species` alone is enough — a scalar `multi`
/// occurrence is not required first, since nothing at this stage depends on
/// scalar fields being set before species are added).
///
/// Switching shape afterwards (e.g. `single` arrives after `species` already
/// locked `Multi`) resets the slot to the new shape rather than attempting
/// to merge incompatible field sets, and the caller is expected to surface
/// the returned [`ShapeSwitch`] as a visible warning.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelBuilder {
    Single(SingleModelBuilder),
    Multi(MultiModelBuilder),
}

/// Pending fields for a `LangmuirSingle` model.
///
/// Field names mirror `LangmuirSingle`'s own struct fields exactly (not its
/// constructor's `spatial_points` parameter name, which differs from the
/// `n_points` field it sets).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SingleModelBuilder {
    pub lambda: Option<f64>,
    pub langmuir_k: Option<f64>,
    /// `f64` here — `LangmuirSingle::port_number` is `f64`, unlike
    /// [`SpeciesBuilder::port_number`] (`u32`, mirroring `SpeciesParams`).
    pub port_number: Option<f64>,
    pub column_length: Option<f64>,
    pub n_points: Option<usize>,
    pub dz: Option<f64>,
    pub fe: Option<f64>,
    pub ue: Option<f64>,
}

impl SingleModelBuilder {
    /// Merges `other` into `self`: any field set in `other` overwrites the
    /// corresponding field in `self`; fields left unset in `other` are
    /// untouched. Matches the "one call setting all fields produces the
    /// same pending state as N chained single-field calls" requirement.
    pub fn merge(&mut self, other: SingleModelBuilder) {
        if other.lambda.is_some() {
            self.lambda = other.lambda;
        }
        if other.langmuir_k.is_some() {
            self.langmuir_k = other.langmuir_k;
        }
        if other.port_number.is_some() {
            self.port_number = other.port_number;
        }
        if other.column_length.is_some() {
            self.column_length = other.column_length;
        }
        if other.n_points.is_some() {
            self.n_points = other.n_points;
        }
        if other.dz.is_some() {
            self.dz = other.dz;
        }
        if other.fe.is_some() {
            self.fe = other.fe;
        }
        if other.ue.is_some() {
            self.ue = other.ue;
        }
    }
}

/// Pending fields for a `LangmuirMulti` model.
///
/// Scalar field names mirror `LangmuirMulti`'s own struct fields exactly.
/// `species` accumulates — it is never reset by a scalar-only `multi`
/// occurrence, and a repeated species name is accepted here (name-collision
/// checking belongs to `save`/`run`, not to construction).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MultiModelBuilder {
    pub n_points: Option<usize>,
    pub porosity: Option<f64>,
    pub velocity: Option<f64>,
    pub column_length: Option<f64>,
    pub dz: Option<f64>,
    pub fe: Option<f64>,
    pub ue: Option<f64>,
    pub stationary_fraction: Option<f64>,
    pub species: Vec<SpeciesBuilder>,
}

impl MultiModelBuilder {
    /// Merges the *scalar* fields of `other` into `self`, in place.
    /// `other.species` is ignored here — species are added exclusively via
    /// [`Self::push_species`], never merged as a batch, so that ordering
    /// across chained occurrences stays predictable.
    pub fn merge_scalars(&mut self, other: MultiModelBuilder) {
        if other.n_points.is_some() {
            self.n_points = other.n_points;
        }
        if other.porosity.is_some() {
            self.porosity = other.porosity;
        }
        if other.velocity.is_some() {
            self.velocity = other.velocity;
        }
        if other.column_length.is_some() {
            self.column_length = other.column_length;
        }
        if other.dz.is_some() {
            self.dz = other.dz;
        }
        if other.fe.is_some() {
            self.fe = other.fe;
        }
        if other.ue.is_some() {
            self.ue = other.ue;
        }
        if other.stationary_fraction.is_some() {
            self.stationary_fraction = other.stationary_fraction;
        }
    }

    /// Appends one species to the list, in call order. Never replaces or
    /// deduplicates — a repeated name is accepted at this stage.
    pub fn push_species(&mut self, species: SpeciesBuilder) {
        self.species.push(species);
    }
}

/// Pending fields for one `SpeciesParams` entry.
///
/// `port_number` is `u32` here — mirroring `SpeciesParams::port_number`,
/// which differs in type from `LangmuirSingle::port_number` (`f64`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeciesBuilder {
    pub name: Option<String>,
    pub lambda: Option<f64>,
    pub langmuir_k: Option<f64>,
    pub port_number: Option<u32>,
}

/// Result of a [`ModelBuilder`] shape change, returned so the caller (a
/// future `config`/`build` command handler) can print a visible warning
/// rather than resetting silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeSwitch {
    /// The slot held `Multi` and has just been reset to `Single`.
    ToSingle,
    /// The slot held `Single` and has just been reset to `Multi`.
    ToMulti,
}

impl std::fmt::Display for ShapeSwitch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeSwitch::ToSingle => write!(
                f,
                "warning: pending model switched from multi-species to single-species — \
                 previously accumulated multi-species fields (including any species list) \
                 were discarded"
            ),
            ShapeSwitch::ToMulti => write!(
                f,
                "warning: pending model switched from single-species to multi-species — \
                 previously accumulated single-species fields were discarded"
            ),
        }
    }
}

// ============================================================================
// Solver builder
// ============================================================================

/// Pending fields for `SolverConfiguration` + the solver-name string.
///
/// Field names mirror `solver.yml`'s own keys (`solver_type` maps to the
/// file's `type:` key).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolverBuilder {
    /// `"RK4"` or `"Euler"`.
    pub solver_type: Option<String>,
    pub total_time: Option<f64>,
    pub time_steps: Option<usize>,
    /// `None` means "unset" at this stage — matches `solver.yml`'s own
    /// null-means-full-trajectory convention once the value is provided.
    pub step: Option<usize>,
}

impl SolverBuilder {
    /// Merges `other` into `self`, field by field, last-write-wins.
    pub fn merge(&mut self, other: SolverBuilder) {
        if other.solver_type.is_some() {
            self.solver_type = other.solver_type;
        }
        if other.total_time.is_some() {
            self.total_time = other.total_time;
        }
        if other.time_steps.is_some() {
            self.time_steps = other.time_steps;
        }
        if other.step.is_some() {
            self.step = other.step;
        }
    }
}

// ============================================================================
// Scenario builder
// ============================================================================

/// Pending fields for one `TemporalInjection` (`Gaussian`/`Dirac`/`None`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InjectionBuilder {
    /// `"Gaussian"`, `"Dirac"`, or `"None"`.
    pub injection_type: Option<String>,
    pub center: Option<f64>,
    pub width: Option<f64>,
    pub peak_concentration: Option<f64>,
    pub time: Option<f64>,
    pub amount: Option<f64>,
}

impl InjectionBuilder {
    /// Merges `other` into `self`, field by field, last-write-wins.
    pub fn merge(&mut self, other: InjectionBuilder) {
        if other.injection_type.is_some() {
            self.injection_type = other.injection_type;
        }
        if other.center.is_some() {
            self.center = other.center;
        }
        if other.width.is_some() {
            self.width = other.width;
        }
        if other.peak_concentration.is_some() {
            self.peak_concentration = other.peak_concentration;
        }
        if other.time.is_some() {
            self.time = other.time;
        }
        if other.amount.is_some() {
            self.amount = other.amount;
        }
    }
}

/// Pending `scenario.yml` state: initial condition, default injection, and
/// per-species overrides.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScenarioBuilder {
    /// Only `"zero"` is a supported value today, mirrored as a plain
    /// `String` here rather than an enum — validation stays at `save`/`run`.
    pub initial_condition: Option<String>,
    pub default_injection: Option<InjectionBuilder>,
    /// `(species name, injection)` — a repeated species name overwrites its
    /// previous entry (last write wins per species), unlike
    /// [`MultiModelBuilder::species`] which never deduplicates.
    pub species_overrides: Vec<(String, InjectionBuilder)>,
}

impl ScenarioBuilder {
    /// Sets or merges the initial condition, last-write-wins.
    pub fn set_initial_condition(&mut self, value: String) {
        self.initial_condition = Some(value);
    }

    /// Merges `injection` into the default injection slot.
    pub fn merge_default_injection(&mut self, injection: InjectionBuilder) {
        match &mut self.default_injection {
            Some(existing) => existing.merge(injection),
            None => self.default_injection = Some(injection),
        }
    }

    /// Merges `injection` into the override for `species`, creating a new
    /// entry (appended, in call order) if `species` has no override yet.
    pub fn merge_species_override(&mut self, species: &str, injection: InjectionBuilder) {
        if let Some((_, existing)) = self
            .species_overrides
            .iter_mut()
            .find(|(name, _)| name == species)
        {
            existing.merge(injection);
        } else {
            self.species_overrides
                .push((species.to_string(), injection));
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── SingleModelBuilder ───────────────────────────────────────────────

    #[test]
    fn test_single_merge_overwrites_only_set_fields() {
        let mut base = SingleModelBuilder {
            lambda: Some(1.2),
            langmuir_k: Some(0.4),
            ..Default::default()
        };
        let incoming = SingleModelBuilder {
            langmuir_k: Some(0.9),
            n_points: Some(100),
            ..Default::default()
        };
        base.merge(incoming);

        assert_eq!(base.lambda, Some(1.2)); // untouched
        assert_eq!(base.langmuir_k, Some(0.9)); // overwritten
        assert_eq!(base.n_points, Some(100)); // newly set
        assert_eq!(base.port_number, None);
    }

    #[test]
    fn test_single_one_call_equals_chained_calls() {
        let mut one_call = SingleModelBuilder::default();
        one_call.merge(SingleModelBuilder {
            lambda: Some(1.2),
            langmuir_k: Some(0.4),
            port_number: Some(2.0),
            column_length: Some(0.25),
            n_points: Some(100),
            dz: Some(0.0025),
            fe: Some(1.5),
            ue: Some(0.0025),
        });

        let mut chained = SingleModelBuilder::default();
        chained.merge(SingleModelBuilder {
            lambda: Some(1.2),
            ..Default::default()
        });
        chained.merge(SingleModelBuilder {
            langmuir_k: Some(0.4),
            ..Default::default()
        });
        chained.merge(SingleModelBuilder {
            port_number: Some(2.0),
            ..Default::default()
        });
        chained.merge(SingleModelBuilder {
            column_length: Some(0.25),
            ..Default::default()
        });
        chained.merge(SingleModelBuilder {
            n_points: Some(100),
            ..Default::default()
        });
        chained.merge(SingleModelBuilder {
            dz: Some(0.0025),
            ..Default::default()
        });
        chained.merge(SingleModelBuilder {
            fe: Some(1.5),
            ..Default::default()
        });
        chained.merge(SingleModelBuilder {
            ue: Some(0.0025),
            ..Default::default()
        });

        assert_eq!(one_call, chained);
    }

    // ── MultiModelBuilder / SpeciesBuilder ───────────────────────────────

    #[test]
    fn test_multi_species_accumulate_in_call_order() {
        let mut multi = MultiModelBuilder::default();
        multi.push_species(SpeciesBuilder {
            name: Some("Ascorbic".to_string()),
            ..Default::default()
        });
        multi.push_species(SpeciesBuilder {
            name: Some("Erythorbic".to_string()),
            ..Default::default()
        });

        assert_eq!(multi.species.len(), 2);
        assert_eq!(multi.species[0].name.as_deref(), Some("Ascorbic"));
        assert_eq!(multi.species[1].name.as_deref(), Some("Erythorbic"));
    }

    #[test]
    fn test_multi_duplicate_species_name_accepted_at_construction() {
        let mut multi = MultiModelBuilder::default();
        multi.push_species(SpeciesBuilder {
            name: Some("A".to_string()),
            ..Default::default()
        });
        multi.push_species(SpeciesBuilder {
            name: Some("A".to_string()),
            ..Default::default()
        });

        // No dedup, no error here — name-collision checking is deferred.
        assert_eq!(multi.species.len(), 2);
    }

    #[test]
    fn test_multi_merge_scalars_ignores_species() {
        let mut base = MultiModelBuilder::default();
        base.push_species(SpeciesBuilder {
            name: Some("A".to_string()),
            ..Default::default()
        });

        base.merge_scalars(MultiModelBuilder {
            n_points: Some(100),
            species: vec![SpeciesBuilder {
                name: Some("B".to_string()),
                ..Default::default()
            }],
            ..Default::default()
        });

        assert_eq!(base.n_points, Some(100));
        // Species list untouched by merge_scalars — still just "A".
        assert_eq!(base.species.len(), 1);
        assert_eq!(base.species[0].name.as_deref(), Some("A"));
    }

    // ── SolverBuilder ─────────────────────────────────────────────────────

    #[test]
    fn test_solver_merge_last_write_wins() {
        let mut base = SolverBuilder {
            solver_type: Some("RK4".to_string()),
            total_time: Some(600.0),
            ..Default::default()
        };
        base.merge(SolverBuilder {
            time_steps: Some(10_000),
            ..Default::default()
        });

        assert_eq!(base.solver_type.as_deref(), Some("RK4"));
        assert_eq!(base.total_time, Some(600.0));
        assert_eq!(base.time_steps, Some(10_000));
        assert_eq!(base.step, None);
    }

    // ── ScenarioBuilder / InjectionBuilder ───────────────────────────────

    #[test]
    fn test_scenario_default_injection_merges() {
        let mut scenario = ScenarioBuilder::default();
        scenario.merge_default_injection(InjectionBuilder {
            injection_type: Some("Gaussian".to_string()),
            center: Some(10.0),
            ..Default::default()
        });
        scenario.merge_default_injection(InjectionBuilder {
            width: Some(3.0),
            ..Default::default()
        });

        let default_injection = scenario.default_injection.expect("must be set");
        assert_eq!(
            default_injection.injection_type.as_deref(),
            Some("Gaussian")
        );
        assert_eq!(default_injection.center, Some(10.0));
        assert_eq!(default_injection.width, Some(3.0));
    }

    #[test]
    fn test_scenario_species_override_merges_by_species_name() {
        let mut scenario = ScenarioBuilder::default();
        scenario.merge_species_override(
            "Erythorbic",
            InjectionBuilder {
                injection_type: Some("Dirac".to_string()),
                time: Some(5.0),
                ..Default::default()
            },
        );
        scenario.merge_species_override(
            "Erythorbic",
            InjectionBuilder {
                amount: Some(0.05),
                ..Default::default()
            },
        );
        scenario.merge_species_override(
            "Ascorbic",
            InjectionBuilder {
                injection_type: Some("None".to_string()),
                ..Default::default()
            },
        );

        assert_eq!(scenario.species_overrides.len(), 2); // merged, not appended twice
        let (name, injection) = &scenario.species_overrides[0];
        assert_eq!(name, "Erythorbic");
        assert_eq!(injection.time, Some(5.0));
        assert_eq!(injection.amount, Some(0.05));
    }

    // ── ShapeSwitch ───────────────────────────────────────────────────────

    #[test]
    fn test_shape_switch_display() {
        assert!(ShapeSwitch::ToSingle.to_string().contains("single-species"));
        assert!(ShapeSwitch::ToMulti.to_string().contains("multi-species"));
    }
}
