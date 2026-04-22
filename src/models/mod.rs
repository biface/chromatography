//! Physical models for chromatography simulation.
//!
//! All models implement the [`PhysicalModel`](crate::physics::PhysicalModel) trait.
//! The solver calls `compute_physics` at each time step — models are responsible
//! for the physics (transport, adsorption), the solver for the time integration.
//!
//! # Available Models
//!
//! ## [`LangmuirSingle`](crate::models::LangmuirSingle) — single species
//!
//! One chemical species transported through the column with a Langmuir isotherm.
//! Use this model to study retention and peak shape for a pure compound.
//!
//! ## [`LangmuirMulti`](crate::models::LangmuirMulti) — multiple species with competitive adsorption
//!
//! Two or more species competing for the same adsorption sites. The presence
//! of one species reduces the capacity available to all others, producing the
//! characteristic displacement and band-crossing effects of real mixtures.
//!
//! # Injection
//!
//! Both models use [`TemporalInjection`](crate::models::TemporalInjection) to
//! define how concentration enters the column at the inlet ($z = 0$) as a
//! function of time. The solver writes the current time into the `PhysicalState`
//! metadata before each call to `compute_physics`.

// ================================================================================================
// Module Declarations
// ================================================================================================

/// Temporal injection profiles for inlet boundary conditions.
pub mod injection;

/// Multi-species Langmuir chromatography model with competitive adsorption.
pub mod langmuir_multi;

/// Single-species Langmuir chromatography model.
pub mod langmuir_single;

// ================================================================================================
// Public Re-exports
// ================================================================================================

pub use injection::TemporalInjection;
pub use langmuir_multi::{LangmuirMulti, SpeciesParams};
pub use langmuir_single::LangmuirSingle;
