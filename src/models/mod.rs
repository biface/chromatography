//! Physical models for chromatography simulation
//!
//! All models implement the [`PhysicalModel`](crate::physics::PhysicalModel) trait.
//! The solver calls `compute_physics` at each time step — models are responsible
//! for the physics (transport, adsorption), the solver for the time integration.
//!
//! # Available Models
//!
//! ## [`LangmuirSingle`] — single species
//!
//! One chemical species transported through the column with a Langmuir isotherm.
//! Use this model to study retention and peak shape for a pure compound.
//!
//! ## [`LangmuirMulti`] — multiple species with competitive adsorption
//!
//! Two or more species competing for the same adsorption sites. The presence
//! of one species reduces the capacity available to all others, producing the
//! characteristic displacement and band-crossing effects of real mixtures.
//!
//! # Injection
//!
//! Both models use [`TemporalInjection`] to define how concentration enters
//! the column at the inlet ($z = 0$) as a function of time. The solver writes
//! the current time into the `PhysicalState` metadata before each call to
//! `compute_physics`.

// =================================================================================================
// Module Declarations
// =================================================================================================

//mod langmuir_single_simple;
pub mod injection;
pub mod langmuir_single;
pub mod langmuir_multi;
// =================================================================================================
// Public Re-exports
// =================================================================================================


//pub use langmuir_single_simple::LangmuirSingleSimple;
pub use langmuir_single::LangmuirSingle;
pub use langmuir_multi::{SpeciesParams, LangmuirMulti};
pub use injection::TemporalInjection;
