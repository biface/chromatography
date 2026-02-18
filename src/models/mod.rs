//! Physical models for chromatography simulation
//!
//! This module contains implementations of various chromatographic models,
//! particularly focusing on Langmuir isotherms for single and multi-component
//! adsorption.
//!
//! # Model Architecture
//!
//! All models implement the [`PhysicalModel`](crate::physics::PhysicalModel) trait,
//! which defines the interface for computing physical processes (transport,
//! adsorption, dispersion) from a given state.
//!
//! # Available Models
//!
//! ## Single-Component Langmuir Models
//!
//! Three implementations with different design trade-offs:
//!
//! ### 1. LangmuirSingleSimple ⭐⭐⭐⭐⭐
//!
//! **Best for**: Production use, default choice
//!
//! - Direct scalar field storage
//! - Zero overhead parameter access
//! - Simplest implementation
//! - ~72 bytes memory
//!
//! ```rust
//! use chrom_rs::models::{LangmuirSingle, TemporalInjection};
//!
//! let model = LangmuirSingle::new(
//!     1.2,   // λ
//!     0.4,   // K̃
//!     2.0,   // N
//!     0.4,   // ε
//!     0.001, // u
//!     0.25,  // L
//!     100,   // nz
//!     TemporalInjection::dirac(5.0, 0.1)
//! );
//! ```
//!

// =================================================================================================
// Module Declarations
// =================================================================================================

mod langmuir_single_simple;
mod injection;
mod langmuir_single;
// =================================================================================================
// Public Re-exports
// =================================================================================================

/// Single-component Langmuir model (scalar fields, zero overhead)
///
/// This is the simplest implementation with direct field access.
/// Best for extreme performance requirements.
///
/// # Example
///
/// ```rust
/// use chrom_rs::models::{LangmuirSingle, TemporalInjection};
/// use chrom_rs::physics::PhysicalModel;
///
/// let model = LangmuirSingle::new(
///     1.2,   // λ
///     0.4,   // K̃ \[L/mol\]
///     2.0,   // N \[mol/L\]
///     0.4,   // ε (porosity)
///     0.001, // u \[m/s\]
///     0.25,  // L \[m\]
///     100,   // `nz`
///     TemporalInjection::dirac(5.0, 0.1)
/// );
///
/// // Direct field access
/// assert_eq!(model.length(), 0.25);
/// assert_eq!(model.name(), "Langmuir single specie with temporal injection");
/// ```

pub use langmuir_single_simple::LangmuirSingleSimple;
pub use langmuir_single::LangmuirSingle;
pub use injection::TemporalInjection;
