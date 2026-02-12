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
//! ```rust,ignore
//! use chrom_rs::models::LangmuirSingleSimple;
//!
//! let model = LangmuirSingleSimple::new(
//!     1.2,   // λ
//!     0.4,   // K̃
//!     2.0,   // N
//!     0.4,   // ε
//!     0.001, // u
//!     0.25,  // L
//!     100,   // nz
//! );
//! ```
//!

// =================================================================================================
// Module Declarations
// =================================================================================================

mod langmuir_single_simple;

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
/// ```rust,ignore
/// use chrom_rs::models::LangmuirSingleSimple;
///
/// let model = LangmuirSingleSimple::new(
///     1.2,   // λ
///     0.4,   // K̃ [L/mol]
///     2.0,   // N [mol/L]
///     0.4,   // ε (porosity)
///     0.001, // u [m/s]
///     0.25,  // L [m]
///     100,   // nz
/// );
///
/// // Direct field access
/// let lambda = model.lambda;
/// let langmuir_k = model.langmuir_k;
/// ```
pub use langmuir_single_simple::LangmuirSingleSimple;
