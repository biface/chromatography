//! Physical models
//!
//! This module provides traits and implementations for physical models.
//! A physical model encapsulates the physics equations of a system
//! (e.g., chromatography, heat transfer, reaction-diffusion).
//!
//! # Core Concepts
//!
//! - **Physical Model**: Computes the physics equations at a given state
//! - **Physical State**: Container for all physical quantities (concentration, temperature, etc.)
//! - **Physical Quantity**: Type-safe identifier for physical variables
//!
//! # Architecture
//!
//! Physical models are **separate from numerical solvers**:
//! - The model provides the **equations** (physics)
//! - The solver provides the **method** to solve them (numerics)
//!
//! This separation allows:
//! - Same model with different solvers (Euler, Runge-Kutta, etc.)
//! - Same solver with different models (chromatography, thermal, etc.)
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::physics::{PhysicalModel, PhysicalState};
//! use chrom_rs::physics::langmuir1d::LangmuirModel1D;
//!
//! // Create a physical model
//! let model = LangmuirModel1D::new(isotherm_params, system_params)?;
//!
//! // Get initial state
//! let initial_state = model.create_initial_state();
//!
//! // Compute physics at current state
//! let physics_result = model.compute_physics(&initial_state);
//! ```
//!
//! # Implementing a New Physical Model
//!
//! To create a new physical model, implement the `PhysicalModel` trait:
//!
//! ```rust
//! use chrom_rs::physics::{PhysicalModel, PhysicalState};
//!
//! struct MyCustomModel {
//!     // Model parameters
//! }
//!
//! impl PhysicalModel for MyCustomModel {
//!     fn n_spatial_points(&self) -> usize {
//!         // Return number of spatial points
//!     }
//!
//!     fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
//!         // Compute and return the physics equations
//!     }
//!
//!     fn create_initial_state(&self) -> PhysicalState {
//!         // Create initial state
//!     }
//!
//!     fn name(&self) -> &str {
//!         "My Custom Model"
//!     }
//! }
//! ```
//!
//! # Available Models
//!
//! Currently implemented physical models:
//! - **Langmuir 1D**: 1D chromatography with modified Langmuir isotherm

// module declaration
pub mod traits;
mod data;
// Model implementation

// re-export commonly used types for convenience
pub use traits::{
    PhysicalModel,
    PhysicalState,
    PhysicalQuantity};