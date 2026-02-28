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
//! use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
//!
//! # struct MyModel;
//! # impl PhysicalModel for MyModel {
//! #     fn points(&self) -> usize { 1 }
//! #     fn compute_physics(&self, state: &PhysicalState) -> PhysicalState { state.clone() }
//! #     fn setup_initial_state(&self) -> PhysicalState {
//! #         PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(nalgebra::DVector::from_vec(vec![1.0])))
//! #     }
//! #     fn name(&self) -> &str { "MyModel" }
//! # }
//! # fn main() {
//! // Create a physical model
//! let model = MyModel;
//!
//! // Get initial state
//! let initial_state = model.setup_initial_state();
//!
//! // Compute physics at current state
//! let physics_result = model.compute_physics(&initial_state);
//! # }
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
//!     fn points(&self) -> usize {
//!         // Return number of spatial points
//!         1
//!     }
//!
//!     fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
//!         // Compute and return the physics equations
//!         state.clone()
//!     }
//!
//!     fn setup_initial_state(&self) -> PhysicalState {
//!         // Create initial state
//!         # use chrom_rs::physics::{PhysicalQuantity, PhysicalData};
//!         # PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(nalgebra::DVector::from_vec(vec![1.0])))
//!         /* ... */
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
pub mod data;
// Model implementation

// re-export commonly used types for convenience
pub use data::PhysicalData;
pub use traits::{
    PhysicalModel,
    PhysicalQuantity,
    PhysicalState, };