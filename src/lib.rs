//! chrom-rs: Chromatography Simulation Framework
//!
//! A flexible and extensible framework for simulating chromatographic processes
//! using numerical methods. Built with Rust for performance and safety.
//!
//! # Architecture
//!
//! chrom-rs is built on two core principles:
//!
//! 1. **Separation of Physics and Numerics**
//!    - Physical models define equations (what to solve)
//!    - Numerical solvers provide methods (how to solve)
//!
//! 2. **Extensibility and Type Safety**
//!    - Trait-based design for easy extension
//!    - Type-safe state management
//!    - Stable API (v0.1.0+)
//!
//! # Quick Start
//!
//! ```rust
//! use chrom_rs::physics::langmuir1d::{LangmuirModel1D, LangmuirIsothermParams, System1DParams};
//! use chrom_rs::solver::{EulerSolver, Solver, SolverConfig};
//!
//! // 1. Configure physical model
//! let isotherm = LangmuirIsothermParams {
//!     lambda: 1.0,
//!     capacity: 6.0,
//!     langmuir_constant: 0.5,
//! };
//!
//! let system = System1DParams {
//!     length: 0.3,
//!     n_points: 35,
//!     velocity: 0.0025,
//!     dispersion: 1e-9,
//!     retention_factor: 1.5,
//! };
//!
//! let model = LangmuirModel1D::new(isotherm, system)?;
//!
//! // 2. Configure solver
//! let config = SolverConfig {
//!     total_time: 600.0,
//!     time_steps: 10000,
//!     tolerance: None,
//!     max_iterations: None,
//! };
//!
//! // 3. Run simulation
//! let solver = EulerSolver;
//! let result = solver.solve(&model, &config)?;
//!
//! // 4. Access results
//! println!("Simulation completed!");
//! println!("Time points: {}", result.time_points.len());
//! ```
//!
//! # Modules
//!
//! - [`physics`]: Physical models (equations)
//! - [`solver`]: Numerical solvers (methods)
//! - [`cli`]: Command-line interface (optional)
//! - [`output`]: Result visualization and export (optional)

// Core modules
pub mod physics;
pub mod solver;

pub mod prelude {
    //! Convenient imports for common usage
    //!
    //! ```rust
    //!
    //! use chrom_rs::prelude::*;
    //! ```
    pub use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity};

}