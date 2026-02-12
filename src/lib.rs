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
//! use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
//! use chrom_rs::solver::{EulerSolver, Solver, SolverConfiguration, Scenario, DomainBoundaries};
//! use nalgebra::DVector;
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
//! # fn main() -> Result<(), String> {
//! // 1. Configure physical model and scenario
//! let model = Box::new(MyModel);
//! let initial_state = model.setup_initial_state();
//! let boundaries = DomainBoundaries::temporal(initial_state);
//! let scenario = Scenario::new(model, boundaries);
//!
//! // 2. Configure solver
//! let config = SolverConfiguration::time_evolution(
//!     600.0,    // 10 minutes total time
//!     1000,     // 1000 time steps
//! );
//!
//! // 3. Run simulation
//! let solver = EulerSolver::new();
//! let result = solver.solve(&scenario, &config)?;
//!
//! // 4. Access results
//! println!("Simulation completed!");
//! println!("Trajectory length: {}", result.len());
//! # Ok(())
//! # }
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

pub mod models;
pub mod solver;

pub mod prelude {
    //! Convenient imports for common usage
    //!
    //! ```rust
    //!
    //! use chrom_rs::prelude::*;
    //! ```
    pub use crate::physics::{PhysicalData,
                             PhysicalQuantity,
                             PhysicalState,
                             PhysicalModel};
    pub use crate::solver::{Solver,
                            SolverConfiguration,
                            SolverType,
                            Scenario,
                            SimulationResult,
                            EulerSolver,
                            RK4Solver};
}