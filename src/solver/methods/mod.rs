//! Numerical methods for solving differential equations
//!
//! This module contains concrete implementations of the [`Solver`](crate::solver::Solver) trait.
//!
//! # Architecture
//!
//! The separation between abstract solver interface (`solver::traits`) and concrete
//! implementations (`solver::methods`) follows the Open-Closed Principle:
//! - **Open** for extension: Add new methods without modifying existing code
//! - **Closed** for modification: The `Solver` trait is stable and never changes
//!
//! # Available Methods
//!
//! ## Explicit Time-Stepping Methods
//!
//! These methods are suitable for non-stiff ordinary differential equations (ODEs)
//! where the right-hand side function can be evaluated explicitly.
//!
//! - **[`EulerSolver`]**: Forward Euler method
//!   - Order: First-order O(dt)
//!   - Cost: 1 function evaluation per step
//!   - Use: Prototyping, educational purposes, non-stiff problems with relaxed accuracy
//!
//! - **[`RK4Solver`]**: Classical fourth-order Runge-Kutta
//!   - Order: Fourth-order O(dt⁴)
//!   - Cost: 4 function evaluations per step
//!   - Use: **Production simulations**, non-stiff to moderately stiff problems
//!
//! # Future Methods (Planned)
//!
//! ## Adaptive Methods (v0.2.0+)
//! - **RK45**: Runge-Kutta-Fehlberg with adaptive step size control
//! - **Dormand-Prince**: Higher-order adaptive method
//!
//! ## Implicit Methods (v0.3.0+)
//! - **BDF**: Backward Differentiation Formulas for stiff problems
//! - **Rosenbrock**: Semi-implicit methods
//!
//! ## Iterative Methods (v0.4.0+)
//! - **Newton-Raphson**: For steady-state problems
//! - **GMRES**: For large linear systems
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::solver::{EulerSolver, RK4Solver};
//! use chrom_rs::solver::{Scenario, DomainBoundaries, Solver, SolverConfiguration};
//! use chrom_rs::physics::{ PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData };
//! use nalgebra::DVector;
//!
//! struct MyModel;
//!
//! impl PhysicalModel for MyModel {
//!      fn points(&self) -> usize { 1 }
//!      fn compute_physics(&self, state: &PhysicalState) -> PhysicalState { state.clone() }
//!      fn setup_initial_state(&self) -> PhysicalState {
//!          PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(DVector::from_vec(vec![1.0])))
//!      }
//!      fn name(&self) -> &str { "MyModel" }
//!  }
//!
//! fn main() -> Result<(), String> {
//!
//!     let model = Box::new(MyModel);
//!     let boundaries = DomainBoundaries::temporal(model.setup_initial_state());
//!     let scenario = Scenario::new(model, boundaries);
//!
//!
//!     // Using Euler for production
//!     let euler = EulerSolver::new();
//!     let configuration = SolverConfiguration::time_evolution(600.0, 10000);
//!     let result = euler.solve(&scenario, &configuration)?;
//!
//!     // Using Runge-Kutta for production
//!     let rk4 = RK4Solver::new();
//!     let configuration = SolverConfiguration::time_evolution(600.0, 10000);
//!     let result = euler.solve(&scenario, &configuration)?;
//!
//!     Ok(())
//! }
//!
//! ```
//!
//! # Design Philosophy
//!
//! Each solver is:
//! - **Self-contained**: No shared mutable state
//! - **Stateless**: Can be reused for multiple simulations
//! - **Well-tested**: 30+ tests per solver with ~95% coverage
//! - **Documented**: Complete rustdoc with mathematical background
//!
//! # Performance Considerations
//!
//! All solvers benefit from:
//! - **Rayon parallelization** (feature `parallel`) for large spatial grids
//! - **Configurable threshold** via `set_parallel_threshold()`
//! - **Efficient nalgebra** operations for matrix arithmetic
//!
//! See individual solver documentation for specific performance characteristics.

pub mod euler;
mod rk4;

// Re-exports for convenience
pub use euler::EulerSolver;
pub use rk4::RK4Solver;

