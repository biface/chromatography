//! Numerical solvers
//!
//! This module provides traits and implementations for numerical solvers.
//! A numerical solver applies a numerical method to solve the equations
//! provided by a physical model.
//!
//! # Core Concepts
//!
//! - **Solver**: Applies a numerical method to solve physics equations
//! - **Solver Config**: Configuration for the numerical method (time steps, tolerance, etc.)
//! - **Simulation Result**: Contains the solution (time evolution, final state, metadata)
//!
//! # Architecture
//!
//! Solvers are **independent from physical models**:
//! - The solver receives equations from the model via `compute_physics()`
//! - The solver applies its numerical method (time integration, iteration, etc.)
//! - The solver returns the result to the user
//!
//! This separation allows:
//! - Same solver for different physics (transport, heat, reaction, etc.)
//! - Different solvers for same physics (compare Euler vs Runge-Kutta)
//! - Easy benchmarking and method comparison
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::physics::PhysicalModel;
//! use chrom_rs::solver::{Solver, SolverConfig};
//! use chrom_rs::solver::euler::EulerSolver;
//!
//! // Create solver configuration
//! let config = SolverConfig {
//!     total_time: 600.0,      // 10 minutes
//!     time_steps: 10000,      // 10000 steps
//!     tolerance: None,
//!     max_iterations: None,
//! };
//!
//! // Create solver
//! let solver = EulerSolver;
//!
//! // Solve (model must implement PhysicalModel)
//! let result = solver.solve(&model, &config)?;
//!
//! // Access results
//! println!("Simulation completed in {} steps", result.time_points.len());
//! println!("Final state: {:?}", result.final_state);
//! ```
//!
//! # Solver Types
//!
//! Different types of numerical solvers:
//!
//! - **Time Integrators**: Solve dy/dt = f(y) over time
//!   - Explicit methods: Forward Euler, Runge-Kutta
//!   - Implicit methods: Backward Euler, Crank-Nicolson
//!
//! - **Iterative Solvers**: Solve F(x) = 0 for steady-state
//!   - Newton-Raphson, Fixed-point iteration
//!
//! - **Direct Solvers**: Solve Ax = b (linear systems)
//!   - LU decomposition, Cholesky
//!
//! # Implementing a New Solver
//!
//! To create a new numerical solver, implement the `Solver` trait:
//!
//! ```rust
//! use chrom_rs::solver::{Solver, SolverConfig, SimulationResult};
//! use chrom_rs::physics::PhysicalModel;
//!
//! struct MyCustomSolver;
//!
//! impl Solver for MyCustomSolver {
//!     fn solve(
//!         &self,
//!         model: &dyn PhysicalModel,
//!         config: &SolverConfig,
//!     ) -> Result<SimulationResult, String> {
//!         // Implement your numerical method here
//!         // 1. Get initial state from model
//!         // 2. Apply numerical scheme
//!         // 3. Return result
//!     }
//!
//!     fn name(&self) -> &str {
//!         "My Custom Solver"
//!     }
//! }
//! ```
//!
//! # Available Solvers
//!
//! Currently implemented numerical solvers:
//! - **Euler**: Forward Euler explicit time integrator (1st order)
//!
//! # Performance Considerations
//!
//! - **Explicit methods** (Euler): Simple, fast per step, but require small time steps
//! - **Implicit methods** (future): More stable, allow larger time steps, but require solving linear systems
//! - **Adaptive methods** (future): Automatically adjust time step for accuracy/efficiency

// module declaration
mod traits;
mod boundary;
// Solver implementation

// Re-export implementation