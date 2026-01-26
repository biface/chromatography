//! Numerical solvers
//!
//! This module provides traits and implementations for numerical solvers.
//! A numerical solver applies a numerical method to solve the equations
//! provided by a physical model within a specific scenario.
//!
//! # Core Concepts
//!
//! ## The Architecture (WHAT vs HOW)
//!
//! The solver architecture separates concerns into three layers:
//!
//! 1. **Scenario** (`Scenario`) - WHAT to solve
//!    - Physical model (equations)
//!    - Domain boundaries (boundary conditions)
//!    - Problem definition
//!
//! 2. **Configuration** (`SolverConfiguration`) - HOW to solve
//!    - Solver type (time evolution, iterative, etc.)
//!    - Numerical parameters (time steps, tolerance, etc.)
//!    - Method selection
//!
//! 3. **Solver** (`Solver` trait) - The numerical method
//!    - Applies the numerical scheme
//!    - Returns the solution
//!    - Independent of physics
//!
//! This separation allows:
//! - Same solver for different physics
//! - Different solvers for same scenario
//! - Easy benchmarking and method comparison
//! - Flexible configuration without code changes
//!
//! # Module Organization
//!
//! - **`traits`**: Core trait definitions and types
//!   - `Solver` trait: Stable interface for all solvers
//!   - `SolverType`: Enumeration of solver types
//!   - `SolverConfiguration`: Configuration structure
//!   - `SimulationResult`: Result structure
//!
//! - **`boundary`**: Boundary conditions and domain definition
//!   - `DomainBoundaries`: Spatial and temporal boundaries
//!   - Factory methods for common boundary types
//!
//! - **`scenario`**: Problem definition
//!   - `Scenario`: Combines model + boundaries
//!   - Validation and introspection methods
//!
//! - **Solver implementations** (future):
//!   - `euler`: Forward Euler method
//!   - `runge_kutta`: RK4 method
//!   - etc.
//!
//! # Quick Start Example
//!
//! ```rust
//! use chrom_rs::physics::PhysicalModel;
//! use chrom_rs::solver::{
//!     Scenario, DomainBoundaries, SolverConfiguration, Solver
//! };
//! // use chrom_rs::solver::euler::EulerSolver;  // Future implementation
//!
//! // 1. Create scenario (WHAT to solve)
//! let model: Box<dyn PhysicalModel> = /* your physical model */;
//! let boundaries = DomainBoundaries::temporal(initial_state);
//! let scenario = Scenario::new(model, boundaries);
//!
//! // 2. Create configuration (HOW to solve)
//! let config = SolverConfiguration::time_evolution(
//!     600.0,    // 10 minutes total time
//!     10000,    // 10000 time steps
//! );
//!
//! // 3. Create solver and solve
//! // let solver = EulerSolver;
//! // let result = solver.solve(&scenario, &config)?;
//!
//! // 4. Access results
//! // println!("Simulation completed in {} steps", result.len());
//! // println!("Final state: {:?}", result.final_state);
//! ```
//!
//! # Workflow Diagram
//!
//! ```text
//! ┌─────────────────┐
//! │  Physical Model │  (equations)
//! └────────┬────────┘
//!          │
//!          ├──────────────┐
//!          │              │
//! ┌────────▼────────┐ ┌──▼──────────────┐
//! │ Domain          │ │ Scenario        │ ← WHAT to solve
//! │ Boundaries      │ │ (model + bounds)│
//! └─────────────────┘ └────────┬────────┘
//!                              │
//!                     ┌────────▼─────────────┐
//!                     │ Solver Configuration │ ← HOW to solve
//!                     │ (type + parameters)  │
//!                     └────────┬─────────────┘
//!                              │
//!                     ┌────────▼────────┐
//!                     │ Numerical Solver│ ← The method
//!                     │ (Euler, RK4...) │
//!                     └────────┬────────┘
//!                              │
//!                     ┌────────▼────────────┐
//!                     │ Simulation Result   │ ← The solution
//!                     │ (trajectory + meta) │
//!                     └─────────────────────┘
//! ```
//!
//! # Solver Types
//!
//! Different types of numerical solvers for different problem classes:
//!
//! ## Time Integrators
//!
//! Solve dy/dt = f(y) over time:
//!
//! - **Explicit methods**: Forward Euler, Runge-Kutta
//!   - Simple, fast per step
//!   - Require small time steps for stability
//!   - Good for non-stiff problems
//!
//! - **Implicit methods**: Backward Euler, Crank-Nicolson
//!   - More stable, allow larger time steps
//!   - Require solving linear systems
//!   - Good for stiff problems
//!
//! ## Iterative Solvers
//!
//! Solve F(x) = 0 for steady-state:
//!
//! - Newton-Raphson
//! - Fixed-point iteration
//! - Gradient descent
//!
//! ## Direct Solvers
//!
//! Solve Ax = b (linear systems):
//!
//! - LU decomposition
//! - Cholesky factorization
//! - QR decomposition
//!
//! # Creating a Scenario
//!
//! A scenario combines a physical model with boundary conditions:
//!
//! ```rust
//! use chrom_rs::solver::{Scenario, DomainBoundaries};
//! use chrom_rs::physics::{PhysicalState, PhysicalQuantity};
//! use nalgebra::DVector;
//!
//! // Define initial state
//! let initial_state = PhysicalState::new(
//!     PhysicalQuantity::Concentration,
//!     DVector::from_vec(vec![1.0, 0.5, 0.2]),
//! );
//!
//! // Create boundaries (temporal only for time evolution)
//! let boundaries = DomainBoundaries::temporal(initial_state);
//!
//! // Create scenario with a model
//! // let scenario = Scenario::new(model, boundaries);
//!
//! // Validate scenario
//! // scenario.validate()?;
//! ```
//!
//! # Configuring a Solver
//!
//! Different solver types require different configurations:
//!
//! ```rust
//! use chrom_rs::solver::{SolverConfiguration, SolverType};
//!
//! // Time evolution (ODE integration)
//! let config = SolverConfiguration::time_evolution(
//!     600.0,    // Total time (seconds)
//!     10000,    // Number of time steps
//! );
//!
//! // Iterative (convergence to steady-state)
//! let config = SolverConfiguration::iterative(
//!     1e-6,     // Convergence tolerance
//!     100,      // Maximum iterations
//! );
//!
//! // Analytical (exact solution at specific time)
//! let config = SolverConfiguration::analytical(5.0);
//!
//! // Spatial discretization (PDE on grid)
//! let config = SolverConfiguration::spatial_discretization(
//!     100,      // Grid points
//!     1000,     // Time steps
//! );
//!
//! // Validate before use
//! config.validate()?;
//! ```
//!
//! # Implementing a New Solver
//!
//! To create a new numerical solver, implement the `Solver` trait:
//!
//! ```rust
//! use chrom_rs::solver::{Solver, SolverConfiguration, SimulationResult, Scenario};
//!
//! /// My custom numerical solver
//! pub struct MyCustomSolver {
//!     // Solver-specific state (if needed)
//! }
//!
//! impl Solver for MyCustomSolver {
//!     fn solve(
//!         &self,
//!         scenario: &Scenario,
//!         config: &SolverConfiguration,
//!     ) -> Result<SimulationResult, String> {
//!         // 1. Validate configuration
//!         config.validate()?;
//!         scenario.validate()?;
//!
//!         // 2. Get initial state from scenario
//!         let initial_state = /* extract from scenario */;
//!
//!         // 3. Apply your numerical method
//!         let (time_points, trajectory, final_state) = /* your algorithm */;
//!
//!         // 4. Build and return result
//!         Ok(SimulationResult::new(time_points, trajectory, final_state))
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
//! Currently available numerical solvers:
//!
//! - **Forward Euler** (future): First-order explicit time integrator
//! - **Runge-Kutta 4** (future): Fourth-order explicit time integrator
//!
//! # Performance Considerations
//!
//! ## Choosing a Solver
//!
//! - **For non-stiff problems**: Explicit methods (Euler, RK4)
//!   - Fast per step
//!   - Require small time steps
//!
//! - **For stiff problems**: Implicit methods
//!   - More expensive per step
//!   - Allow much larger time steps
//!
//! - **For steady-state**: Iterative methods
//!   - No time stepping
//!   - Converge directly to solution
//!
//! ## Time Step Selection
//!
//! Rule of thumb for explicit methods:
//! - `dt < stability_limit` (problem-dependent)
//! - Start conservative, increase carefully
//! - Monitor solution for oscillations/divergence
//!
//! # Error Handling
//!
//! All solver methods return `Result<T, String>`:
//!
//! ```rust
//! // Example error handling
//! match solver.solve(&scenario, &config) {
//!     Ok(result) => {
//!         println!("Success! {} steps computed", result.len());
//!     }
//!     Err(e) => {
//!         eprintln!("Solver failed: {}", e);
//!         // Handle error...
//!     }
//! }
//! ```
//!
//! Common errors:
//! - Invalid configuration (negative time, zero steps)
//! - Invalid scenario (incompatible boundaries)
//! - Numerical instability (divergence, NaN values)
//! - Convergence failure (max iterations exceeded)

// =================================================================================================
// Module Declarations
// =================================================================================================
mod traits;
mod boundary;
mod scenario;

// Solver implementation

// Re-export implementation

pub use traits::{
    Solver,
    SolverType,
    SolverConfiguration,
    SimulationResult
};

// Boundary conditions

pub use boundary::DomainBoundaries;

// Scenario définition

pub use scenario::Scenario;