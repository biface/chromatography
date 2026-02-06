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
//! - **Solver implementations**:
//!   - `EulerSolver`: Forward Euler method
//!   - `RK4Solver`: 4 Steps Runge Kutta method
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
mod methods;

// =================================================================================================
// Parallel Execution Threshold
// =================================================================================================
//
// Deciding *when* to hand work off to Rayon is a numerical-execution concern,
// not a physics concern.  It therefore lives here (solver) rather than in
// physics/data.rs.  See DD02 for the rationale.
//
// The threshold is stored in an AtomicUsize so that it can be changed at
// runtime (useful in benchmarks and tests) without requiring a mutex on every
// `apply()` call.  Relaxed ordering is sufficient: the value is a
// performance hint, not a synchronisation point.
// =================================================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

/// Default number of elements above which [`PhysicalData::apply()`] switches
/// to parallel iteration.
///
/// The crossover is set at 1 000 elements.  Below that point the overhead of
/// Rayon's thread-pool dispatch outweighs the per-element work for the
/// arithmetic closures that chromatography simulations typically use.
const DEFAULT_PARALLEL_THRESHOLD: usize = 999;

/// Runtime-configurable parallel-execution threshold.
///
/// Read via [`parallel_threshold()`], written via [`set_parallel_threshold()`].
static PARALLEL_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

/// Return the current parallel-execution threshold.
///
/// `PhysicalData::apply()` uses sequential iteration when the data contains
/// fewer elements than this value, and switches to Rayon when it contains
/// more — but only when the crate is compiled with the `parallel` feature.
///
/// # Example
///
/// ```rust
/// use chrom_rs::solver::parallel_threshold;
///
/// assert!(parallel_threshold() > 0);
/// ```
pub fn parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

/// Set the parallel-execution threshold to a new value.
///
/// # Panics
///
/// Panics when `threshold == 0`.  A zero-element threshold would force
/// parallel dispatch on every single-element `apply()`, which is never
/// the intended behaviour.
///
/// # Example
///
/// ```rust
/// use chrom_rs::solver::{parallel_threshold, set_parallel_threshold};
///
/// let previous = parallel_threshold();
/// set_parallel_threshold(2048);
/// assert_eq!(parallel_threshold(), 2048);
///
/// // Restore so other tests are not affected.
/// set_parallel_threshold(previous);
/// ```
pub fn set_parallel_threshold(threshold: usize) {
    assert!(threshold > 0, "parallel threshold must be at least 1");
    PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

/// RAII guard that saves the current threshold on construction and restores
/// it on drop.
///
/// Only compiled in test builds.  Prevents one test from leaking a modified
/// threshold value into the next.
///
/// ```rust,ignore
/// let _guard = crate::solver::ThresholdGuard::save(50);
/// // threshold is now 50 …
/// // … and is automatically restored when _guard is dropped.
/// ```
#[cfg(test)]
pub(crate) struct ThresholdGuard {
    previous: usize,
}

#[cfg(test)]
impl ThresholdGuard {
    /// Set the threshold to `new_value` and return a guard that will
    /// restore the previous value on drop.
    pub(crate) fn save(new_value: usize) -> Self {
        let previous = parallel_threshold();
        set_parallel_threshold(new_value);
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for ThresholdGuard {
    fn drop(&mut self) {
        // Bypass the public setter so that restoring to any value (including
        // the original default) never panics.
        PARALLEL_THRESHOLD.store(self.previous, Ordering::Relaxed);
    }
}

// Solver implementation

// =================================================================================================
// Public Re-exports
// =================================================================================================

pub use traits::{
    SimulationResult,
    Solver,
    SolverConfiguration,
    SolverType,
};

pub use boundary::DomainBoundaries;
pub use scenario::Scenario;

pub use methods::{EulerSolver, RK4Solver};


// =================================================================================================
// Helper Functions
// =================================================================================================

use crate::physics::PhysicalState;

/// Validate physical state for numerical issues
///
/// Checks that the state does not contain NaN or Inf values, which would
/// indicate numerical instability or errors in the physics computation.
///
/// # Arguments
///
/// * `state` - Physical state to validate
/// * `step` - Current time step (for error reporting)
///
/// # Returns
///
/// `Ok(())` if state is valid, `Err(msg)` with diagnostic information otherwise
///
/// # Example
///
/// ```rust,ignore
/// validate_state(&state, 42)?;  // Validates state at step 42
/// ```
pub(crate) fn validate_state(state: &PhysicalState, step: usize) -> Result<(), String> {
    // Check each quantity in the state
    for (quantity, data) in &state.quantities {
        // Check for NaN values
        // NaN can arise from 0/0, Inf - Inf, or other undefined operations
        let has_nan = match data {
            crate::physics::PhysicalData::Scalar(x) => x.is_nan(),
            crate::physics::PhysicalData::Vector(v) => v.iter().any(|x| x.is_nan()),
            crate::physics::PhysicalData::Matrix(m) => m.iter().any(|x| x.is_nan()),
            crate::physics::PhysicalData::Array(a) => a.iter().any(|x| x.is_nan()),
        };

        if has_nan {
            return Err(format!(
                "NaN detected in {} at step {}. This indicates numerical instability. \
                 Try reducing time step (increase time_steps parameter).",
                quantity, step
            ));
        }

        // Check for Inf values
        // Inf can indicate overflow or division by zero
        let has_inf = match data {
            crate::physics::PhysicalData::Scalar(x) => x.is_infinite(),
            crate::physics::PhysicalData::Vector(v) => v.iter().any(|x| x.is_infinite()),
            crate::physics::PhysicalData::Matrix(m) => m.iter().any(|x| x.is_infinite()),
            crate::physics::PhysicalData::Array(a) => a.iter().any(|x| x.is_infinite()),
        };

        if has_inf {
            return Err(format!(
                "Infinity detected in {} at step {}. This indicates numerical overflow. \
                 Try reducing time step or check physics model for division by zero.",
                quantity, step
            ));
        }
    }

    Ok(())
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_threshold_value() {
        assert_eq!(DEFAULT_PARALLEL_THRESHOLD, 999);
    }

    #[test]
    fn test_get_and_set_threshold() {
        let _guard = ThresholdGuard::save(500);
        assert_eq!(parallel_threshold(), 500);
    }

    #[test]
    #[should_panic(expected = "parallel threshold must be at least 1")]
    fn test_zero_threshold_panics() {
        set_parallel_threshold(0);
    }

    #[test]
    fn test_threshold_guard_restores_previous_value() {
        let before = parallel_threshold();
        {
            let _guard = ThresholdGuard::save(42);
            assert_eq!(parallel_threshold(), 42);
        }
        // Guard dropped — value must be back to what it was before.
        assert_eq!(parallel_threshold(), before);
    }

    #[test]
    fn test_threshold_is_visible_across_threads() {
        use std::thread;

        let _guard = ThresholdGuard::save(1234);

        let handles: Vec<_> = (0..8)
            .map(|_| thread::spawn(|| parallel_threshold()))
            .collect();

        for handle in handles {
            assert_eq!(handle.join().unwrap(), 1234);
        }
    }
}

