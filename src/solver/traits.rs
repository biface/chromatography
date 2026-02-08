//! Numerical solver traits and types
//!
//! # Design Philosophy
//!
//! This module follows the same pattern as `PhysicalQuantity`:
//! - Central enum `SolverType` defines the type of numerical solution
//! - `SolverConfig` adapts its parameters based on `SolverType`
//! - `SolverResult` adapts its outputs based on `SolverType`
//! - Both have metadata for extensibility
//!
//! # Stability Guarantee
//!
//! - `Solver` trait: STABLE since v0.1.0, will NEVER change
//! - `SolverType` enum: EXTENSIBLE (new variants can be added)
//! - Core structures: STABLE (fields won't be removed)


use std::{collections::HashMap, string::ToString};
use crate::physics::traits::{PhysicalState};
use crate::solver::scenario::Scenario;

// ============================================================================
// Central Solver Type Enumeration (Like PhysicalQuantity)
// ============================================================================

/// Type of numerical solution method
///
/// # Design Pattern
///
/// Similar to `PhysicalQuantity`, this enum is the central abstraction
/// that defines what KIND of numerical solution we're computing.
///
/// Each variant carries the data specific to that solution type.
///
/// # Extensibility
///
/// New variants can be added without breaking existing code.
/// Use `Custom(name, data)` for specialized solution types.
///
/// # Examples
///
/// ```rust
/// # use chrom_rs::solver::SolverType;
/// // Time evolution solution
/// let solver_type = SolverType::TimeEvolution {
///     total_time: 10.0,
///     time_steps: 1000,
/// };
///
/// // Iterative convergence solution
/// let solver_type = SolverType::Iterative {
///     tolerance: 1e-6,
///     max_iterations: 100,
/// };
///
/// // Analytical exact solution
/// let solver_type = SolverType::Analytical {
///     evaluation_time: Some(5.0),
/// };
/// ```

#[derive(Clone, Debug, PartialEq)]
pub enum SolverType {

    /// Time evolution solution (ODE/PDE integration)
    ///
    /// Used by: Euler, Runge-Kutta, Adams-Bashforth, etc.
    ///
    /// # Parameters
    /// - `total_time`: Total simulation time (seconds)
    /// - `time_steps`: Number of time steps
    TimeEvolution {
        total_time: f64,
        time_steps: usize,
    },

    /// Iterative solution to convergence
    ///
    /// Used by: Newton-Raphson, Fixed-Point, GMRES, etc.
    ///
    /// # Parameters
    /// - `tolerance`: Convergence criterion
    /// - `max_iterations`: Safety limit
    Iterative {
        tolerance: f64,
        max_iterations: usize,
    },

    /// Analytical exact solution
    ///
    /// Used by: Closed-form solutions, special cases
    ///
    /// # Parameters
    /// - `evaluation_time`: Optional time at which to evaluate (for time-dependent)
    Analytical {
        evaluation_time: Option<f64>,
    },

    /// Spatial discretization solution (PDE on grid)
    ///
    /// Used by: Finite Difference, Finite Element, Spectral methods
    ///
    /// # Parameters
    /// - `grid_points`: Number of spatial points
    /// - `time_steps`: Number of time steps (if time-dependent)
    SpatialDiscretization {
        grid_points: usize,
        time_steps: Option<usize>,
    },

    /// Custom solver type for specialized needs
    ///
    /// # Example
    /// ```rust
    /// # use chrom_rs::solver::SolverType;
    /// # use std::collections::HashMap;
    /// let mut data = HashMap::new();
    /// data.insert("tolerance".to_string(), 1e-6);
    /// data.insert("initial_dt".to_string(), 0.01);
    ///
    /// let custom = SolverType::Custom(
    ///     "Adaptive-Step-RK45".to_string(),
    ///     data
    /// );
    /// ```
    Custom(String, HashMap<String, f64>),
}

impl SolverType {
    /// Get name identifier
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::solver::SolverType;
    ///
    /// let solver_type = SolverType::TimeEvolution {
    ///     total_time: 10.0,
    ///     time_steps: 1000,
    /// };
    ///
    /// assert_eq!(solver_type.name(), "TimeEvolution");
    /// ```
    pub fn name(&self) -> &str {
        match self {
            SolverType::TimeEvolution { .. } => "TimeEvolution",
            SolverType::Iterative { .. } => "Iterative",
            SolverType::Analytical { .. } => "Analytical",
            SolverType::SpatialDiscretization { .. } => "SpatialDiscretization",
            SolverType::Custom(name, _) => name,
        }
    }

    /// Validate that parameters are physically meaningful
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Time or tolerance values are non-positive
    /// - Step counts are zero
    /// - Custom parameters contain non-finite values
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::solver::SolverType;
    ///
    /// let valid = SolverType::TimeEvolution {
    ///     total_time: 10.0,
    ///     time_steps: 1000,
    /// };
    /// assert!(valid.validate().is_ok());
    ///
    /// let invalid = SolverType::TimeEvolution {
    ///     total_time: -1.0,  // Negative time!
    ///     time_steps: 1000,
    /// };
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        match self {
            SolverType::TimeEvolution { total_time, time_steps } => {
                if *total_time <= 0.0 {
                    return Err("Total time must be positive".to_string());
                }
                if *time_steps == 0 {
                    return Err("TimeSteps must be greater than 0".to_string());
                }
                Ok(())
            }
            SolverType::Iterative {tolerance, max_iterations } => {
                if *tolerance <= 0.0 {
                    return Err("Tolerance must be positive".to_string());
                }
                if *max_iterations == 0 {
                    return Err("Maximum iterations must be positive".to_string());
                }
                Ok(())
            }
            SolverType::Analytical { evaluation_time: _ } => {
                Ok(())
            }
            SolverType::SpatialDiscretization {grid_points, time_steps } => {
                if *grid_points == 0 {
                    return Err("Grid Points cannot be null (no grid)".to_string());
                }
                if let Some(steps) = time_steps
                    && *steps == 0 {
                        return Err("Steps must be greater than 0".to_string());
                    }

                Ok(())
            }
            SolverType::Custom(_, parameters) => {
                for (key, value) in parameters {
                    if !value.is_finite() {
                        return Err(format!("Parameter {} is not finite", key));
                    }
                }
                Ok(())
            }
        }
    }
}

// =================================================================================================
// Solver configuration
// =================================================================================================

/// Configuration for numerical solver
///
/// # Design
///
/// Contains the `SolverType` which defines what kind of solution we want,
/// plus optional metadata for additional context.
///
/// # Examples
///
/// ```rust
/// # use chrom_rs::solver::{SolverConfiguration, SolverType};
/// # use std::collections::HashMap;
/// // Time evolution config
/// let config = SolverConfiguration::new(
///     SolverType::TimeEvolution {
///         total_time: 10.0,
///         time_steps: 1000,
///     }
/// );
///
/// // Iterative convergence config
/// let config = SolverConfiguration::new(
///     SolverType::Iterative {
///         tolerance: 1e-6,
///         max_iterations: 100,
///     }
/// );
/// ```
#[derive(Clone, Debug)]
pub struct SolverConfiguration {
    /// Type of solver and its paraméters
    pub solver_type: SolverType,

}

impl SolverConfiguration {
    /// Create a new configuration with a given solver type
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::solver::{SolverConfiguration, SolverType};
    ///
    /// let config = SolverConfiguration::new(
    ///     SolverType::TimeEvolution {
    ///         total_time: 10.0,
    ///         time_steps: 1000,
    ///     }
    /// );
    /// ```
    pub fn new(solver_type: SolverType) -> Self {
        Self { solver_type }
    }

    /// Create a time evolution configuration
    ///
    /// # Arguments
    /// * `total_time` - Total simulation time (e.g., 600.0 seconds)
    /// * `time_steps` - Number of time steps (e.g., 10000)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::solver::SolverConfiguration;
    ///
    /// let config = SolverConfiguration::time_evolution(600.0, 10000);
    /// assert!(config.validate().is_ok());
    /// ```
    pub fn time_evolution(total_time: f64, time_steps: usize) -> Self {
        Self::new(SolverType::TimeEvolution { total_time, time_steps })
    }

    /// Create an iterative solver configuration
    ///
    /// # Arguments
    /// * `tolerance` - Convergence tolerance (e.g., 1e-6)
    /// * `max_iterations` - Maximum number of iterations (e.g., 100)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::solver::SolverConfiguration;
    ///
    /// let config = SolverConfiguration::iterative(1e-6, 100);
    /// ```
    pub fn iterative(tolerance: f64, max_iterations: usize) -> Self {
        Self::new(SolverType::Iterative {tolerance, max_iterations})
    }
    
    /// Create an analytical solver configuration
    ///
    /// # Arguments
    /// * `evaluation_time` - Time at which to evaluate the solution
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::solver::SolverConfiguration;
    ///
    /// let config = SolverConfiguration::analytical(5.0);
    /// ```
    pub fn analytical(evaluation_time: f64) -> Self {
        Self::new(SolverType::Analytical { evaluation_time: Some(evaluation_time) })
    }

    /// Create a spatial discretization solver configuration
    ///
    /// # Arguments
    /// * `grid_points` - Number of spatial grid points
    /// * `time_steps` - Number of time steps (for time-dependent problems)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::solver::SolverConfiguration;
    ///
    /// let config = SolverConfiguration::spatial_discretization(100, 1000);
    /// ```
    pub fn spatial_discretization(grid_points: usize, time_steps: usize) -> Self {
        Self::new(SolverType::SpatialDiscretization {grid_points, time_steps: Some(time_steps) })
    }

    /// Validate configuration
    ///
    /// Delegates to `SolverType::validate()`
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::solver::SolverConfiguration;
    ///
    /// let config = SolverConfiguration::time_evolution(10.0, 1000);
    /// assert!(config.validate().is_ok());
    ///
    /// let bad_config = SolverConfiguration::time_evolution(-1.0, 1000);
    /// assert!(bad_config.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        self.solver_type.validate()
    }
}

// ============================================================================
// Simulation Result
// ============================================================================

/// Result of a numerical simulation
///
/// # Design
///
/// Contains the solution trajectory and metadata about the simulation.
/// The structure adapts to different solver types:
/// - Time evolution: Full trajectory over time
/// - Iterative: Convergence history
/// - Analytical: Single evaluation
///
/// # Examples
///
/// ```rust
/// # use chrom_rs::solver::SimulationResult;
/// # use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
/// # use nalgebra::DVector;
/// # let final_state = PhysicalState::new(
/// #     PhysicalQuantity::Concentration,
/// #     PhysicalData::Vector(DVector::from_vec(vec![1.0, 2.0, 3.0]))
/// # );
/// // Time evolution result
/// let result = SimulationResult::new(
///     vec![0.0, 0.1, 0.2],  // time points
///     vec![final_state.clone(), final_state.clone(), final_state.clone()],    // state trajectory
///     final_state,
/// );
/// ```
#[derive(Clone, Debug)]
pub struct SimulationResult {
    // Time points at which the solution was computed
    ///
    /// For time-dependent problems, this is the time grid.
    /// For steady-state problems, this may be empty or contain iteration numbers.
    pub time_points: Vec<f64>,

    /// State trajectory
    ///
    /// Sequence of physical states over time (or iterations).
    /// For analytical solutions, this may contain only the final state.
    pub state_trajectory: Vec<PhysicalState>,

    /// Final converged state
    pub final_state: PhysicalState,

    /// Metadata about the simulation
    ///
    /// Can contain:
    /// - "iterations": Number of iterations performed
    /// - "convergence_error": Final convergence error
    /// - "cpu_time": Computation time in seconds
    /// - etc.
    pub metadata: HashMap<String, String>,
}

impl SimulationResult {
    /// Create a new simulation result
    ///
    /// # Arguments
    /// * `time_points` - Time points or iteration indices
    /// * `state_trajectory` - Sequence of states
    /// * `final_state` - Final converged state
    ///
    /// # Example
    /// ```rust
    /// # use chrom_rs::solver::SimulationResult;
    /// # use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    /// # use nalgebra::DVector;
    ///
    /// let final_state = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::Vector(DVector::from_vec(vec![1.0, 2.0, 3.0]))
    /// );
    ///
    /// let result = SimulationResult::new(
    ///     vec![0.0, 1.0, 2.0],
    ///     vec![],  // Empty trajectory for simplicity
    ///     final_state,
    /// );
    /// ```
    pub fn new(
        time_points: Vec<f64>,
        state_trajectory: Vec<PhysicalState>,
        final_state: PhysicalState,
    ) -> Self {
        Self {
            time_points,
            state_trajectory,
            final_state,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the result
    ///
    /// # Example
    /// ```rust
    /// # use chrom_rs::solver::SimulationResult;
    /// # use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    /// # use nalgebra::DVector;
    /// # let final_state = PhysicalState::new(
    /// #     PhysicalQuantity::Concentration,
    /// #     PhysicalData::Vector(DVector::zeros(3))
    /// # );
    /// let mut result = SimulationResult::new(vec![], vec![], final_state);
    ///
    /// result.add_metadata("iterations", "42");
    /// result.add_metadata("cpu_time", "1.23");
    /// ```
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get number of time points
    pub fn len(&self) -> usize {
        self.time_points.len()
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.time_points.is_empty()
    }
}

// ============================================================================
// Solver Trait (STABLE - Never Changes)
// ============================================================================

/// Trait for numerical solvers
///
/// # Design Philosophy
///
/// This trait is the **core stable interface** for all numerical solvers.
/// It will NEVER change to ensure backward compatibility.
///
/// A solver:
/// 1. Takes a `Scenario` (model + boundaries)
/// 2. Takes a `SolverConfiguration` (how to solve)
/// 3. Returns a `SimulationResult` (the solution)
///
/// # Separation of Concerns
///
/// - **Scenario**: WHAT to solve (physics + boundaries)
/// - **SolverConfiguration**: HOW to solve (method + parameters)
/// - **Solver**: The numerical method implementation
///
/// # Stability Guarantee
///
/// This trait signature will NEVER change. New functionality will be added
/// through:
/// - New variants in `SolverType`
/// - New fields in `SolverConfiguration` (with defaults)
/// - Metadata in `SimulationResult`
///
/// # Examples
///
/// ```rust,ignore
/// use chrom_rs::solver::{Solver, SolverConfiguration};
/// // use chrom_rs::solver::euler::EulerSolver;
///
/// // Create solver
/// // let solver = EulerSolver;
///
/// // Create configuration
/// let config = SolverConfiguration::time_evolution(600.0, 10000);
///
/// // Solve (scenario must be created first)
/// // let result = solver.solve(&scenario, &config)?;
/// ```
pub trait Solver {
    /// Solve the scenario with the given configuration
    ///
    /// # Arguments
    /// * `scenario` - The scenario to solve (model + boundaries)
    /// * `config` - How to solve it (method + parameters)
    ///
    /// # Returns
    ///
    /// The simulation result containing the solution trajectory and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration is invalid
    /// - Scenario is invalid
    /// - Numerical method fails to converge
    /// - Computation encounters numerical issues
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, String>;

    /// Get the name of this solver
    ///
    /// Used for logging and debugging.
    ///
    /// # Example
    /// ```rust,ignore
    /// use chrom_rs::solver::Solver;
    /// // use chrom_rs::solver::euler::EulerSolver;
    ///
    /// // let solver = EulerSolver;
    /// // assert_eq!(solver.name(), "Forward Euler");
    /// ```
    fn name(&self) -> &str;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== SolverType Tests ====================

    #[test]
    fn test_solver_type_name() {
        let time_ev = SolverType::TimeEvolution {
            total_time: 10.0,
            time_steps: 100,
        };
        assert_eq!(time_ev.name(), "TimeEvolution");

        let iterative = SolverType::Iterative {
            tolerance: 1e-6,
            max_iterations: 100,
        };
        assert_eq!(iterative.name(), "Iterative");
    }

    #[test]
    fn test_solver_type_validate_time_evolution() {
        // Valid
        let valid = SolverType::TimeEvolution {
            total_time: 10.0,
            time_steps: 1000,
        };
        assert!(valid.validate().is_ok());

        // Invalid: negative time
        let invalid_time = SolverType::TimeEvolution {
            total_time: -1.0,
            time_steps: 1000,
        };
        assert!(invalid_time.validate().is_err());

        // Invalid: zero steps
        let invalid_steps = SolverType::TimeEvolution {
            total_time: 10.0,
            time_steps: 0,
        };
        assert!(invalid_steps.validate().is_err());
    }

    #[test]
    fn test_solver_type_validate_iterative() {
        // Valid
        let valid = SolverType::Iterative {
            tolerance: 1e-6,
            max_iterations: 100,
        };
        assert!(valid.validate().is_ok());

        // Invalid: negative tolerance
        let invalid_tol = SolverType::Iterative {
            tolerance: -1e-6,
            max_iterations: 100,
        };
        assert!(invalid_tol.validate().is_err());
    }

    // ==================== SolverConfiguration Tests ====================

    #[test]
    fn test_solver_configuration_factory_methods() {
        let time_ev = SolverConfiguration::time_evolution(10.0, 1000);
        assert!(matches!(time_ev.solver_type, SolverType::TimeEvolution { .. }));

        let iterative = SolverConfiguration::iterative(1e-6, 100);
        assert!(matches!(iterative.solver_type, SolverType::Iterative { .. }));

        let analytical = SolverConfiguration::analytical(5.0);
        assert!(matches!(analytical.solver_type, SolverType::Analytical { .. }));
    }

    #[test]
    fn test_solver_configuration_validate() {
        let valid = SolverConfiguration::time_evolution(10.0, 1000);
        assert!(valid.validate().is_ok());

        let invalid = SolverConfiguration::time_evolution(-1.0, 1000);
        assert!(invalid.validate().is_err());
    }

    // ==================== SimulationResult Tests ====================

    //#[test]
    //fn test_simulation_result_creation() {
    //    use crate::physics::{PhysicalState, PhysicalQuantity};
    //    use nalgebra::DVector;

    //    let final_state = PhysicalState::new(
    //        PhysicalQuantity::Concentration,
    //        DVector::from_vec(vec![1.0, 2.0, 3.0]),
    //    );

    //    let result = SimulationResult::new(
    //        vec![0.0, 1.0, 2.0],
    //        vec![],
    //        final_state,
    //    );

    //    assert_eq!(result.len(), 3);
    //    assert!(!result.is_empty());
    //}

    //#[test]
    //fn test_simulation_result_metadata() {
    //    use crate::physics::{PhysicalState, PhysicalQuantity};
    //    use nalgebra::DVector;

    //    let final_state = PhysicalState::new(
    //        PhysicalQuantity::Concentration,
    //        DVector::zeros(1),
    //    );

    //    let mut result = SimulationResult::new(vec![], vec![], final_state);

    //    result.add_metadata("iterations", "42");
    //    result.add_metadata("cpu_time", "1.23");

    //    assert_eq!(result.metadata.get("iterations"), Some(&"42".to_string()));
    //    assert_eq!(result.metadata.get("cpu_time"), Some(&"1.23".to_string()));
    //}
}

