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

#[derive(Clone, Debug)]
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
    /// SolverType::Custom(
    ///     "Adaptive-Step-RK45".to_string(),
    ///     vec![
    ///         ("tolerance".to_string(), 1e-6),
    ///         ("initial_dt".to_string(), 0.01),
    ///     ].into_iter().collect()
    /// )
    /// ```
    Custom(String, HashMap<String, f64>),
}

impl SolverType {
    /// Get name identifier
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
/// // Time evolution config
/// let config = SolverConfig {
///     solver_type: SolverType::TimeEvolution {
///         total_time: 10.0,
///         time_steps: 1000,
///     },
///     metadata: HashMap::new(),
/// };
///
/// // With metadata
/// let mut config = SolverConfig {
///     solver_type: SolverType::Iterative {
///         tolerance: 1e-6,
///         max_iterations: 100,
///     },
///     metadata: HashMap::new(),
/// };
/// config.metadata.insert("relaxation_factor".to_string(), "0.8".to_string());
/// ```
#[derive(Clone, Debug)]
pub struct SolverConfiguration {
    /// Type of solver and its paraméters
    pub solver_type: SolverType,

}

impl SolverConfiguration {
    /// Create a new configuration with a given solver type
    pub fn new(solver_type: SolverType) -> Self {
        Self { solver_type }
    }

    /// Create a time resolution configuration
    pub fn time_evolution(total_time: f64, time_steps: usize) -> Self {
        Self::new(SolverType::TimeEvolution { total_time, time_steps })
    }

    /// Create an iterative solver configuration
    pub fn iterative(tolerance: f64, max_iterations: usize) -> Self {
        Self::new(SolverType::Iterative {tolerance, max_iterations})
    }
    
    /// Create an analytical solver configuration
    pub fn analytical(evaluation_time: f64) -> Self {
        Self::new(SolverType::Analytical { evaluation_time: Some(evaluation_time) })
    }

    /// Create a spatial discretisation solver configuration
    pub fn spatial_discretization(grid_points: usize, time_steps: usize) -> Self {
        Self::new(SolverType::SpatialDiscretization {grid_points, time_steps: Some(time_steps) })
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        self.solver_type.validate()
    }
}

