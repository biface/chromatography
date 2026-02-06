//! Forward Euler numerical solver
//!
//! # Mathematical Background
//!
//! The Forward Euler method is the simplest explicit time-stepping scheme
//! for solving ordinary differential equations (ODEs):
//!
//! ```text
//! dy/dt = f(y, t)
//! ```
//!
//! The scheme approximates the solution at time t_{n+1} = t_n + dt using:
//!
//! ```text
//! y_{n+1} = y_n + dt * f(y_n, t_n)
//! ```
//!
//! # Characteristics
//!
//! - **Order**: First-order accurate (error ~ O(dt))
//! - **Stability**: Conditionally stable (requires small time steps)
//! - **Complexity**: 1 function evaluation per step
//! - **Memory**: O(1) - only stores current state
//!
//! # Advantages
//!
//! ✅ Simplest possible time integrator
//! ✅ Easy to understand and implement
//! ✅ Low computational cost per step
//! ✅ Good for educational purposes
//!
//! # Limitations
//!
//! ⚠️ First-order accuracy only (needs small dt for precision)
//! ⚠️ Stability restrictions (especially for stiff problems)
//! ⚠️ Not suitable for production chromatography simulations
//!
//! # When to Use
//!
//! - Prototyping and testing architecture
//! - Educational demonstrations
//! - Quick exploratory simulations
//! - Non-stiff problems with relaxed accuracy requirements
//!
//! # When NOT to Use
//!
//! - Production chromatography simulations → Use RK4 or higher-order methods
//! - Stiff problems → Use implicit methods
//! - High-accuracy requirements → Use RK4, RK45, or adaptive methods
//!
//! # Example
//!
//! ```rust,ignore
//! use chrom_rs::solver::{EulerSolver, Solver, SolverConfiguration, Scenario};
//!
//! let solver = EulerSolver;
//! let config = SolverConfiguration::time_evolution(600.0, 10000);
//!
//! // scenario must be created with model + boundaries
//! let result = solver.solve(&scenario, &config)?;
//! ```

use crate::physics::PhysicalState;
use crate::solver;
use crate::solver::{Scenario, SimulationResult, Solver, SolverConfiguration, SolverType};

// =================================================================================================
// Forward Euler Solver
// =================================================================================================

/// Forward Euler time-stepping solver
///
/// Implements the simplest explicit time integration scheme:
/// y_{n+1} = y_n + dt * f(y_n)
///
/// # Algorithm
///
/// For ODE system dy/dt = f(y):
///
/// 1. Start with initial state y_0
/// 2. For each time step n = 0, 1, 2, ..., N-1:
///    - Compute physics: k = f(y_n)
///    - Update state: y_{n+1} = y_n + dt * k
///    - Store trajectory point
/// 3. Return complete trajectory
///
/// # Stability
///
/// The method is **conditionally stable**. For linear problems dy/dt = λy,
/// the stability condition is:
///
/// ```text
/// |1 + λ * dt| ≤ 1
/// ```
///
/// For chromatography models, this typically requires dt to be small enough
/// that the Courant-Friedrichs-Lewy (CFL) condition is satisfied.
///
/// # Error Analysis
///
/// - **Local truncation error**: O(dt²) per step
/// - **Global error**: O(dt) after T/dt steps
/// - **Convergence**: First-order convergence when refining dt
///
/// # Example
///
/// ```rust,ignore
/// use chrom_rs::solver::{EulerSolver, Solver};
///
/// let solver = EulerSolver;
///
/// // Solver requires TimeEvolution configuration
/// let config = SolverConfiguration::time_evolution(
///     600.0,    // Total time (seconds)
///     10000,    // Number of time steps
/// );
///
/// let result = solver.solve(&scenario, &config)?;
/// println!("Final time: {}", result.time_points.last().unwrap());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EulerSolver;

impl EulerSolver {
    /// Create a new Forward Euler solver
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::solver::EulerSolver;
    /// use chrom_rs::solver::traits::Solver;
    ///
    /// let solver = EulerSolver::new();
    /// assert_eq!(solver.name(), "Forward Euler");
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Solver for EulerSolver {

    fn solve(&self, scenario: &Scenario, config: &SolverConfiguration) -> Result<SimulationResult, String> {

        // ====== Step 1: Validation ======

        // Validate configuration parameters
        config.validate()? ;

        // Validate scenario (model and boundaries)

        scenario.validate()?;

        // Forward Euler is dedicated to time evolution

        let (total_time, time_steps) = match &config.solver_type {
            SolverType::TimeEvolution { total_time, time_steps } => {
                (*total_time, *time_steps)
            }
            other   => {
                return Err(format!(
                    "EulerSolver only supports TimeEvolution configuration, got {}",
                    other.name()
                ));
            }
        };

        // ====== Step 2: Setup ======

        // Compute time step size
        // dt = T / N where T is total time and N is number of steps
        let dt = total_time / (time_steps as f64);

        // Get initial state from scenario boundaries

        let mut state = match scenario.conditions.initial_condition() {
            Some(initial_state) => initial_state.clone(),
            None => return Err("No initial condition found in domain boundaries".to_string()),
        };

        // preallocate storage for trajectory
        // Reserve exact capacity to avoid reallocation during integration
        let mut time_points = Vec::with_capacity(time_steps + 1);
        let mut state_trajectory = Vec::with_capacity(time_steps + 1);

        // Store initial condition

        time_points.push(0.0);
        state_trajectory.push(state.clone());

        // ====== Step 3: Time Integration ======

        // Forward Euler time-stepping loop
        // For each step n = 0, 1, ..., time_steps - 1:
        //   1. Compute f(y_n, t_n) using the physical model
        //   2. Update: y_{n+1} = y_n + dt * f(y_n, t_n)
        //   3. Store state and time point
        //   4. Validate state (check for NaN, physical bounds, etc.)

        for step in 0..time_steps {
            // Current time
            let t = dt * step as f64;

            // ====== Euler step ======

            // 1. Compute physics: f(y_n, t_n)
            //    This returns the right-hand side of dy/dt = f(y, t)
            //    For chromatography: transport + dispersion + adsorption terms
            let physics: PhysicalState = scenario.model.compute_physics(&state);

            // 2. Update state: y_{n+1} = y_n + dt * f(y_n)
            //    PhysicalState implements Add and Mul<f64>, so this works naturally
            //    The clone() is necessary because Add consumes self
            state = state.clone() + physics * dt;

            // ====== Storage ======

            // Store new state in trajectory
            // This creates a full history of the simulation for later analysis

            state_trajectory.push(state.clone());

            // Store time point: t_{n+1} = (step + 1) * dt
            // IMPORTANT: Calculate directly from index to avoid accumulation of
            // floating-point rounding errors. This ensures that the final time point
            // is exactly total_time (within machine epsilon).
            //
            // Why not t += dt? Because 0.1 is not exactly representable in binary:
            //   0.1 ≈ 0.1000000000000000055511151231...
            // After 100 additions: error accumulates to ~1e-14
            // Direct calculation: (100 * 0.1) has much smaller error
            time_points.push((step as f64 + 1.0) * dt);

            // ====== Validation ======

            // Check for numerical issues (NaN, Inf, etc.)
            // This helps catch problems early rather than propagating them
            solver::validate_state(&state, step + 1)?;

        }

        // ====== Step 4: Build Result ======

        // Extract final state (as the last element of trajectory)
        let final_state: PhysicalState = state;

        // Create a simulation result

        let mut result = SimulationResult::new(
            time_points,
            state_trajectory,
            final_state
        );

        // Add metadata for diagnostics and reproducibility

        result.add_metadata("solver", "Forward Euler");
        result.add_metadata("time steps", &time_steps.to_string());
        result.add_metadata("dt", &dt.to_string());
        result.add_metadata("total time", &total_time.to_string());

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "Forward Euler"
    }
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
    use crate::solver::boundary::DomainBoundaries;

    // ====== Mock Models for Testing ======

    /// Mock model: exponential decay dy/dt = -k * y
    ///
    /// Analytical solution: y(t) = y_0 * exp(-k * t)
    ///
    /// This is used to test numerical accuracy since we know the exact solution.
    struct ExponentialDecay {
        points: usize,
        decay_rate: f64, // k in dy/dt = -k*y
    }

    impl PhysicalModel for ExponentialDecay {
        fn points(&self) -> usize {
            self.points
        }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            // dy/dt = -k * y
            // return -k * y as the physics

            let mut result = state.clone();

            if let Some(conc) =
                result.get_mut(PhysicalQuantity::Concentration) {
                conc.apply(|y| - self.decay_rate * y) ;
            }

            result
        }

        fn setup_initial_state(&self) -> PhysicalState {
            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::uniform_vector(self.points, 1.0)
            )
        }

        fn name(&self) -> &'static str {
            "Exponential Decay"
        }
    }

    /// Mock model: constant growth dy/dt = c
    ///
    /// Analytical solution: y(t) = y_0 + c * t
    struct ConstantGrowth {
        points: usize,
        growth_rate: f64,
    }

    impl PhysicalModel for ConstantGrowth {
        fn points(&self) -> usize {
            self.points
        }

        fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {
            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::uniform_vector(self.points, self.growth_rate)
            )
        }

        fn setup_initial_state(&self) -> PhysicalState {
            PhysicalState::new(
                PhysicalQuantity::Concentration,
                PhysicalData::uniform_vector(self.points, 0.0)
            )
        }

        fn name(&self) -> &'static str {
            "Constant Growth"
        }
    }

    // ====== Solver Creation Tests ======

    #[test]
    fn test_euler_solver_creation() {
        let solver = EulerSolver::new();
        assert_eq!(solver.name(), "Forward Euler");
    }

    #[test]
    fn test_euler_solver_default() {
        let solver = EulerSolver::default();
        assert_eq!(solver.name(), "Forward Euler");
    }

    // ====== Configuration Tests ======

    #[test]
    fn test_euler_accepts_time_evolution() {
        let solver = EulerSolver::new();
        let config = SolverConfiguration::time_evolution(10.0, 100);

        // Create simple scenario
        let model = Box::new(ConstantGrowth {
            points: 10,
            growth_rate: 1.0
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        // let's rock

        let result = solver.solve(&scenario, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_euler_solver_iterative_failed() {
        let solver = EulerSolver::new();
        let config = SolverConfiguration::iterative(1e-6, 100);

        // Create scenario
        let model = Box::new(ConstantGrowth {
            points: 10,
            growth_rate: 1.0
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        // Should fail with message on TimeEvolution support
        let result = solver.solve(&scenario, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("only supports TimeEvolution"));
    }

    #[test]
    fn test_euler_solver_analytical_failed() {
        let solver = EulerSolver::new();
        let config = SolverConfiguration::analytical(0.5);

        // Create scenario
        let model = Box::new(ConstantGrowth {
            points: 10,
            growth_rate: 1.0
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let result = solver.solve(&scenario, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("only supports TimeEvolution"));
    }

    // ====== Numerical Accuracy Tests ======

    #[test]
    fn test_euler_constant_growth() {
        // dy/dt = c → y(t) = y_0 + c*t
        // Euler should be exact in that case

        let solver = EulerSolver::new();
        let growth_rate = 2.0;

        let model = Box::new(ConstantGrowth {
            points: 5,
            growth_rate
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let total_time = 10.0;
        let config = SolverConfiguration::time_evolution(total_time, 100);
        let result = solver.solve(&scenario, &config).unwrap();

        // Check final time (should be close to 10.0)
        println!("{:?}", (10.0 - result.time_points.last().unwrap()));
        assert!((result.time_points.last().unwrap() - total_time).abs() < 1e-10);

        // Check final value as y(10) = 2.0 * 10.0 + 0.0
        // y(10) = 20.0
        let final_state = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();

        let expected_concentration = growth_rate * total_time ;
        let actual_concentration = final_state.as_vector()[0];

        // Check result should be exact

        assert!((actual_concentration - expected_concentration).abs() < 1e-10);
    }

    #[test]
    fn test_euler_exponential_decay() {
        // dy/dt = -k*y → y(t) = y_0 * exp(-k*t)
        // Euler has first-order error for this

        let solver = EulerSolver::new();
        let decay_rate = 0.1;

        let model = Box::new(ExponentialDecay {
            points: 5,
            decay_rate
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let total_time = 10.0;
        let time_step = 1000 ;
        let config = SolverConfiguration::time_evolution(total_time, time_step);

        let result = solver.solve(&scenario, &config).unwrap();

        let expected = (- decay_rate * total_time).exp();
        let final_concentration = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();
        let actual_concentration = final_concentration.as_vector()[0];

        // Euler has O(dt) error, with dt = 10/1000 = 0.01
        // Error should be ~ 0.01 * something
        let error = (actual_concentration - expected).abs();
        assert!(error < 0.01, "Error {} too large for dt=0.01", error);
    }

    #[test]
    fn test_euler_convergence() {
        // Tests that error decrease linearly with dt (first-order convergence)

        let solver = EulerSolver::new();
        let decay_rate = 0.5;
        let total_time = 5.0;

        let model = Box::new(ExponentialDecay {
            points: 3,
            decay_rate
        });

        let exact = (-decay_rate * total_time).exp();

        // Running tests with different timesteps

        let vsteps: Vec<usize> = vec![100, 200, 400, 800];
        let mut verrors : Vec<f64> = Vec::new();

        for &steps in &vsteps {
            let initial = model.setup_initial_state();
            let boundaries = DomainBoundaries::temporal(initial);
            let scenario = Scenario::new(
                Box::new(ExponentialDecay {
                    points: 3,
                    decay_rate
                }),
                boundaries
            );

            let config = SolverConfiguration::time_evolution(total_time, steps);
            let result = solver.solve(&scenario, &config).unwrap();

            let final_state = result.final_state
                .get(PhysicalQuantity::Concentration)
                .unwrap();
            let actual = final_state.as_vector()[0];
            let error = (actual - exact).abs();
            verrors.push(error);
        }

        // Check first-order convergence: error(dt/2) ≈ error(dt) / 2

        for i in 0..verrors.len() - 1 {
            let ratio = verrors[i] / verrors[i + 1];

            assert!(ratio > 1.8 && ratio < 2.2,
            "Convergence ration {} not a first order at step {}", ratio, i);
        }
    }

    // ====== Trajectory tests ======

    #[test]
    fn test_euler_trajectory_length() {

        let solver = EulerSolver::new();
        let model = Box::new(ConstantGrowth {
            points: 5,
            growth_rate: 1.0,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let time_steps = 100 ;
        let config = SolverConfiguration::time_evolution(10.0, time_steps);

        let result = solver.solve(&scenario, &config).unwrap();

        // Should have 101 records (time steps and initial condition)

        assert_eq!(result.time_points.len(), time_steps + 1);
        assert_eq!(result.state_trajectory.len(), time_steps + 1);
    }

    #[test]
    fn test_euler_time_points() {

        let solver = EulerSolver::new();
        let model = Box::new(ConstantGrowth {
            points:5,
            growth_rate: 1.0,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let total_time = 20.0;
        let time_steps = 100 ;
        let dt = total_time / (time_steps as f64);

        let config = SolverConfiguration::time_evolution(total_time, time_steps);

        let result = solver.solve(&scenario, &config).unwrap();

        // Check first point is initial as 0

        assert!((result.time_points[0] - 0.0).abs() <= 1e-10);

        // Check last point is equal to total time

        let final_time = *result.time_points.last().unwrap();

        assert!(
            (final_time - total_time).abs() <= 1e-14,
            "Final time {} should be very close to {}. Difference {:e}",
            final_time,
            total_time,
            (final_time - total_time).abs()
        );

        // Check a uniform spacing between calculation points even if some small rounding is expected

        for i in 1..result.state_trajectory.len() {

            let spacing = result.time_points[i] - result.time_points[i - 1];
            assert!(
                (spacing - dt).abs() <= 1e-12,
                "Time step {} differs from mathematical dt {} by more than 1e-12",
                spacing,
                dt
            );
        }
    }

    #[test]
    fn test_euler_time_precision() {
        // Specific test for floating-point precision in time points
        // This test would fail with accumulation (t += dt)

        let solver = EulerSolver::new();

        let model = Box::new(ConstantGrowth {
            points: 3,
            growth_rate: 1.0,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        // Use a total_time that's not exactly representable in binary
        // and a number of steps that will magnify rounding errors
        let total_time = 10.0;
        let time_steps = 100;
        let config = SolverConfiguration::time_evolution(total_time, time_steps);

        let result = solver.solve(&scenario, &config).unwrap();

        // With direct calculation: (step + 1) * dt
        // Final time should be exactly total_time within machine epsilon (~2e-16)
        let final_time = *result.time_points.last().unwrap();

        // Machine epsilon for f64 is ~2.22e-16
        // With direct calculation, error should be O(epsilon)
        // With accumulation, error would be O(n * epsilon) ≈ 100 * 2e-16 = 2e-14
        assert!(
            (final_time - total_time).abs() < 1e-14,
            "Direct calculation maintains precision: {} ≈ {} (error: {:e})",
            final_time, total_time, (final_time - total_time).abs()
        );
    }

    // ====== Metadata Tests ======

    #[test]
    fn test_euler_metadata() {
        let solver = EulerSolver::new();

        let model = Box::new(ConstantGrowth {
            points: 5,
            growth_rate: 1.0,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let total_time = 100.0;
        let time_steps = 500;
        let config = SolverConfiguration::time_evolution(total_time, time_steps);

        let result = solver.solve(&scenario, &config).unwrap();

        // Check metadata
        assert_eq!(result.metadata.get("solver"), Some(&"Forward Euler".to_string()));
        assert_eq!(result.metadata.get("time steps"), Some(&"500".to_string()));
        assert_eq!(result.metadata.get("total time"), Some(&"100".to_string()));

        // dt = 100 / 500 = 0.2
        let dt_str = result.metadata.get("dt").unwrap();
        let dt: f64 = dt_str.parse().unwrap();
        assert!((dt - 0.2).abs() < 1e-10);
    }

    // ====== Validation Tests ======

    #[test]
    fn test_euler_detects_nan() {
        // Create a model that produces NaN
        struct NaNModel {
            points: usize,
        }

        impl PhysicalModel for NaNModel {
            fn points(&self) -> usize {
                self.points
            }

            fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {
                // Return NaN to trigger validation error
                PhysicalState::new(
                    PhysicalQuantity::Concentration,
                    PhysicalData::uniform_vector(self.points, f64::NAN),
                )
            }

            fn setup_initial_state(&self) -> PhysicalState {
                PhysicalState::new(
                    PhysicalQuantity::Concentration,
                    PhysicalData::uniform_vector(self.points, 1.0),
                )
            }

            fn name(&self) -> &str {
                "NaN Model"
            }
        }

        let solver = EulerSolver::new();
        let model = Box::new(NaNModel { points: 5 });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(10.0, 10);
        let result = solver.solve(&scenario, &config);

        // Should fail with NaN detection
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("NaN"));
    }

    #[test]
    fn test_euler_detects_inf() {
        // Create a model that produces Inf
        struct InfModel {
            points: usize,
        }

        impl PhysicalModel for InfModel {
            fn points(&self) -> usize {
                self.points
            }

            fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {
                // Return Inf to trigger validation error
                PhysicalState::new(
                    PhysicalQuantity::Concentration,
                    PhysicalData::uniform_vector(self.points, f64::INFINITY),
                )
            }

            fn setup_initial_state(&self) -> PhysicalState {
                PhysicalState::new(
                    PhysicalQuantity::Concentration,
                    PhysicalData::uniform_vector(self.points, 1.0),
                )
            }

            fn name(&self) -> &str {
                "Inf Model"
            }
        }

        let solver = EulerSolver::new();
        let model = Box::new(InfModel { points: 5 });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(10.0, 10);
        let result = solver.solve(&scenario, &config);

        // Should fail with Inf detection
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Infinity"));
    }

    // ====== Edge cases ======

    #[test]
    fn test_euler_single_step() {

        let solver = EulerSolver::new();
        let model = Box::new(ConstantGrowth {
            points:3,
            growth_rate: 5.0
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        // Single time step
        let config = SolverConfiguration::time_evolution(1.0, 1);
        let result = solver.solve(&scenario, &config).unwrap();

        // Should have two points initial and final
        assert_eq!(result.time_points.len(), 2);
        assert_eq!(result.state_trajectory.len(), 2);

        // y(1) = 0 + 5 * 1

        let final_state = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();

        assert!((final_state.as_vector()[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euler_multiple_quantity() {

        // Simulation with two different PhysicalQuantity

        struct MultiQuantityModel {
            points: usize,
        }

        impl PhysicalModel for MultiQuantityModel {
            fn points(&self) -> usize { self.points }

            fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {

                let mut result = PhysicalState::empty();

                result.set(
                    PhysicalQuantity::Concentration,
                    PhysicalData::uniform_vector(self.points, 1.0),
                );

                result.set(
                    PhysicalQuantity::Temperature,
                    PhysicalData::uniform_vector(self.points, 0.1)
                );

                result
            }

            fn setup_initial_state(&self) -> PhysicalState {
                let mut state = PhysicalState::empty();
                state.set(
                    PhysicalQuantity::Concentration,
                    PhysicalData::uniform_vector(self.points, 0.0),
                );
                state.set(
                    PhysicalQuantity::Temperature,
                    PhysicalData::uniform_vector(self.points, -273.15),
                );
                state
            }

            fn name(&self) -> &str {
                "Multiple Physical Quantity Model"
            }
        }

        let solver = EulerSolver::new();
        let model = Box::new(MultiQuantityModel {points: 5});
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(10.0, 100);
        let result = solver.solve(&scenario, &config).unwrap();

        // Check concentration as y(10) = 0 + 10*1 = 10

        let final_concentration = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();

        assert!((final_concentration.as_vector()[0] - 10.0).abs() < 1e-10);

        // Check temperature as t(10) = -273.15 + 10*0.1 = -272.15

        let final_temperature = result.final_state
            .get(PhysicalQuantity::Temperature)
            .unwrap();

        assert!((final_temperature.as_vector()[0] + 272.15).abs() < 1e-10); // - (- 272.15)
    }
}

