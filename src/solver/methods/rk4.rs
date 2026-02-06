//! Runge-Kutta 4 (RK4) numerical solver
//!
//! # Mathematical Background
//!
//! The classical fourth-order Runge-Kutta method (RK4) is one of the most
//! widely used numerical integrators for ordinary differential equations:
//!
//! ```text
//! dy/dt = f(y, t)
//! ```
//!
//! The RK4 scheme uses a weighted average of four slope estimates:
//!
//! ```text
//! k₁ = f(yₙ, tₙ)
//! k₂ = f(yₙ + dt/2 * k₁, tₙ + dt/2)
//! k₃ = f(yₙ + dt/2 * k₂, tₙ + dt/2)
//! k₄ = f(yₙ + dt * k₃, tₙ + dt)
//!
//! yₙ₊₁ = yₙ + dt/6 * (k₁ + 2k₂ + 2k₃ + k₄)
//! ```
//!
//! # Characteristics
//!
//! - **Order**: Fourth-order accurate (error ~ O(dt⁴) per step)
//! - **Stability**: Better stability than Euler, suitable for moderate stiffness
//! - **Complexity**: 4 function evaluations per step
//! - **Memory**: O(1) - stores only k₁, k₂, k₃, k₄ intermediates
//!
//! # Advantages
//!
//! ✅ Fourth-order accuracy (much better than Euler's first-order)
//! ✅ Excellent accuracy/cost tradeoff
//! ✅ Industry standard for non-stiff ODEs
//! ✅ Good stability properties
//! ✅ No tuning parameters required
//!
//! # Limitations
//!
//! ⚠️ 4× more function evaluations than Euler
//! ⚠️ Still explicit (not ideal for very stiff problems)
//! ⚠️ Fixed time step (not adaptive)
//!
//! # When to Use
//!
//! - **Production chromatography simulations** ✅
//! - Non-stiff or moderately stiff problems
//! - When accuracy is important but adaptivity is not required
//! - Standard choice for most ODE problems
//!
//! # When NOT to Use
//!
//! - Very stiff problems → Use implicit methods (BDF, Rosenbrock)
//! - Need error control → Use adaptive RK45 or Dormand-Prince
//! - Extreme efficiency needed → Consider higher-order methods
//!
//! # Comparison with Euler
//!
//! | Method | Order | Evals/Step | Typical dt | Error |
//! |--------|-------|------------|------------|-------|
//! | Euler  | 1     | 1          | Small      | O(dt) |
//! | RK4    | 4     | 4          | Moderate   | O(dt⁴)|
//!
//! **Example**: For dt = 0.01:
//! - Euler error: ~ 0.01
//! - RK4 error: ~ 0.00000001 (10,000× better!)
//!
//! # Example
//!
//! ```rust,ignore
//! use chrom_rs::solver::{RK4Solver, Solver, SolverConfiguration};
//!
//! let solver = RK4Solver::new();
//! let config = SolverConfiguration::time_evolution(600.0, 1000);
//!
//! let result = solver.solve(&scenario, &config)?;
//! ```

use crate::physics::PhysicalState;
use crate::solver::{Solver, SolverConfiguration, SolverType, SimulationResult, Scenario, validate_state};

// =================================================================================================
// RK4 Solver
// =================================================================================================

/// Classical fourth-order Runge-Kutta solver
///
/// Implements the RK4 time integration scheme with four intermediate stages
/// per time step, providing fourth-order accuracy.
///
/// # Algorithm
///
/// For ODE system dy/dt = f(y):
///
/// 1. Start with initial state y₀
/// 2. For each time step n = 0, 1, 2, ..., N-1:
///    - **Stage 1**: k₁ = f(yₙ, tₙ)
///      - Slope at beginning of interval
///    - **Stage 2**: k₂ = f(yₙ + dt/2·k₁, tₙ + dt/2)
///      - Slope at midpoint using Euler step with k₁
///    - **Stage 3**: k₃ = f(yₙ + dt/2·k₂, tₙ + dt/2)
///      - Slope at midpoint using Euler step with k₂
///    - **Stage 4**: k₄ = f(yₙ + dt·k₃, tₙ + dt)
///      - Slope at end of interval using Euler step with k₃
///    - **Update**: yₙ₊₁ = yₙ + dt/6·(k₁ + 2k₂ + 2k₃ + k₄)
///      - Weighted average with Simpson's rule weights
/// 3. Return complete trajectory
///
/// # Stability
///
/// RK4 has better stability than Forward Euler. The stability region
/// (where |R(z)| ≤ 1 for z = λ·dt) is larger, allowing bigger time steps.
///
/// For linear problems dy/dt = λy, RK4 is stable when:
///
/// ```text
/// |1 + z + z²/2 + z³/6 + z⁴/24| ≤ 1
/// ```
///
/// This allows ~2.78× larger time steps than Euler for the same stability.
///
/// # Error Analysis
///
/// - **Local truncation error**: O(dt⁵) per step
/// - **Global error**: O(dt⁴) after T/dt steps
/// - **Convergence**: Fourth-order convergence when refining dt
///
/// **Practical implication**: Halving dt reduces error by factor of 16!
///
/// # Example
///
/// ```rust
/// # use chrom_rs::solver::{RK4Solver, Solver, SolverConfiguration, Scenario, DomainBoundaries};
/// # use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
/// # use nalgebra::DVector;
/// # struct MyModel;
/// # impl PhysicalModel for MyModel {
/// #     fn points(&self) -> usize { 1 }
/// #     fn compute_physics(&self, state: &PhysicalState) -> PhysicalState { state.clone() }
/// #     fn setup_initial_state(&self) -> PhysicalState {
/// #         PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(DVector::from_vec(vec![1.0])))
/// #     }
/// #     fn name(&self) -> &str { "MyModel" }
/// # }
/// # fn main() -> Result<(), String> {
/// # let model = Box::new(MyModel);
/// # let boundaries = DomainBoundaries::temporal(model.setup_initial_state());
/// # let scenario = Scenario::new(model, boundaries);
/// let solver = RK4Solver::new();
///
/// // Can use larger time steps than Euler for same accuracy
/// let config = SolverConfiguration::time_evolution(
///     600.0,   // Total time (seconds)
///     1000,    // Much fewer steps needed than Euler!
/// );
///
/// let result = solver.solve(&scenario, &config)?;
/// println!("Solved with {} evaluations", 4 * 1000);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RK4Solver;

impl RK4Solver {
    /// Create a new RK4 solver
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::solver::{RK4Solver, Solver};
    ///
    /// let solver = RK4Solver::new();
    /// assert_eq!(solver.name(), "Runge Kutta (RK4)");
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Solver for RK4Solver {
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration
    ) -> Result<SimulationResult, String> {

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

        // RK4 time-stepping loop
        // For each step n = 0, 1, ..., time_steps - 1:
        //   1. Compute k₁, k₂, k₃, k₄ (4 stages)
        //   2. Update: yₙ₊₁ = yₙ + dt/6·(k₁ + 2k₂ + 2k₃ + k₄)
        //   3. Store state and time point
        //   4. Validate state

        for step in 0..time_steps {

            let t = (step as f64) * dt;

            // ====== RK4 Stages ======

            // Stage 1: Slope at beginning of interval
            // k₁ = f(yₙ, tₙ)

            let k1 = scenario.model.compute_physics(&state);

            // Stage 2: Slope at midpoint using Euler prediction with k₁
            // k₂ = f(yₙ + dt/2·k₁, tₙ + dt/2)

            let state_k2 = state.clone() + k1.clone() * (dt / 2.0);
            let k2 = scenario.model.compute_physics(&state_k2);

            // Stage 3: Slope at midpoint using Euler prediction with k₂
            // k₃ = f(yₙ + dt/2·k₂, tₙ + dt/2)

            let state_k3 = state.clone() + k2.clone() * (dt / 2.0);
            let k3 = scenario.model.compute_physics(&state_k3);

            // Stage 4: Slope at end using Euler prediction with k₃
            // k₄ = f(yₙ + dt·k₃, tₙ + dt)

            let state_k4 = state.clone() + k3.clone() * dt;
            let k4 = scenario.model.compute_physics(&state_k4);

            // ====== RK4 Update ======

            // Weighted combination using Simpson's rule weights:
            // yₙ₊₁ = yₙ + dt/6·(k₁ + 2k₂ + 2k₃ + k₄)
            //
            // Why these weights?
            // - k₁ and k₄ (endpoints): weight 1/6 each
            // - k₂ and k₃ (midpoints): weight 2/6 = 1/3 each
            // This comes from Simpson's quadrature rule for integration

            let weighted_slope = k1 +
                k2 * 2.0 +
                k3 * 2.0 +
                k4 ;

            state = state.clone() + weighted_slope * (dt / 6.0);

            // ====== Storage ======

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

            validate_state(&state, step + 1)?;

        }

        // ====== Step 4: Build Result ======

        let final_state = state;

        // Create simulation result

        let mut result = SimulationResult::new(
            time_points,
            state_trajectory,
            final_state,
        );

        // Add metadata
        result.add_metadata("solver", "Runge-Kutta 4");
        result.add_metadata("time steps", &time_steps.to_string());
        result.add_metadata("dt", &dt.to_string());
        result.add_metadata("total time", &total_time.to_string());
        result.add_metadata("function evaluations", &(4 * time_steps).to_string());

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "Runge Kutta (RK4)"
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
    use nalgebra::DVector;

    // ====== Mock Models for Testing ======

    // ====== Common with Euler solver method ======

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

    // ====== Additional model ======

    /// Mock model: simple harmonic oscillator d²y/dt² = -omega^2y
    /// Rewritten as first-order system:
    ///   dy_1/dt = y_2         (velocity)
    ///   dy_2/dt = -omega^2·y_1     (acceleration)
    struct HarmonicOscillator {
        points: usize,
        omega: f64,  // Angular frequency
    }

    impl PhysicalModel for HarmonicOscillator {
        fn points(&self) -> usize {
            self.points
        }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            let mut result = PhysicalState::empty();

            // Get position and velocity
            let y1 = state.get(PhysicalQuantity::Concentration).unwrap();
            let y2 = state.get(PhysicalQuantity::Velocity).unwrap();

            // dy_1/dt = y_2
            result.set(PhysicalQuantity::Concentration, y2.clone());

            // dy_2/dt = -omega^2·y_1
            let mut dy2 = y1.clone();
            dy2.apply(|y| -self.omega * self.omega * y);
            result.set(PhysicalQuantity::Velocity, dy2);

            result
        }

        fn setup_initial_state(&self) -> PhysicalState {
            let mut state = PhysicalState::empty();
            // Initial position: y(0) = 1
            state.set(
                PhysicalQuantity::Concentration,
                PhysicalData::uniform_vector(self.points, 1.0),
            );
            // Initial velocity: dy/dt(0) = 0
            state.set(
                PhysicalQuantity::Velocity,
                PhysicalData::uniform_vector(self.points, 0.0),
            );
            state
        }

        fn name(&self) -> &str {
            "Harmonic Oscillator"
        }
    }

    // ====== Solver creation tests ======

    #[test]
    fn test_rk4_solver_creation() {

        let solver = RK4Solver::new();
        assert_eq!(solver.name(), "Runge Kutta (RK4)");
    }

    #[test]
    fn test_rk4_solver_default() {
        let solver = RK4Solver::default();
        assert_eq!(solver.name(), "Runge Kutta (RK4)");
    }

    // ====== Configuration Tests ======

    #[test]
    fn test_rk4_accepts_time_evolution() {
        let solver = RK4Solver::new();
        let config = SolverConfiguration::time_evolution(10.0, 100);

        let model = Box::new(ConstantGrowth {
            points: 10,
            growth_rate: 1.0,
        });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let result = solver.solve(&scenario, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rk4_rejects_iterative() {
        let solver = RK4Solver::new();
        let config = SolverConfiguration::iterative(1e-6, 100);

        let model = Box::new(ConstantGrowth {
            points: 10,
            growth_rate: 1.0,
        });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let result = solver.solve(&scenario, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("only supports TimeEvolution"));
    }

    #[test]
    fn test_rk4_rejects_analytical() {
        let solver = RK4Solver::new();
        let config = SolverConfiguration::analytical(5.0);

        let model = Box::new(ConstantGrowth {
            points: 10,
            growth_rate: 1.0,
        });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let result = solver.solve(&scenario, &config);
        assert!(result.is_err());
    }

    // ====== Numerical accuracy tests ======

    #[test]
    fn test_rk4_constant_growth() {

        // dy/dt = c → y(t) = y_0 + c*t
        // RK4 should be exact for linear problems (within floating-point precision)

        let solver = RK4Solver::new();
        let growth_rate = 2.0;

        let model = Box::new(ConstantGrowth {
            points: 5,
            growth_rate,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let total_time = 10.0;
        let config = SolverConfiguration::time_evolution(total_time, 100);

        let result = solver.solve(&scenario, &config).unwrap();

        // Check final concentration is y(10) = 0 + 10.0 * 2.0 = 20.0

        let final_concentration = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();

        let expected_concentration = growth_rate * total_time;
        let calculated_concentration = final_concentration.as_vector()[0];

        // RK4 should be exact for constant dy/dt (within floating-point precision)

        assert!((calculated_concentration - expected_concentration).abs() < 1e-10);
    }

    #[test]
    fn test_rk4_exponential_decay() {
        // dy/dt = -k*y → y(t) = y_0 * exp(-k*t)
        // RK4 has fourth-order error

        let solver = RK4Solver::new();
        let decay_rate = 0.1;

        let model = Box::new(ExponentialDecay {
            points:5,
            decay_rate
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let total_time = 10.0;
        let time_steps = 100;

        let config = SolverConfiguration::time_evolution(total_time, time_steps);

        let result = solver.solve(&scenario, &config).unwrap();

        // Analytical solution is y(10) = 1.0 * exp(-0.1 * 10) = exp(-1) = 1/e (≈ 0.367879)

        let expected_concentraion = (- decay_rate * total_time).exp();
        let final_concentration = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();
        let actual_concentration = final_concentration.as_vector()[0];

        // RK4 has O(dt⁴) error, with dt = 10/100 = 0.1
        // Error should be ~ (0.1)⁴ = 0.0001

        let error = (actual_concentration - expected_concentraion).abs();

        assert!(error < 1e-4, "Error {} is too large for RK4", error);
    }

    #[test]
    fn test_rk4_convergence() {

        let solver = RK4Solver::new();
        let decay_rate = 0.1;
        let total_time = 5.0;

        let model = Box::new(ExponentialDecay {
            points:3,
            decay_rate
        });

        let exact_solution = (-decay_rate * total_time).exp();

        // Test with different time steps
        let vsteps = vec![50, 100, 200, 400];
        let mut verrors: Vec<f64> = Vec::new();

        for &steps in &vsteps {

            let initial = model.setup_initial_state();
            let boundaries = DomainBoundaries::temporal(initial);
            let scenario = Scenario::new(
                Box::new(
                ExponentialDecay{
                    points:3,
                    decay_rate }),
                boundaries);

            let config = SolverConfiguration::time_evolution(total_time, steps);
            let result = solver.solve(&scenario, &config).unwrap();

            let final_concentration = result.final_state
                .get(PhysicalQuantity::Concentration)
                .unwrap();

            verrors.push(
                (final_concentration.as_vector()[0] - exact_solution)
                    .abs());
        }

        // Check fourth-order convergence: error(dt/2) ≈ error(dt) / 16

        for i in 0..verrors.len() - 1 {
            let ratio = verrors[i] / verrors[i + 1];

            assert!(
                ratio > 12.0 && ratio < 20.0,
                "Convergence ratio {} is not a fourth-order a step {}",
                ratio,
                i);
        }
    }

    #[test]
    fn test_rk4_solver_harmonic_oscillator() {

        // Test RK4 on oscillatory problem
        // d²y/dt² = -omega^2y → y(t) = cos(ωt)

        let solver = RK4Solver::new();
        let omega = 1.0; // ω = 1 → period T = 2π

        let model = Box::new(
          HarmonicOscillator {
              points: 3,
              omega
          }
        );

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        // Integrate solution over on full period
        let period = 2.0 * std::f64::consts::PI;
        let config = SolverConfiguration::time_evolution(period, 100);

        let result = solver.solve(&scenario, &config).unwrap();

        // After one period, should return to initial position
        // y(2π) = cos(2π) = 1.0

        let final_position = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();

        let expected = 1.0;
        let actual_position = final_position.as_vector()[0];

        assert!((actual_position - expected).abs() < 0.01);
    }

    // ================================= Trajectory Tests =================================

    #[test]
    fn test_rk4_trajectory_length() {
        let solver = RK4Solver::new();

        let model = Box::new(ConstantGrowth {
            points: 5,
            growth_rate: 1.0,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let time_steps = 100;
        let config = SolverConfiguration::time_evolution(10.0, time_steps);

        let result = solver.solve(&scenario, &config).unwrap();

        // Should have time_steps + 1 points (including initial)
        assert_eq!(result.time_points.len(), time_steps + 1);
        assert_eq!(result.state_trajectory.len(), time_steps + 1);
    }

    #[test]
    fn test_rk4_time_points() {
        let solver = RK4Solver::new();

        let model = Box::new(ConstantGrowth {
            points: 5,
            growth_rate: 1.0,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let total_time = 20.0;
        let time_steps = 100;
        let dt = total_time / (time_steps as f64);
        let config = SolverConfiguration::time_evolution(total_time, time_steps);

        let result = solver.solve(&scenario, &config).unwrap();

        // Check first time point is 0
        assert!((result.time_points[0] - 0.0).abs() < 1e-10);

        // Check last time point is total_time
        assert!((result.time_points.last().unwrap() - total_time).abs() < 1e-10);

        // Check uniform spacing
        for i in 1..result.time_points.len() {
            let actual_dt = result.time_points[i] - result.time_points[i - 1];
            assert!((actual_dt - dt).abs() < 1e-10);
        }
    }

    // ====== Metadata Tests ======

    #[test]
    fn test_rk4_metadata() {
        let solver = RK4Solver::new();

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
        assert_eq!(result.metadata.get("solver"), Some(&"Runge-Kutta 4".to_string()));
        assert_eq!(result.metadata.get("time steps"), Some(&"500".to_string()));
        assert_eq!(result.metadata.get("total time"), Some(&"100".to_string()));

        // Function evaluations = 4 * time_steps
        assert_eq!(result.metadata.get("function evaluations"), Some(&"2000".to_string()));

        // dt = 100 / 500 = 0.2
        let dt_str = result.metadata.get("dt").unwrap();
        let dt: f64 = dt_str.parse().unwrap();
        assert!((dt - 0.2).abs() < 1e-10);
    }

    // ====== Validation Tests ======

    #[test]
    fn test_rk4_detects_nan() {
        struct NaNModel {
            points: usize,
        }

        impl PhysicalModel for NaNModel {
            fn points(&self) -> usize {
                self.points
            }

            fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {
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

        let solver = RK4Solver::new();
        let model = Box::new(NaNModel { points: 5 });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(10.0, 10);
        let result = solver.solve(&scenario, &config);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("NaN"));
    }

    #[test]
    fn test_rk4_detects_inf() {
        struct InfModel {
            points: usize,
        }

        impl PhysicalModel for InfModel {
            fn points(&self) -> usize {
                self.points
            }

            fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {
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

        let solver = RK4Solver::new();
        let model = Box::new(InfModel { points: 5 });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(10.0, 10);
        let result = solver.solve(&scenario, &config);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Infinity"));
    }

    // ====== Edge Cases ======

    #[test]
    fn test_rk4_single_step() {
        let solver = RK4Solver::new();

        let model = Box::new(ConstantGrowth {
            points: 3,
            growth_rate: 5.0,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(1.0, 1);
        let result = solver.solve(&scenario, &config).unwrap();

        // Should have 2 points (initial + final)
        assert_eq!(result.time_points.len(), 2);
        assert_eq!(result.state_trajectory.len(), 2);

        // y(1) = 0 + 5*1 = 5
        let final_conc = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();
        assert!((final_conc.as_vector()[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rk4_multi_quantity() {
        struct MultiQuantityModel {
            points: usize,
        }

        impl PhysicalModel for MultiQuantityModel {
            fn points(&self) -> usize {
                self.points
            }

            fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {
                let mut result = PhysicalState::empty();

                // dC/dt = 1
                result.set(
                    PhysicalQuantity::Concentration,
                    PhysicalData::uniform_vector(self.points, 1.0),
                );

                // dT/dt = 0.1
                result.set(
                    PhysicalQuantity::Temperature,
                    PhysicalData::uniform_vector(self.points, 0.1),
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
                    PhysicalData::uniform_vector(self.points, 298.0),
                );
                state
            }

            fn name(&self) -> &str {
                "Multi-Quantity Model"
            }
        }

        let solver = RK4Solver::new();
        let model = Box::new(MultiQuantityModel { points: 5 });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(10.0, 100);
        let result = solver.solve(&scenario, &config).unwrap();

        // Check concentration: y(10) = 0 + 1*10 = 10
        let final_conc = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap();
        assert!((final_conc.as_vector()[0] - 10.0).abs() < 1e-10);

        // Check temperature: T(10) = 298 + 0.1*10 = 299
        let final_temp = result.final_state
            .get(PhysicalQuantity::Temperature)
            .unwrap();
        assert!((final_temp.as_vector()[0] - 299.0).abs() < 1e-10);
    }

    // ====== Comparison with Analytical Solutions ======

    #[test]
    fn test_rk4_vs_analytical_accuracy() {
        // Compare RK4 with known analytical solution
        let solver = RK4Solver::new();
        let k = 0.3;

        let model = Box::new(ExponentialDecay {
            points: 1,
            decay_rate: k,
        });

        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        // Test at multiple time points
        let test_times = vec![1.0, 5.0, 10.0, 20.0];

        for &t in &test_times {
            let config = SolverConfiguration::time_evolution(t, 100);
            let result = solver.solve(&scenario, &config).unwrap();

            let analytical = (-k * t).exp();
            let numerical = result.final_state
                .get(PhysicalQuantity::Concentration)
                .unwrap()
                .as_vector()[0];

            let relative_error = ((numerical - analytical) / analytical).abs();

            // RK4 should maintain <0.1% relative error with dt=0.1
            assert!(
                relative_error < 0.001,
                "At t={}: relative error {} too large",
                t, relative_error
            );
        }
    }

}