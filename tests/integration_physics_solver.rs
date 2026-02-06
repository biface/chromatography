//! Integration tests: physics module + solver module
//!
//! These tests verify that the physics and solver modules
//! work correctly together.

use chrom_rs::solver::{
    Solver, SolverConfiguration, Scenario, DomainBoundaries,
};
use chrom_rs::solver::{EulerSolver, RK4Solver};
use chrom_rs::physics::{PhysicalModel, PhysicalQuantity};

mod common;
use common::{ExponentialDecay, ConstantGrowth};
use common::test_helpers::relative_error;

// =================================================================================================
// Basic Integration Tests
// =================================================================================================

#[test]
fn test_euler_with_exponential_decay() {
    // Setup
    let model = Box::new(ExponentialDecay::new(5, 0.1));
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(model, boundaries);

    // Solve
    let config = SolverConfiguration::time_evolution(10.0, 1000);
    let solver = EulerSolver::new();
    let result = solver.solve(&scenario, &config).unwrap();

    // Verify
    assert_eq!(result.time_points.len(), 1001);
    assert!(result.time_points[0].abs() < 1e-10);
    assert!((result.time_points.last().unwrap() - 10.0).abs() < 1e-10);

    // Check final value
    let final_conc = result.final_state
        .get(PhysicalQuantity::Concentration)
        .unwrap()
        .as_vector()[0];

    // Analytical: y(10) = exp(-0.1 * 10) = exp(-1) ≈ 0.3679
    let expected = (-0.1 * 10.0f64).exp();
    let error:f64 = relative_error(final_conc, expected);

    // Euler with dt=0.01 should have ~1% error
    assert!(error < 0.02, "Error {} too large", error);
}

#[test]
fn test_rk4_with_exponential_decay() {
    // Setup
    let model = Box::new(ExponentialDecay::new(5, 0.1));
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(model, boundaries);

    // Solve
    let config = SolverConfiguration::time_evolution(10.0, 100);  // Fewer steps
    let solver = RK4Solver::new();
    let result = solver.solve(&scenario, &config).unwrap();

    // Check final value
    let final_conc = result.final_state
        .get(PhysicalQuantity::Concentration)
        .unwrap()
        .as_vector()[0];

    // Analytical
    let expected = (-0.1 * 10.0f64).exp();
    let error = relative_error(final_conc, expected);

    // RK4 should be very accurate even with dt=0.1
    assert!(error < 1e-5, "Error {} too large for RK4", error);
}

#[test]
fn test_euler_is_exact_for_constant_growth() {
    // Setup
    let model = Box::new(ConstantGrowth::new(3, 2.0));
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(model, boundaries);

    // Solve
    let config = SolverConfiguration::time_evolution(5.0, 10);
    let solver = EulerSolver::new();
    let result = solver.solve(&scenario, &config).unwrap();

    // Check final value
    let final_conc = result.final_state
        .get(PhysicalQuantity::Concentration)
        .unwrap()
        .as_vector()[0];

    // Analytical: y(5) = 0 + 2*5 = 10
    let expected = 2.0 * 5.0;

    // Euler should be exact for constant dy/dt
    assert!((final_conc - expected).abs() < 1e-10);
}

// =================================================================================================
// Cross-Solver Comparison Tests
// =================================================================================================

#[test]
fn test_euler_vs_rk4_same_problem() {
    // Setup same problem for both solvers
    let decay_rate = 0.5;
    let total_time = 5.0;
    let time_steps = 500;

    // Euler
    let model1 = Box::new(ExponentialDecay::new(10, decay_rate));
    let initial1 = model1.setup_initial_state();
    let boundaries1 = DomainBoundaries::temporal(initial1);
    let scenario1 = Scenario::new(model1, boundaries1);
    let config1 = SolverConfiguration::time_evolution(total_time, time_steps);

    let euler = EulerSolver::new();
    let result_euler = euler.solve(&scenario1, &config1).unwrap();

    // RK4
    let model2 = Box::new(ExponentialDecay::new(10, decay_rate));
    let initial2 = model2.setup_initial_state();
    let boundaries2 = DomainBoundaries::temporal(initial2);
    let scenario2 = Scenario::new(model2, boundaries2);
    let config2 = SolverConfiguration::time_evolution(total_time, time_steps);

    let rk4 = RK4Solver::new();
    let result_rk4 = rk4.solve(&scenario2, &config2).unwrap();

    // Compare errors
    let exact = (-decay_rate * total_time).exp();

    let euler_final = result_euler.final_state
        .get(PhysicalQuantity::Concentration)
        .unwrap()
        .as_vector()[0];
    let euler_error = relative_error(euler_final, exact);

    let rk4_final = result_rk4.final_state
        .get(PhysicalQuantity::Concentration)
        .unwrap()
        .as_vector()[0];
    let rk4_error = relative_error(rk4_final, exact);

    // RK4 should be significantly more accurate
    assert!(
        rk4_error < euler_error / 10.0,
        "RK4 error {} not much better than Euler error {}",
        rk4_error, euler_error
    );
}

// =================================================================================================
// Error Detection Tests
// =================================================================================================

#[test]
fn test_solver_detects_invalid_config() {
    let model = Box::new(ConstantGrowth::new(5, 1.0));
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    let scenario = Scenario::new(model, boundaries);

    // Invalid: iterative config for time evolution solver
    let config = SolverConfiguration::iterative(1e-6, 100);

    let euler = EulerSolver::new();
    let result = euler.solve(&scenario, &config);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("only supports TimeEvolution"));
}