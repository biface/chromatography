//! Convergence tests for numerical solvers
//!
//! These tests verify that solvers exhibit the expected
//! convergence rates when refining the time step.

use chrom_rs::solver::{Solver, SolverConfiguration, Scenario, DomainBoundaries};
use chrom_rs::solver::{EulerSolver, RK4Solver};
use chrom_rs::physics::{PhysicalModel, PhysicalQuantity};

mod common;
use common::ExponentialDecay;

#[test]
fn test_euler_first_order_convergence() {
    // Euler should have first-order convergence: error ~ O(dt)
    // When dt → dt/2, error should → error/2

    let decay_rate = 0.3;
    let total_time = 10.0;
    let exact = (-decay_rate * (total_time as f64)).exp();

    let steps_list = vec![100, 200, 400, 800];
    let mut errors = Vec::new();

    let euler = EulerSolver::new();

    for &steps in &steps_list {
        let model = Box::new(ExponentialDecay::new(5, decay_rate));
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(total_time, steps);
        let result = euler.solve(&scenario, &config).unwrap();

        let final_conc = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_vector()[0];

        let error = (final_conc - exact).abs();
        errors.push(error);
    }

    // Check convergence ratios
    for i in 0..errors.len() - 1 {
        let ratio = errors[i] / errors[i + 1];
        println!("Euler convergence ratio {}->{}: {}", i, i+1, ratio);

        // Should be close to 2 for first-order
        assert!(
            ratio > 1.8 && ratio < 2.2,
            "Convergence ratio {} not first-order",
            ratio
        );
    }
}

#[test]
fn test_rk4_fourth_order_convergence() {
    // RK4 should have fourth-order convergence: error ~ O(dt^4)
    // When dt → dt/2, error should → error/16

    let decay_rate = 0.3;
    let total_time = 5.0;
    let exact = (-decay_rate * (total_time as f64)).exp();

    let steps_list = vec![10, 20, 40, 80];
    let mut errors = Vec::new();

    let rk4 = RK4Solver::new();

    for &steps in &steps_list {
        let model = Box::new(ExponentialDecay::new(5, decay_rate));
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        let config = SolverConfiguration::time_evolution(total_time, steps);
        let result = rk4.solve(&scenario, &config).unwrap();

        let final_conc = result.final_state
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_vector()[0];

        let error = (final_conc - exact).abs();
        errors.push(error);
    }

    // Check convergence ratios
    for i in 0..errors.len() - 1 {
        let ratio = errors[i] / errors[i + 1];
        println!("RK4 convergence ratio {}->{}: {}", i, i+1, ratio);

        // Should be close to 16 for fourth-order
        assert!(
            ratio > 12.0 && ratio < 20.0,
            "Convergence ratio {} not fourth-order",
            ratio
        );
    }
}