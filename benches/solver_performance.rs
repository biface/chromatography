//! Performance benchmarks for numerical solvers

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use chrom_rs::solver::{Solver, SolverConfiguration, Scenario, DomainBoundaries};
use chrom_rs::solver::{EulerSolver, RK4Solver};
use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};

// =================================================================================================
// Simple Model for Benchmarking
// =================================================================================================

/// Simple physical model for benchmarking purposes
///
/// Implements exponential decay: dy/dt = -k*y with k = 0.1
struct SimpleModel {
    points: usize,
}

impl PhysicalModel for SimpleModel {
    fn points(&self) -> usize {
        self.points
    }

    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
        let mut result = state.clone();
        if let Some(conc) = result.get_mut(PhysicalQuantity::Concentration) {
            conc.apply(|x| -0.1 * x);
        }
        result
    }

    fn setup_initial_state(&self) -> PhysicalState {
        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(self.points, 1.0),
        )
    }

    fn name(&self) -> &str {
        "Simple Model"
    }
}

// =================================================================================================
// Benchmark Functions
// =================================================================================================

/// Benchmark Euler solver with different problem sizes
///
/// Tests performance with 10, 50, 100, and 500 spatial points
fn benchmark_euler_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler_solver");

    for points in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(points),
            points,
            |b, &points| {
                let model = Box::new(SimpleModel { points });
                let initial = model.setup_initial_state();
                let boundaries = DomainBoundaries::temporal(initial);
                let scenario = Scenario::new(model, boundaries);
                let config = SolverConfiguration::time_evolution(10.0, 100);
                let solver = EulerSolver::new();

                b.iter(|| {
                    solver.solve(black_box(&scenario), black_box(&config)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RK4 solver with different problem sizes
///
/// Tests performance with 10, 50, 100, and 500 spatial points
fn benchmark_rk4_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("rk4_solver");

    for points in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(points),
            points,
            |b, &points| {
                let model = Box::new(SimpleModel { points });
                let initial = model.setup_initial_state();
                let boundaries = DomainBoundaries::temporal(initial);
                let scenario = Scenario::new(model, boundaries);
                let config = SolverConfiguration::time_evolution(10.0, 100);
                let solver = RK4Solver::new();

                b.iter(|| {
                    solver.solve(black_box(&scenario), black_box(&config)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Direct comparison between Euler and RK4 solvers
///
/// Runs both solvers on the same problem (100 points, 1000 time steps)
/// to directly compare their performance characteristics.
fn benchmark_solver_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_comparison");
    let points = 100;
    let time_steps = 1000;

    // Euler
    group.bench_function("euler", |b| {
        let model = Box::new(SimpleModel { points });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);
        let config = SolverConfiguration::time_evolution(100.0, time_steps);
        let solver = EulerSolver::new();

        b.iter(|| {
            solver.solve(black_box(&scenario), black_box(&config)).unwrap()
        });
    });

    // RK4
    group.bench_function("rk4", |b| {
        let model = Box::new(SimpleModel { points });
        let initial = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);
        let config = SolverConfiguration::time_evolution(100.0, time_steps);
        let solver = RK4Solver::new();

        b.iter(|| {
            solver.solve(black_box(&scenario), black_box(&config)).unwrap()
        });
    });

    group.finish();
}

// =================================================================================================
// Criterion Configuration
// =================================================================================================

criterion_group!(
    benches,
    benchmark_euler_solver,
    benchmark_rk4_solver,
    benchmark_solver_comparison
);
criterion_main!(benches);