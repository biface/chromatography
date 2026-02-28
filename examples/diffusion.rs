//! Fisher-KPP Equation with Mixed Boundaries
//!
//! ∂u/∂t = D·∂²u/∂x² + r·u·(1-u)
//!
//! Uses DomainBoundaries::mixed() for:
//! - Spatial boundaries: u(x=0)=1.0, u(x=L)=0.0
//! - Temporal initial state: u(x,t=0) = profile

use chrom_rs::{
    physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData},
    solver::{Scenario, SolverConfiguration, DomainBoundaries, Solver, EulerSolver, RK4Solver},
    output::visualization::{plot_steady_state, plot_steady_state_comparison, PlotConfig},
};
use nalgebra::DVector;
use std::error::Error;

/// Fisher-KPP diffusion-reaction model
struct FisherKPP {
    n_points: usize,
    length: f64,
    dx: f64,
    diffusion: f64,
    growth_rate: f64,
    bc_left: f64,
    bc_right: f64,
}

impl FisherKPP {
    fn new(
        n_points: usize,
        length: f64,
        diffusion: f64,
        growth_rate: f64,
        bc_left: f64,
        bc_right: f64,
    ) -> Self {
        let dx = length / (n_points - 1) as f64;
        Self {
            n_points,
            length,
            dx,
            diffusion,
            growth_rate,
            bc_left,
            bc_right,
        }
    }
}

impl PhysicalModel for FisherKPP {
    fn points(&self) -> usize {
        self.n_points
    }

    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
        let u = state.get(PhysicalQuantity::Concentration)
            .expect("Concentration not found")
            .as_vector();

        let mut du_dt = DVector::zeros(self.n_points);

        for i in 0..self.n_points {
            // Conditions de Dirichlet : bords fixes
            if i == 0 || i == self.n_points - 1 {
                du_dt[i] = 0.0;  // ← Bords ne changent PAS
                continue;
            }

            // Diffusion term: D·∂²u/∂x² (points intérieurs seulement)
            let diffusion_term = self.diffusion * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (self.dx * self.dx);

            // Reaction term: r·u·(1-u)
            let reaction_term = self.growth_rate * u[i] * (1.0 - u[i]);

            du_dt[i] = diffusion_term + reaction_term;
        }

        PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(du_dt))
    }

    fn setup_initial_state(&self) -> PhysicalState {
        let mut u = DVector::zeros(self.n_points);

        // Initial condition: linear interpolation + sinusoidal perturbation
        for i in 0..self.n_points {
            let x = (i as f64 / (self.n_points - 1) as f64) * self.length;
            let t = i as f64 / (self.n_points - 1) as f64;

            let u_linear = self.bc_left * (1.0 - t) + self.bc_right * t;
            let perturbation = 0.1 * (3.0 * std::f64::consts::PI * x / self.length).sin();

            u[i] = (u_linear + perturbation).max(0.0).min(1.0);
        }

        // Forcer les conditions de Dirichlet
        u[0] = self.bc_left;                    // ← u(0) = 1.0
        u[self.n_points - 1] = self.bc_right;   // ← u(L) = 0.0

        PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(u))
    }

    fn name(&self) -> &str {
        "Fisher-KPP Diffusion-Reaction"
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Fisher-KPP: Mixed Boundaries (Spatial + Temporal) ===\n");

    // Physical parameters
    let length = 1.0;
    let n_points = 200;
    let diffusion = 0.001;
    let growth_rate = 1.0;
    let bc_left = 1.0;
    let bc_right = 0.0;

    // Simulation parameters
    let final_time = 5.0;
    let n_steps = 2500;

    println!("Physical Parameters:");
    println!("  Domain length: {} m", length);
    println!("  Spatial points: {}", n_points);
    println!("  Diffusion D: {} m²/s", diffusion);
    println!("  Growth rate r: {} s⁻¹", growth_rate);
    println!("\nBoundary Conditions (Spatial):");
    println!("  Left (x=0): u = {}", bc_left);
    println!("  Right (x=L): u = {}", bc_right);
    println!("\nSimulation:");
    println!("  Final time: {} s", final_time);
    println!("  Time steps: {}\n", n_steps);

    // Create model
    let model = Box::new(FisherKPP::new(
        n_points,
        length,
        diffusion,
        growth_rate,
        bc_left,
        bc_right,
    ));

    // Spatial boundary states
    let begin_state = PhysicalState::new(
        PhysicalQuantity::Concentration,
        PhysicalData::Scalar(bc_left),
    );

    let end_state = PhysicalState::new(
        PhysicalQuantity::Concentration,
        PhysicalData::Scalar(bc_right),
    );

    // Initial temporal state
    let initial_state = model.setup_initial_state();

    // Create scenario with mixed boundaries
    let scenario = Scenario::new(
        model,
        DomainBoundaries::mixed(
            &["x"],                 // Spatial dimension name
            vec![begin_state],      // u(x=0, t) = bc_left
            vec![end_state],        // u(x=L, t) = bc_right
            initial_state,          // u(x, t=0) = initial profile
        ),
    );

    let config = SolverConfiguration::time_evolution(final_time, n_steps);

    // Solve with Euler
    println!("Solving with Forward Euler...");
    let start = std::time::Instant::now();
    let result_euler = EulerSolver.solve(&scenario, &config)?;
    let elapsed_euler = start.elapsed();
    println!("✓ Euler completed in {:.3}s\n", elapsed_euler.as_secs_f64());

    let u_final = result_euler.state_trajectory
        .last()
        .unwrap()
        .get(PhysicalQuantity::Concentration).unwrap().as_vector();

    println!("\nBoundary values (Euler):");
    println!("  u(x=0) = {:.10}", u_final[0]);                  // Doit être 1.0
    println!("  u(x=L) = {:.10}", u_final[n_points - 1]);       // Doit être 0.0

    // Solve with RK4
    println!("Solving with RK4...");
    let start = std::time::Instant::now();
    let result_rk4 = RK4Solver.solve(&scenario, &config)?;
    let elapsed_rk4 = start.elapsed();
    println!("✓ RK4 completed in {:.3}s\n", elapsed_rk4.as_secs_f64());

    let u_final = result_euler.state_trajectory
        .last()
        .unwrap()
        .get(PhysicalQuantity::Concentration).unwrap().as_vector();

    println!("\nBoundary values (RK4):");
    println!("  u(x=0) = {:.10}", u_final[0]);                  // Doit être 1.0
    println!("  u(x=L) = {:.10}", u_final[n_points - 1]);       // Doit être 0.0

    // Analysis
    let u_euler_final = result_euler.state_trajectory.last()
        .unwrap()
        .get(PhysicalQuantity::Concentration)
        .unwrap()
        .as_vector();

    let u_rk4_final = result_rk4.state_trajectory.last()
        .unwrap()
        .get(PhysicalQuantity::Concentration)
        .unwrap()
        .as_vector();

    // Find front position (u = 0.5)
    let find_front = |u: &nalgebra::DVector<f64>| -> Option<f64> {
        for i in 0..u.len() - 1 {
            if u[i] >= 0.5 && u[i + 1] < 0.5 {
                let t = (0.5 - u[i + 1]) / (u[i] - u[i + 1]);
                let x = (i as f64 + t) / (n_points - 1) as f64 * length;
                return Some(x);
            }
        }
        None
    };

    println!("Analysis:");
    if let Some(front_euler) = find_front(u_euler_final) {
        println!("  Euler front position (u=0.5): {:.4} m", front_euler);
    }
    if let Some(front_rk4) = find_front(u_rk4_final) {
        println!("  RK4 front position (u=0.5):   {:.4} m", front_rk4);
    }

    let diff: f64 = (0..n_points)
        .map(|i| (u_euler_final[i] - u_rk4_final[i]).powi(2))
        .sum();
    let l2_diff = (diff / n_points as f64).sqrt();
    println!("  L² difference Euler/RK4: {:.6}\n", l2_diff);

    // Generate plots
    println!("Generating plots...");

    let tmp_dir = std::env::temp_dir();

    let config_euler = PlotConfig::steady_state("Fisher-KPP: Euler - Final Profile");
    plot_steady_state(&result_euler, length, tmp_dir.join("fisher_kpp_euler.png").to_str().unwrap(), Some(&config_euler))?;
    println!("✓ fisher_kpp_euler.png");

    let config_rk4 = PlotConfig::steady_state("Fisher-KPP: RK4 - Final Profile");
    plot_steady_state(&result_rk4, length, tmp_dir.join("fisher_kpp_rk4.png").to_str().unwrap(), Some(&config_rk4))?;
    println!("✓ fisher_kpp_rk4.png");

    // Comparison plot
    let x_grid: Vec<f64> = (0..n_points)
        .map(|i| (i as f64 / (n_points - 1) as f64) * length)
        .collect();

    let u_euler_vec: Vec<f64> = u_euler_final.iter().cloned().collect();
    let u_rk4_vec: Vec<f64> = u_rk4_final.iter().cloned().collect();

    let profiles = vec![
        ("Euler", x_grid.as_slice(), u_euler_vec.as_slice()),
        ("RK4", x_grid.as_slice(), u_rk4_vec.as_slice()),
    ];

    let config_comp = PlotConfig::steady_state("Fisher-KPP: Euler vs RK4");
    plot_steady_state_comparison(profiles, "fisher_kpp_comparison.png", Some(&config_comp))?;
    println!("✓ fisher_kpp_comparison.png");

    println!("\n=== Simulation Complete ===");
    println!("Using mixed boundaries:");
    println!("  - Spatial: u(x=0)={}, u(x=L)={}", bc_left, bc_right);
    println!("  - Temporal: initial profile at t=0");
    println!("\nExpected: Travelling wave front from left to right");

    Ok(())
}