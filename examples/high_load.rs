//! Example: High-Load Chromatography - Langmuir Saturation
//!
//! Demonstrates non-linear Langmuir behavior with high concentration.
//! Uses the same injection patterns as TFA but with stronger adsorption parameters.
//!
//! **Physical System**:
//! - Column: Preparative C18, 25 cm length
//! - Mobile phase: Methanol/water
//! - Sample: High concentration pulse (column overload)
//!
//! **Parameters** (stronger adsorption than TFA):
//! - λ = 0.8 (lower → more retention)
//! - K̃ = 1.2 L/mol (higher → stronger adsorption)
//! - N = 1.5 (lower → easier saturation)
//! - εₑ = 0.35 (tighter packing)
//! - u = 0.0008 m/s (slower flow)
//! - L = 0.25 m
//! - Spatial points: 100

use chrom_rs::{
    models::{LangmuirSingle, TemporalInjection},
    physics::{PhysicalModel, PhysicalQuantity},
    solver::{Scenario, SolverConfiguration, SimulationResult, DomainBoundaries, EulerSolver, RK4Solver, Solver},
    output::{plot_chromatogram, PlotConfig},
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  High-Load Chromatography - Langmuir Saturation");
    println!("═══════════════════════════════════════════════════════\n");

    // ====== High-Load Physical parameters ======

    let lambda = 0.6;           // Lower → more retention
    let langmuir_k = 2.5;       // Higher → stronger adsorption
    let port_number = 0.6;      // Lower → easier saturation
    let porosity = 0.35;        // Tighter packing
    let velocity = 0.0008;      // Slower flow
    let column_length = 0.25;
    let n_points = 3000;

    println!("High-Load Parameters:");
    println!("  λ (lambda)     : {} (lower than TFA)", lambda);
    println!("  K̃ (langmuir_k) : {} L/mol (higher than TFA)", langmuir_k);
    println!("  N (capacity)   : {} mol/L (lower than TFA)", port_number);
    println!("  ε (porosity)   : {}", porosity);
    println!("  u (velocity)   : {} m/s (slower than TFA)", velocity);
    println!("  L (length)     : {} m", column_length);
    println!("  nz (points)    : {}\n", n_points);

    // ====== Simulation configuration ======

    let total_time = 800.0;     // Longer due to stronger retention
    let time_steps = 12000;
    println!("Simulation:");
    println!("  Total time : {} s ({:.1} min)", total_time, total_time / 60.0);
    println!("  Time steps : {}", time_steps);
    println!("  dt         : {:.6} s\n", total_time / time_steps as f64);

    // ====== Temporary directory ======

    let tmp_dir = std::env::temp_dir();

    // ====== Injection profiles ======

    // Dirac: instantaneous 0.1 mol/L at 5.0 seconds
    let dirac = TemporalInjection::dirac(5.0, 1.0);

    // Gaussian: smooth pulse centered at 10.0 seconds, width 3.0 seconds, peak 0.1 mol/L
    let gaussian = TemporalInjection::gaussian(10.0, 3.0, 0.8);

    // Rectangle : continuous injection for a certain period of time

    let rectangle = TemporalInjection::rectangle(5.0, 15.0, 0.6);

    println!("═══════════════════════════════════════════════════════\n");
    println!("Injection Profiles:");
    println!("  1. Dirac    : δ(t-5)");
    println!("  2. Gaussian : 0.1·exp(-(t-10)²/(2·3²)) mol/L\n");

    // ====== Build injection × solver matrix ======

    let injections = vec![
        ("Dirac", dirac),
        ("Gaussian", gaussian),
        ("Rectangle", rectangle),
    ];

    let solvers: Vec<(&str, Box<dyn Solver>)> = vec![
        ("Euler", Box::new(EulerSolver::new())),
        ("RK4", Box::new(RK4Solver::new())),
    ];

    let configuration = SolverConfiguration::time_evolution(total_time, time_steps);

    let mut results: Vec<(&str, &str, f64, f64, f64, f64, SimulationResult)> = Vec::new();

    println!("═══════════════════════════════════════════════════════");
    println!("  Running Simulations: 2 Injections × 2 Solvers");
    println!("═══════════════════════════════════════════════════════\n");

    for (injection_name, injection) in &injections {
        for (solver_name, solver) in &solvers {
            println!("  Injection {} × Solver {}:", injection_name, solver_name);
            std::io::Write::flush(&mut std::io::stdout())?;

            let current_time = Instant::now();

            // Create physical model
            let model = Box::new(LangmuirSingle::new(
                lambda,
                langmuir_k,
                port_number,
                porosity,
                velocity,
                column_length,
                n_points,
                injection.clone(),
            ));

            // Setup scenario
            let initial = model.setup_initial_state();
            let boundaries = DomainBoundaries::temporal(initial);
            let scenario = Scenario::new(model, boundaries);

            // Solve
            let result = solver.solve(&scenario, &configuration)?;

            let elapsed_time = current_time.elapsed().as_secs_f64();

            // Extract outlet concentrations
            let outlet: Vec<f64> = result
                .state_trajectory
                .iter()
                .map(|state| {
                    match state.get(PhysicalQuantity::Concentration).unwrap() {
                        chrom_rs::physics::PhysicalData::Vector(v) => v[n_points - 1],
                        _ => 0.0
                    }
                })
                .collect();

            // Analyze peak
            let max_concentration = outlet.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let peak_index = outlet
                .iter()
                .position(|&c| (c - max_concentration).abs() < 1e-10)
                .unwrap_or(0);
            let retention_time = result.time_points[peak_index];
            let final_concentration = *outlet.last().unwrap();

            println!("✓ {:.2}s", elapsed_time);

            results.push((
                *injection_name,
                *solver_name,
                elapsed_time,
                max_concentration,
                retention_time,
                final_concentration,
                result,
            ));
        }
    }

    // ====== Results Analysis ======

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Results: Peak Characteristics");
    println!("═══════════════════════════════════════════════════════\n");

    println!("{:<10} {:<14} {:>12} {:>12} {:>12}",
             "Injection", "Solver", "Peak (mol/L)", "Ret.Time (s)", "Final (mol/L)");
    println!("{:-<60}", "");

    for (inj, solver, _, peak, rt, final_c, _) in &results {
        println!("{:<10} {:<14} {:>12.6} {:>12.2} {:>12.6}",
                 inj, solver, peak, rt, final_c);
    }

    // ====== Performance Comparison ======

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Performance Comparison");
    println!("═══════════════════════════════════════════════════════\n");

    println!("{:<10} {:<14} {:>12}", "Injection", "Solver", "Time (s)");
    println!("{:-<35}", "");

    for (inj, solver, elapsed, _, _, _, _) in &results {
        println!("{:<10} {:<14} {:>12.2}", inj, solver, elapsed);
    }

    let euler_dirac = results.iter().find(|(i, s, _, _, _, _, _)| *i == "Dirac" && *s == "Euler").unwrap();
    let rk4_dirac = results.iter().find(|(i, s, _, _, _, _, _)| *i == "Dirac" && *s == "RK4").unwrap();
    let speedup = rk4_dirac.2 / euler_dirac.2;

    println!("\nRK4/Euler Speedup: {:.2}x slower (expected ~4x)", speedup);

    // ====== Accuracy: Euler vs RK4 ======

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Accuracy: Euler vs RK4");
    println!("═══════════════════════════════════════════════════════\n");

    for injection_method in ["Dirac", "Gaussian"] {
        let euler_results = results
            .iter()
            .find(|(i, s, _, _, _, _, _)| *i == injection_method && *s == "Euler")
            .unwrap();
        let rk4_results = results
            .iter()
            .find(|(i, s, _, _, _, _, _)| *i == injection_method && *s == "RK4")
            .unwrap();

        let peak_difference = (rk4_results.3 - euler_results.3).abs();
        let peak_percentage = (peak_difference / euler_results.3) * 100.0;
        let retention_difference = (rk4_results.4 - euler_results.4).abs();
        let retention_percentage = (retention_difference / euler_results.4) * 1000000.0;

        println!("\n {} Injection:", injection_method);
        println!("  Peak difference    : {:.6} mol/L ({:.2}%)",
                 peak_difference, peak_percentage);
        println!("  Ret.Time difference: {:.2} s ({:.2} ppm)",
                 retention_difference, retention_percentage);
    }

    // ====== Peak Validation ======

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Peak Validation");
    println!("═══════════════════════════════════════════════════════\n");

    for (injection, solver, _, peak, _, final_concentration, _) in &results {
        let is_peak = *peak > 1e-03 && *final_concentration < *peak * 0.1;
        let status = if is_peak { "✅ PEAK" } else { "❌ PLATEAU" };
        println!("{:<14} {:<8} : {}", injection, solver, status);
    }

    // ====== Generate Plots ======

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Generating Plots");
    println!("═══════════════════════════════════════════════════════\n");

    for (injection, solver, _, _, _, _, result) in &results {
        let filename = format!("highload_{}_{}.png",
                               injection.to_lowercase(),
                               solver.to_lowercase());
        let path = tmp_dir.join(&filename);
        let draw_features = PlotConfig::chromatogram(format!(
            "High-Load: {} × {}",
            injection,
            solver
        ));
        plot_chromatogram(result, n_points, path.to_str().unwrap(), Some(&draw_features))?;
        println!("  {} × {} : {:?}", injection, solver, path);
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Comparison with TFA");
    println!("═══════════════════════════════════════════════════════\n");
    println!("Parameter differences:");
    println!("  λ: 0.8 vs 1.2 (33% lower → more retention)");
    println!("  K̃: 1.2 vs 0.4 (3× higher → stronger adsorption)");
    println!("  N: 1.5 vs 2.0 (25% lower → easier saturation)");
    println!("\nExpected effects:");
    println!("  ✅ Longer retention time");
    println!("  ✅ Peak asymmetry (tailing) due to saturation");
    println!("  ✅ Broader peaks");

    Ok(())
}