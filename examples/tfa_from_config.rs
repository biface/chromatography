//! Example: TFA Chromatography — config-file variant
//!
//! Reproduces [`tfa`](tfa.rs) using the three-file configuration layout
//! defined in DD-015. The physical parameters and simulation results are
//! numerically identical to the direct-API variant; only the setup differs.
//!
//! ## Configuration files used
//!
//! ```text
//! examples/config/tfa/
//! ├── model.yml            ← LangmuirSingle column parameters
//! ├── scenario_dirac.yml   ← Dirac injection at t = 5 s
//! ├── scenario_gaussian.yml← Gaussian injection centered at t = 10 s
//! ├── solver_rk4.yml       ← RK4, 600 s, 10 000 steps
//! └── solver_euler.yml     ← Euler, 600 s, 10 000 steps
//! ```
//!
//! ## How to run
//!
//! ```bash
//! cargo run --example tfa_from_config
//! ```
//!
//! ## Expected output
//!
//! Peak characteristics and retention times identical to `tfa` example.
//! Plots written to the system temporary directory.

use chrom_rs::{
    config::{model::load_model, scenario::load_scenario, solver::load_solver},
    output::{PlotConfig, plot_chromatogram},
    physics::PhysicalQuantity,
    solver::{EulerSolver, RK4Solver, Scenario, SimulationResult, Solver},
};

use std::time::Instant;

// ── Config file paths ─────────────────────────────────────────────────────────

const MODEL: &str = "examples/config/tfa/model.yml";
const SCENARIO_DIRAC: &str = "examples/config/tfa/scenario_dirac.yml";
const SCENARIO_GAUSSIAN: &str = "examples/config/tfa/scenario_gaussian.yml";
const SOLVER_RK4: &str = "examples/config/tfa/solver_rk4.yml";
const SOLVER_EULER: &str = "examples/config/tfa/solver_euler.yml";

// ── Run descriptor ────────────────────────────────────────────────────────────

struct Run {
    injection_name: &'static str,
    solver_name: &'static str,
    elapsed_secs: f64,
    peak_concentration: f64,
    retention_time_secs: f64,
    final_concentration: f64,
    result: SimulationResult,
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  TFA Chromatography — config-file variant (DD-015)");
    println!("═══════════════════════════════════════════════════════\n");

    let tmp_dir = std::env::temp_dir();

    // ── Simulation matrix: 2 scenarios × 2 solvers ───────────────────────────

    let scenarios = [("Dirac", SCENARIO_DIRAC), ("Gaussian", SCENARIO_GAUSSIAN)];

    let solver_files = [("Euler", SOLVER_EULER), ("Runge-Kutta", SOLVER_RK4)];

    println!("═══════════════════════════════════════════════════════");
    println!("  Running Simulations: 2 Scenarios × 2 Solvers");
    println!("═══════════════════════════════════════════════════════\n");

    let mut runs: Vec<Run> = Vec::new();

    for (injection_name, scenario_path) in &scenarios {
        for (solver_name, solver_path) in &solver_files {
            print!("  {injection_name:8} × {solver_name:12}: ");
            std::io::Write::flush(&mut std::io::stdout())?;

            let t0 = Instant::now();

            // 1. Load model (injection set to None in model.yml)
            let mut model = load_model(MODEL)?;

            // 2. Load scenario — applies injection to the model in place
            let boundaries = load_scenario(scenario_path, &mut *model)?;

            // 3. Load solver configuration
            let solver_cfg = load_solver(solver_path)?;

            // 4. Build scenario and solve
            let n_points = model.points();
            let scenario = Scenario::new(model, boundaries);

            let result = match solver_cfg.solver_name.as_str() {
                "RK4" => RK4Solver::new().solve(&scenario, &solver_cfg.config)?,
                "Euler" => EulerSolver::new().solve(&scenario, &solver_cfg.config)?,
                other => return Err(format!("unknown solver '{other}'").into()),
            };

            let elapsed_secs = t0.elapsed().as_secs_f64();

            // 5. Extract outlet concentration series (last spatial point)
            let outlet: Vec<f64> = result
                .state_trajectory
                .iter()
                .map(
                    |state| match state.get(PhysicalQuantity::Concentration).unwrap() {
                        chrom_rs::physics::PhysicalData::Vector(v) => v[n_points - 1],
                        _ => 0.0,
                    },
                )
                .collect();

            let peak_concentration = outlet.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let peak_index = outlet
                .iter()
                .position(|&c| (c - peak_concentration).abs() < 1e-10)
                .unwrap_or(0);
            let retention_time_secs = result.time_points[peak_index];
            let final_concentration = *outlet.last().unwrap_or(&0.0);

            println!("✓ {elapsed_secs:.2} s");

            runs.push(Run {
                injection_name,
                solver_name,
                elapsed_secs,
                peak_concentration,
                retention_time_secs,
                final_concentration,
                result,
            });
        }
    }

    // ── Results ───────────────────────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Results: Peak Characteristics");
    println!("═══════════════════════════════════════════════════════\n");

    println!(
        "{:<10} {:<14} {:>12} {:>12} {:>12}",
        "Injection", "Solver", "Peak (mol/L)", "Ret.Time (s)", "Final (mol/L)"
    );
    println!("{:-<60}", "");

    for r in &runs {
        println!(
            "{:<10} {:<14} {:>12.6} {:>12.2} {:>12.6}",
            r.injection_name,
            r.solver_name,
            r.peak_concentration,
            r.retention_time_secs,
            r.final_concentration,
        );
    }

    // ── Performance ───────────────────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Performance Comparison");
    println!("═══════════════════════════════════════════════════════\n");

    println!("{:<10} {:<14} {:>10}", "Injection", "Solver", "Time (s)");
    println!("{:-<35}", "");

    for r in &runs {
        println!(
            "{:<10} {:<14} {:>10.2}",
            r.injection_name, r.solver_name, r.elapsed_secs
        );
    }

    let euler_dirac = runs
        .iter()
        .find(|r| r.injection_name == "Dirac" && r.solver_name == "Euler")
        .unwrap();
    let rk4_dirac = runs
        .iter()
        .find(|r| r.injection_name == "Dirac" && r.solver_name == "Runge-Kutta")
        .unwrap();
    println!(
        "\nRK4/Euler speedup: {:.2}× slower (expected ~4×)",
        rk4_dirac.elapsed_secs / euler_dirac.elapsed_secs
    );

    // ── Accuracy: Euler vs RK4 ────────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Accuracy: Euler vs RK4");
    println!("═══════════════════════════════════════════════════════\n");

    for injection_name in ["Dirac", "Gaussian"] {
        let euler = runs
            .iter()
            .find(|r| r.injection_name == injection_name && r.solver_name == "Euler")
            .unwrap();
        let rk4 = runs
            .iter()
            .find(|r| r.injection_name == injection_name && r.solver_name == "Runge-Kutta")
            .unwrap();

        let peak_diff = (rk4.peak_concentration - euler.peak_concentration).abs();
        let peak_pct = peak_diff / euler.peak_concentration * 100.0;
        let rt_diff = (rk4.retention_time_secs - euler.retention_time_secs).abs();
        let rt_ppm = rt_diff / euler.retention_time_secs * 1_000_000.0;

        println!(" {injection_name} Injection:");
        println!("  Peak difference    : {peak_diff:.6} mol/L ({peak_pct:.2}%)");
        println!("  Ret.Time difference: {rt_diff:.4} s ({rt_ppm:.2} ppm)");
    }

    // ── Peak Validation ───────────────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Peak Validation");
    println!("═══════════════════════════════════════════════════════\n");

    for r in &runs {
        let is_peak =
            r.peak_concentration > 1e-3 && r.final_concentration < r.peak_concentration * 0.1;
        let status = if is_peak { "✅ PEAK" } else { "❌ PLATEAU" };
        println!("{:<14} {:<12}: {status}", r.injection_name, r.solver_name);
    }

    // ── Plots ─────────────────────────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Generating Plots");
    println!("═══════════════════════════════════════════════════════\n");

    for r in &runs {
        let filename = format!("tfa_config_{}_{}.png", r.injection_name, r.solver_name);
        let path = tmp_dir.join(&filename);
        let config = PlotConfig::chromatogram(format!(
            "TFA [config]: {} × {}",
            r.injection_name, r.solver_name
        ));
        plot_chromatogram(&r.result, 100, path.to_str().unwrap(), Some(&config))?;
        println!("  {} × {} : {path:?}", r.injection_name, r.solver_name);
    }

    Ok(())
}
