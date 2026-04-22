//! Example: Ascorbic / Erythorbic Acid — config-file variant
//!
//! Reproduces [`acids_multi`](acids_multi.rs) using the three-file configuration
//! layout defined in DD-015. The physical parameters and simulation results are
//! numerically identical to the direct-API variant; only the setup differs.
//!
//! The solo phase (Phase 1) derives its `LangmuirSingle` instances from
//! the shared `model.yml` (LangmuirMulti) — no per-species model file needed.
//!
//! ## Configuration files used
//!
//! ```text
//! examples/config/acids/
//! ├── model.yml             ← LangmuirMulti — parameters for all species
//! ├── scenario_gaussian.yml ← Gaussian injection shared by all phases
//! ├── solver_rk4.yml        ← RK4,   1200 s, 20 000 steps
//! └── solver_euler.yml      ← Euler, 1200 s, 20 000 steps
//! ```
//!
//! ## How to run
//!
//! ```bash
//! cargo run --example acids_from_config
//! ```

use chrom_rs::{
    config::{model::load_model, scenario::load_scenario, solver::load_solver},
    models::{LangmuirMulti, LangmuirSingle, TemporalInjection},
    output::export::{CsvExporter, Exporter},
    output::{
        PlotConfig, plot_chromatogram, plot_chromatogram_multi, plot_chromatograms_comparison,
    },
    physics::{PhysicalData, PhysicalModel, PhysicalQuantity},
    solver::{EulerSolver, RK4Solver, Scenario, SimulationResult, Solver},
};

use std::time::Instant;

// ── Config file paths ─────────────────────────────────────────────────────────

const MODEL_MULTI: &str = "examples/config/acids/model.yml";
const SCENARIO: &str = "examples/config/acids/scenario_gaussian.yml";
const SOLVER_RK4: &str = "examples/config/acids/solver_rk4.yml";
const SOLVER_EULER: &str = "examples/config/acids/solver_euler.yml";

// ── Data structures ───────────────────────────────────────────────────────────

struct SoloRun {
    species_name: String,
    solver_name: &'static str,
    elapsed_secs: f64,
    peak_concentration: f64,
    retention_time_secs: f64,
    final_concentration: f64,
    result: SimulationResult,
}

struct MultiRun {
    solver_name: &'static str,
    elapsed_secs: f64,
    peak_concentrations: Vec<f64>,
    retention_times: Vec<f64>,
    result: SimulationResult,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn print_section(title: &str) {
    println!("\n═══════════════════════════════════════════════════════");
    println!("  {title}");
    println!("═══════════════════════════════════════════════════════\n");
}

fn outlet_single(result: &SimulationResult, n_points: usize) -> Vec<f64> {
    result
        .state_trajectory
        .iter()
        .map(
            |state| match state.get(PhysicalQuantity::Concentration).unwrap() {
                PhysicalData::Vector(v) => v[n_points - 1],
                _ => 0.0,
            },
        )
        .collect()
}

fn outlet_multi(result: &SimulationResult, n_points: usize, n_species: usize) -> Vec<Vec<f64>> {
    let mut per_species: Vec<Vec<f64>> = vec![Vec::new(); n_species];
    for state in &result.state_trajectory {
        if let Some(PhysicalData::Matrix(m)) = state.get(PhysicalQuantity::Concentration) {
            for s in 0..n_species {
                per_species[s].push(m[(n_points - 1, s)]);
            }
        }
    }
    per_species
}

fn peak_stats(outlet: &[f64], time_points: &[f64]) -> (f64, f64, f64) {
    let peak = outlet.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let idx = outlet
        .iter()
        .position(|&c| (c - peak).abs() < 1e-12)
        .unwrap_or(0);
    let rt = time_points[idx];
    let final_c = *outlet.last().unwrap_or(&0.0);
    (peak, rt, final_c)
}

/// Deserialises model.yml into a concrete `LangmuirMulti` to access species
/// parameters and column geometry for the solo phase and the header display.
fn load_multi_model(path: &str) -> Result<LangmuirMulti, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let value: serde_yaml::Value = serde_yaml::from_str(&content)?;
    let inner = value
        .get("LangmuirMulti")
        .ok_or("missing 'LangmuirMulti' key in model file")?
        .clone();
    let model: LangmuirMulti = serde_yaml::from_value(inner)?;
    Ok(model)
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    print_section("Ascorbic / Erythorbic Acid — config-file variant (DD-015)");

    // ── Read model parameters for display and solo phase construction ─────────
    let multi_params = load_multi_model(MODEL_MULTI)?;
    let n_points = multi_params.spatial_points();
    let porosity = multi_params.porosity();
    let velocity = multi_params.velocity();
    let col_len = multi_params.column_length();
    let species = multi_params.species_params();
    let n_species = species.len();
    let species_names: Vec<&str> = multi_params.species_names();

    // Read solver config for display (use RK4 as reference)
    let solver_cfg_ref = load_solver(SOLVER_RK4)?;
    let total_time = match solver_cfg_ref.config.solver_type {
        chrom_rs::solver::SolverType::TimeEvolution {
            total_time,
            time_steps: _,
        } => total_time,
        _ => 1200.0,
    };
    let time_steps = match solver_cfg_ref.config.solver_type {
        chrom_rs::solver::SolverType::TimeEvolution {
            total_time: _,
            time_steps,
        } => time_steps,
        _ => 20_000,
    };

    // ── Header ────────────────────────────────────────────────────────────────
    println!("Species:");
    for sp in species {
        println!(
            "  {:<12} λ={}  K̃={} L/mol  N={}",
            sp.name, sp.lambda, sp.langmuir_k, sp.port_number
        );
    }
    println!("\nColumn:");
    println!("  ε={porosity}  u={velocity} m/s  L={col_len} m  nz={n_points}");
    println!("\nSimulation:");
    println!("  Total time : {total_time} s ({} min)", total_time / 60.0);
    println!("  Time steps : {time_steps}");
    println!("  dt         : {:.6} s", total_time / time_steps as f64);
    println!("\nInjection: Gaussian — center=10 s  σ=3 s  peak=0.1 mol/L");

    let tmp_dir = std::env::temp_dir();
    let exporter = CsvExporter::default();

    let solver_defs: Vec<(&'static str, &'static str, Box<dyn Solver>)> = vec![
        ("Euler", SOLVER_EULER, Box::new(EulerSolver::new())),
        ("Runge-Kutta", SOLVER_RK4, Box::new(RK4Solver::new())),
    ];

    // =========================================================================
    // Phase 1 — Solo (LangmuirSingle derived from model.yml, 2 species × 2 solvers)
    // =========================================================================

    print_section("Phase 1 — Solo Runs: 2 Species × 2 Solvers");

    let mut solo_runs: Vec<SoloRun> = Vec::new();

    for sp in species {
        for (solver_name, solver_path, _solver) in &solver_defs {
            print!("  {} × {solver_name} … ", sp.name);
            std::io::Write::flush(&mut std::io::stdout())?;

            let t0 = Instant::now();

            // Build LangmuirSingle from species params + shared column geometry
            let mut model: Box<dyn PhysicalModel> = Box::new(LangmuirSingle::new(
                sp.lambda,
                sp.langmuir_k,
                sp.port_number as f64,
                porosity,
                velocity,
                col_len,
                n_points,
                TemporalInjection::none(),
            ));

            // Apply injection from scenario file
            let boundaries = load_scenario(SCENARIO, &mut *model)?;
            let solver_cfg = load_solver(solver_path)?;
            let scenario = Scenario::new(model, boundaries);

            let result = match solver_cfg.solver_name.as_str() {
                "RK4" => RK4Solver::new().solve(&scenario, &solver_cfg.config)?,
                "Euler" => EulerSolver::new().solve(&scenario, &solver_cfg.config)?,
                other => return Err(format!("unknown solver '{other}'").into()),
            };

            let elapsed_secs = t0.elapsed().as_secs_f64();
            let outlet = outlet_single(&result, n_points);
            let (peak, rt, final_c) = peak_stats(&outlet, &result.time_points);

            println!("✓ {elapsed_secs:.2} s");

            let png = tmp_dir.join(format!("acids_config_solo_{}_{solver_name}.png", sp.name));
            let cfg = PlotConfig::chromatogram(format!("{} — {solver_name}", sp.name));
            plot_chromatogram(&result, n_points, png.to_str().unwrap(), Some(&cfg))?;
            println!("    plot → {png:?}");

            let csv = tmp_dir.join(format!("acids_config_solo_{}_{solver_name}.csv", sp.name));
            exporter.export_single(&result, None, csv.to_str().unwrap())?;
            println!("    csv  → {csv:?}");

            solo_runs.push(SoloRun {
                species_name: sp.name.clone(),
                solver_name,
                elapsed_secs,
                peak_concentration: peak,
                retention_time_secs: rt,
                final_concentration: final_c,
                result,
            });
        }
    }

    // Comparison plots — one per solver
    for &(solver_name, _, _) in &solver_defs {
        let datasets: Vec<(&str, &SimulationResult, usize)> = solo_runs
            .iter()
            .filter(|r| r.solver_name == solver_name)
            .map(|r| (r.species_name.as_str(), &r.result, n_points))
            .collect();

        let cmp = tmp_dir.join(format!("acids_config_solo_comparison_{solver_name}.png"));
        let cfg = PlotConfig::chromatogram(format!("Ascorbic / Erythorbic — Solo, {solver_name}"));
        plot_chromatograms_comparison(datasets, cmp.to_str().unwrap(), Some(&cfg))?;
        println!("\n  Comparison ({solver_name}) → {cmp:?}");
    }

    // =========================================================================
    // Phase 2 — Competitive (LangmuirMulti, 2 solvers)
    // =========================================================================

    print_section("Phase 2 — Competitive Runs: LangmuirMulti × 2 Solvers");

    let mut multi_runs: Vec<MultiRun> = Vec::new();

    for &(solver_name, solver_path, ref solver) in &solver_defs {
        print!("  Competitive × {solver_name} … ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let t0 = Instant::now();

        let mut model = load_model(MODEL_MULTI)?;
        let boundaries = load_scenario(SCENARIO, &mut *model)?;
        let solver_cfg = load_solver(solver_path)?;
        let scenario = Scenario::new(model, boundaries);

        let result = solver.solve(&scenario, &solver_cfg.config)?;
        let elapsed_secs = t0.elapsed().as_secs_f64();
        let per_species = outlet_multi(&result, n_points, n_species);

        let mut peak_concentrations = Vec::with_capacity(n_species);
        let mut retention_times = Vec::with_capacity(n_species);
        for s in 0..n_species {
            let (peak, rt, _) = peak_stats(&per_species[s], &result.time_points);
            peak_concentrations.push(peak);
            retention_times.push(rt);
        }

        println!("✓ {elapsed_secs:.2} s");

        let png = tmp_dir.join(format!("acids_config_multi_{solver_name}.png"));
        let cfg = PlotConfig::chromatogram(format!(
            "Ascorbic / Erythorbic — Competitive, {solver_name}"
        ));
        plot_chromatogram_multi(
            &result,
            n_points,
            &species_names,
            png.to_str().unwrap(),
            Some(&cfg),
        )?;
        println!("    plot → {png:?}");

        let csv = tmp_dir.join(format!("acids_config_multi_{solver_name}.csv"));
        exporter.export_multi(&result, None, &species_names, csv.to_str().unwrap())?;
        println!("    csv  → {csv:?}");

        multi_runs.push(MultiRun {
            solver_name,
            elapsed_secs,
            peak_concentrations,
            retention_times,
            result,
        });
    }

    // =========================================================================
    // Phase 3 — Analysis
    // =========================================================================

    print_section("Results — Solo: Peak Characteristics");

    println!(
        "{:<12} {:<14} {:>12} {:>12} {:>12}",
        "Species", "Solver", "Peak (mol/L)", "Ret.Time (s)", "Final (mol/L)"
    );
    println!("{:-<65}", "");
    for r in &solo_runs {
        println!(
            "{:<12} {:<14} {:>12.6} {:>12.1} {:>12.6}",
            r.species_name,
            r.solver_name,
            r.peak_concentration,
            r.retention_time_secs,
            r.final_concentration
        );
    }

    print_section("Results — Competitive: Peak Characteristics");

    println!(
        "{:<12} {:<14} {:>12} {:>12}",
        "Species", "Solver", "Peak (mol/L)", "Ret.Time (s)"
    );
    println!("{:-<55}", "");
    for r in &multi_runs {
        for s in 0..n_species {
            println!(
                "{:<12} {:<14} {:>12.6} {:>12.1}",
                species_names[s], r.solver_name, r.peak_concentrations[s], r.retention_times[s]
            );
        }
        println!("{:-<55}", "");
    }

    print_section("Performance Comparison");

    println!("{:<18} {:<14} {:>10}", "Phase", "Solver", "Time (s)");
    println!("{:-<45}", "");
    for r in &solo_runs {
        println!(
            "{:<18} {:<14} {:>10.3}",
            format!("Solo-{}", r.species_name),
            r.solver_name,
            r.elapsed_secs
        );
    }
    for r in &multi_runs {
        println!(
            "{:<18} {:<14} {:>10.3}",
            "Competitive", r.solver_name, r.elapsed_secs
        );
    }
    if multi_runs.len() >= 2 {
        println!(
            "\nRK4/Euler ratio (competitive): {:.2}× slower (expected ~4×)",
            multi_runs[1].elapsed_secs / multi_runs[0].elapsed_secs
        );
    }

    print_section("Accuracy: Euler vs RK4");

    println!("— Solo —");
    for species_name in &species_names {
        let euler = solo_runs
            .iter()
            .find(|r| r.species_name == *species_name && r.solver_name == "Euler")
            .unwrap();
        let rk4 = solo_runs
            .iter()
            .find(|r| r.species_name == *species_name && r.solver_name == "Runge-Kutta")
            .unwrap();

        let peak_diff = (rk4.peak_concentration - euler.peak_concentration).abs();
        let peak_pct = peak_diff / euler.peak_concentration * 100.0;
        let rt_diff = (rk4.retention_time_secs - euler.retention_time_secs).abs();
        let rt_ppm = rt_diff / euler.retention_time_secs * 1_000_000.0;

        println!("\n  {species_name}:");
        println!("    Peak difference     : {peak_diff:.6} mol/L ({peak_pct:.2}%)");
        println!("    Ret. time difference: {rt_diff:.4} s ({rt_ppm:.2} ppm)");
    }

    if multi_runs.len() >= 2 {
        println!("\n— Competitive —");
        let euler = &multi_runs[0];
        let rk4 = &multi_runs[1];
        for s in 0..n_species {
            let peak_diff = (rk4.peak_concentrations[s] - euler.peak_concentrations[s]).abs();
            let peak_pct = peak_diff / euler.peak_concentrations[s] * 100.0;
            let rt_diff = (rk4.retention_times[s] - euler.retention_times[s]).abs();
            let rt_ppm = rt_diff / euler.retention_times[s] * 1_000_000.0;
            println!("\n  {} (competitive):", species_names[s]);
            println!("    Peak difference     : {peak_diff:.6} mol/L ({peak_pct:.2}%)");
            println!("    Ret. time difference: {rt_diff:.4} s ({rt_ppm:.2} ppm)");
        }
    }

    print_section("Peak Validation");

    println!("— Solo —");
    for r in &solo_runs {
        let is_peak =
            r.peak_concentration > 1e-4 && r.final_concentration < r.peak_concentration * 0.1;
        let status = if is_peak {
            "✅ PEAK"
        } else {
            "❌ PLATEAU / INCOMPLETE"
        };
        println!("  {:<12} {:<14} : {status}", r.species_name, r.solver_name);
    }

    println!("\n— Competitive —");
    for r in &multi_runs {
        let per_sp = outlet_multi(&r.result, n_points, n_species);
        for s in 0..n_species {
            let peak = r.peak_concentrations[s];
            let final_c = *per_sp[s].last().unwrap_or(&0.0);
            let is_peak = peak > 1e-4 && final_c < peak * 0.1;
            let status = if is_peak {
                "✅ PEAK"
            } else {
                "❌ PLATEAU / INCOMPLETE"
            };
            println!(
                "  {:<12} {:<14} : {status}",
                species_names[s], r.solver_name
            );
        }
    }

    print_section("Competition Effect — Retention Time Shift (Euler)");

    let euler_multi = multi_runs
        .iter()
        .find(|r| r.solver_name == "Euler")
        .unwrap();

    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Species", "Solo (s)", "Mixed (s)", "Shift (s)"
    );
    println!("{:-<50}", "");
    for s in 0..n_species {
        let solo_euler = solo_runs
            .iter()
            .find(|r| r.species_name == species_names[s] && r.solver_name == "Euler")
            .unwrap();
        let shift = euler_multi.retention_times[s] - solo_euler.retention_time_secs;
        println!(
            "{:<12} {:>12.1} {:>12.1} {:>+12.1}",
            species_names[s], solo_euler.retention_time_secs, euler_multi.retention_times[s], shift
        );
    }

    println!();
    for s in 0..n_species {
        let solo_euler = solo_runs
            .iter()
            .find(|r| r.species_name == species_names[s] && r.solver_name == "Euler")
            .unwrap();
        let shift = euler_multi.retention_times[s] - solo_euler.retention_time_secs;
        let effect = match shift {
            sh if sh.abs() < 1.0 => {
                "no significant shift (competition negligible at these concentrations)"
            }
            sh if sh > 0.0 => "elutes LATER in mixture (competitor occupies shared sites)",
            _ => "elutes EARLIER in mixture (displaced by stronger competitor)",
        };
        println!("  {} : {effect}", species_names[s]);
    }

    println!("\nDone.");
    Ok(())
}
