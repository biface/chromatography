//! Example: Ascorbic / Erythorbic Acid — Solo and Competitive Study
//!
//! Mirrors the structure of `tfa.rs` for two acids across two solvers,
//! then extends it with a competitive (multi-species) phase to quantify
//! the effect of adsorption competition.
//!
//! ## Structure
//!
//! **Phase 1 — Solo** (`LangmuirSingle`, 2 species × 2 solvers = 4 runs)
//! - One `plot_chromatogram` per run (species × solver)
//! - One `plot_chromatograms_comparison` per solver (both species overlaid)
//!
//! **Phase 2 — Competitive** (`LangmuirMulti`, 2 solvers = 2 runs)
//! - One `plot_chromatogram_multi` per solver
//!
//! **Phase 3 — Analysis**
//! - Peak characteristics, performance, Euler vs RK4 accuracy, peak validation
//! - Competition effect: retention time shift (solo → mixed)
//!
//! **Parameters** (Nicoud 2015, Figure 5):
//! - Ascorbic acid  : λ=1.0, K̃=1.1 L/mol, N=2 — lower affinity → elutes first
//! - Erythorbic acid: λ=1.0, K̃=1.7 L/mol, N=2 — higher affinity → elutes later
//! - ε=0.4, u=0.001 m/s, L=0.25 m, nz=100, T=1200 s

use chrom_rs::{
    models::{LangmuirMulti, LangmuirSingle, SpeciesParams, TemporalInjection},
    output::export::{CsvExporter, Exporter},
    output::{plot_chromatogram, plot_chromatogram_multi, plot_chromatograms_comparison, PlotConfig},
    physics::{PhysicalData, PhysicalModel, PhysicalQuantity},
    solver::{
        DomainBoundaries, EulerSolver, RK4Solver, Scenario, SimulationResult,
        Solver, SolverConfiguration,
    },
};

use std::time::Instant;

// =============================================================================
// Data structures
// =============================================================================

/// Result of one solo (single-species) simulation run.
struct SoloRun {
    species_name:        &'static str,
    solver_name:         &'static str,
    elapsed_secs:        f64,
    peak_concentration:  f64,
    retention_time_secs: f64,
    final_concentration: f64,
    result:              SimulationResult,
}

/// Result of one competitive (multi-species) simulation run.
struct MultiRun {
    solver_name:         &'static str,
    elapsed_secs:        f64,
    /// Peak concentration per species [mol/L]
    peak_concentrations: Vec<f64>,
    /// Retention time per species [s]
    retention_times:     Vec<f64>,
    result:              SimulationResult,
}

// =============================================================================
// Helpers
// =============================================================================

/// Prints a titled section banner to stdout.
fn print_section(title: &str) {
    println!("\n═══════════════════════════════════════════════════════");
    println!("  {title}");
    println!("═══════════════════════════════════════════════════════\n");
}

/// Extracts the outlet concentration series from a single-species result.
///
/// The outlet is the last element of the `Vector` state (last spatial point).
fn outlet_single(result: &SimulationResult, n_points: usize) -> Vec<f64> {
    result
        .state_trajectory
        .iter()
        .map(|state| match state.get(PhysicalQuantity::Concentration).unwrap() {
            PhysicalData::Vector(v) => v[n_points - 1],
            _ => 0.0,
        })
        .collect()
}

/// Extracts per-species outlet concentrations from a multi-species result.
///
/// The state is a `[n_points × n_species]` matrix; outlet = last row.
/// Returns `[species][time_step]`.
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

/// Returns (peak_concentration, retention_time, final_concentration) from an outlet series.
fn peak_stats(outlet: &[f64], time_points: &[f64]) -> (f64, f64, f64) {
    let peak = outlet.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let idx  = outlet
        .iter()
        .position(|&c| (c - peak).abs() < 1e-12)
        .unwrap_or(0);
    let rt    = time_points[idx];
    let final_c = *outlet.last().unwrap_or(&0.0);
    (peak, rt, final_c)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {

    print_section("Ascorbic / Erythorbic Acid — Solo and Competitive Study");

    // ====== Column parameters ======

    let n_points      = 100;
    let porosity      = 0.4;
    let velocity      = 0.001;
    let column_length = 0.25;

    // ====== Species parameters ======

    // λ, K̃, N (f64 for LangmuirSingle)
    let params = [
        ("Ascorbic",   1.0_f64, 1.1_f64, 2.0_f64),  // lower K̃ → elutes first
        ("Erythorbic", 1.0_f64, 1.7_f64, 2.0_f64),  // higher K̃ → elutes later
    ];
    let species_names: &[&str] = &["Ascorbic", "Erythorbic"];
    let n_species = params.len();

    println!("Species:");
    for (name, lambda, k, n) in &params {
        println!("  {name:<12} λ={lambda}  K̃={k} L/mol  N={n}");
    }
    println!("\nColumn:");
    println!("  ε={porosity}  u={velocity} m/s  L={column_length} m  nz={n_points}");

    // ====== Simulation configuration ======

    // 1200 s: long enough for erythorbic (K̃=1.7) to fully elute.
    // time_steps scaled to keep the same dt as the TFA example.
    let total_time = 1200.0;
    let time_steps = 20_000;

    println!("\nSimulation:");
    println!("  Total time : {total_time} s ({} min)", total_time / 60.0);
    println!("  Time steps : {time_steps}");
    println!("  dt         : {:.6} s", total_time / time_steps as f64);

    // ====== Shared Gaussian injection (same profile for both species) ======

    let injection_center = 10.0; // s
    let injection_sigma  = 3.0;  // s
    let injection_peak   = 0.1;  // mol/L

    println!("\nInjection: Gaussian — center={injection_center} s  σ={injection_sigma} s  peak={injection_peak} mol/L");

    let configuration = SolverConfiguration::time_evolution(total_time, time_steps);
    let tmp_dir       = std::env::temp_dir();
    let exporter      = CsvExporter::default();

    let solver_defs: Vec<(&'static str, Box<dyn Solver>)> = vec![
        ("Euler",       Box::new(EulerSolver::new())),
        ("Runge-Kutta", Box::new(RK4Solver::new())),
    ];

    // =========================================================================
    // Phase 1 — Solo runs (LangmuirSingle, 2 species × 2 solvers)
    // =========================================================================

    print_section("Phase 1 — Solo Runs: 2 Species × 2 Solvers");

    let mut solo_runs: Vec<SoloRun> = Vec::new();

    for (species_name, lambda, langmuir_k, port_number) in &params {
        for (solver_name, solver) in &solver_defs {
            print!("  {species_name} × {solver_name} … ");
            std::io::Write::flush(&mut std::io::stdout())?;

            let timer = Instant::now();

            let model = LangmuirSingle::new(
                *lambda,
                *langmuir_k,
                *port_number,
                porosity,
                velocity,
                column_length,
                n_points,
                TemporalInjection::gaussian(injection_center, injection_sigma, injection_peak),
            );

            let initial    = model.setup_initial_state();
            let boundaries = DomainBoundaries::temporal(initial);
            let scenario   = Scenario::new(Box::new(model), boundaries);
            let result     = solver.solve(&scenario, &configuration)?;

            let elapsed_secs = timer.elapsed().as_secs_f64();
            let outlet       = outlet_single(&result, n_points);
            let (peak, rt, final_c) = peak_stats(&outlet, &result.time_points);

            println!("✓ {:.2} s", elapsed_secs);

            // Individual chromatogram: species × solver
            let png_name = format!("acids_solo_{}_{}.png", species_name, solver_name);
            let png_path = tmp_dir.join(&png_name);
            let cfg = PlotConfig::chromatogram(format!("{species_name} — {solver_name}"));
            plot_chromatogram(&result, n_points, png_path.to_str().unwrap(), Some(&cfg))?;
            println!("    plot → {:?}", png_path);

            // CSV
            let csv_name = format!("acids_solo_{}_{}.csv", species_name, solver_name);
            let csv_path = tmp_dir.join(&csv_name);
            exporter.export_single(&result, None, csv_path.to_str().unwrap())?;
            println!("    csv  → {:?}", csv_path);

            solo_runs.push(SoloRun {
                species_name,
                solver_name,
                elapsed_secs,
                peak_concentration:  peak,
                retention_time_secs: rt,
                final_concentration: final_c,
                result,
            });
        }
    }

    // Comparison plots — one per solver (both species overlaid)
    for solver_name in ["Euler", "Runge-Kutta"] {
        let datasets: Vec<(&str, &SimulationResult, usize)> = solo_runs
            .iter()
            .filter(|r| r.solver_name == solver_name)
            .map(|r| (r.species_name, &r.result, n_points))
            .collect();

        let cmp_name = format!("acids_solo_comparison_{solver_name}.png");
        let cmp_path = tmp_dir.join(&cmp_name);
        let cfg = PlotConfig::chromatogram(
            format!("Ascorbic / Erythorbic — Solo, {solver_name}"),
        );
        plot_chromatograms_comparison(datasets, cmp_path.to_str().unwrap(), Some(&cfg))?;
        println!("\n  Comparison ({solver_name}) → {:?}", cmp_path);
    }

    // =========================================================================
    // Phase 2 — Competitive runs (LangmuirMulti, 2 solvers)
    // =========================================================================

    print_section("Phase 2 — Competitive Runs: LangmuirMulti × 2 Solvers");

    let mut multi_runs: Vec<MultiRun> = Vec::new();

    for (solver_name, solver) in &solver_defs {
        print!("  Competitive × {solver_name} … ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let timer = Instant::now();

        // Build one SpeciesParams per acid (N is u32 for LangmuirMulti)
        let species_vec: Vec<SpeciesParams> = params
            .iter()
            .map(|(name, lambda, k, n)| {
                SpeciesParams::new(
                    *name,
                    *lambda,
                    *k,
                    *n as u32,
                    TemporalInjection::gaussian(injection_center, injection_sigma, injection_peak),
                )
            })
            .collect();

        let model      = LangmuirMulti::new(species_vec, n_points, porosity, velocity, column_length)?;
        let initial    = model.setup_initial_state();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario   = Scenario::new(Box::new(model), boundaries);
        let result     = solver.solve(&scenario, &configuration)?;

        let elapsed_secs = timer.elapsed().as_secs_f64();
        let per_species  = outlet_multi(&result, n_points, n_species);

        let mut peak_concentrations = Vec::with_capacity(n_species);
        let mut retention_times     = Vec::with_capacity(n_species);
        for s in 0..n_species {
            let (peak, rt, _) = peak_stats(&per_species[s], &result.time_points);
            peak_concentrations.push(peak);
            retention_times.push(rt);
        }

        println!("✓ {:.2} s", elapsed_secs);

        // Multi-species chromatogram
        let png_name = format!("acids_multi_{solver_name}.png");
        let png_path = tmp_dir.join(&png_name);
        let cfg = PlotConfig::chromatogram(
            format!("Ascorbic / Erythorbic — Competitive, {solver_name}"),
        );
        plot_chromatogram_multi(&result, n_points, species_names, png_path.to_str().unwrap(), Some(&cfg))?;
        println!("    plot → {:?}", png_path);

        // CSV
        let csv_name = format!("acids_multi_{solver_name}.csv");
        let csv_path = tmp_dir.join(&csv_name);
        exporter.export_multi(&result, None, species_names, csv_path.to_str().unwrap())?;
        println!("    csv  → {:?}", csv_path);

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

    // ── Solo: peak characteristics ────────────────────────────────────────────

    print_section("Results — Solo: Peak Characteristics");

    println!("{:<12} {:<14} {:>12} {:>12} {:>12}",
             "Species", "Solver", "Peak (mol/L)", "Ret.Time (s)", "Final (mol/L)");
    println!("{:-<65}", "");

    for run in &solo_runs {
        println!("{:<12} {:<14} {:>12.6} {:>12.1} {:>12.6}",
                 run.species_name,
                 run.solver_name,
                 run.peak_concentration,
                 run.retention_time_secs,
                 run.final_concentration);
    }

    // ── Multi: peak characteristics ───────────────────────────────────────────

    print_section("Results — Competitive: Peak Characteristics");

    println!("{:<12} {:<14} {:>12} {:>12}",
             "Species", "Solver", "Peak (mol/L)", "Ret.Time (s)");
    println!("{:-<55}", "");

    for run in &multi_runs {
        for s in 0..n_species {
            println!("{:<12} {:<14} {:>12.6} {:>12.1}",
                     species_names[s],
                     run.solver_name,
                     run.peak_concentrations[s],
                     run.retention_times[s]);
        }
        println!("{:-<55}", "");
    }

    // ── Performance ───────────────────────────────────────────────────────────

    print_section("Performance Comparison");

    println!("{:<12} {:<14} {:>12}", "Phase", "Solver", "Time (s)");
    println!("{:-<40}", "");

    for run in &solo_runs {
        // Print one row per species×solver (each is an independent solve)
        println!("{:<12} {:<14} {:>12.3}",
                 format!("Solo-{}", run.species_name), run.solver_name, run.elapsed_secs);
    }
    for run in &multi_runs {
        println!("{:<12} {:<14} {:>12.3}", "Competitive", run.solver_name, run.elapsed_secs);
    }

    // RK4/Euler ratio on the competitive runs (most representative)
    if multi_runs.len() >= 2 {
        let ratio = multi_runs[1].elapsed_secs / multi_runs[0].elapsed_secs;
        println!("\nRK4/Euler ratio (competitive): {:.2}× slower (expected ~4×)", ratio);
    }

    // ── Accuracy: Euler vs RK4 ────────────────────────────────────────────────

    print_section("Accuracy: Euler vs RK4");

    println!("— Solo —");
    for species_name in species_names {
        let euler = solo_runs.iter().find(|r| r.species_name == *species_name && r.solver_name == "Euler").unwrap();
        let rk4   = solo_runs.iter().find(|r| r.species_name == *species_name && r.solver_name == "Runge-Kutta").unwrap();

        let peak_diff = (rk4.peak_concentration - euler.peak_concentration).abs();
        let peak_pct  = (peak_diff / euler.peak_concentration) * 100.0;
        let rt_diff   = (rk4.retention_time_secs - euler.retention_time_secs).abs();
        let rt_ppm    = (rt_diff / euler.retention_time_secs) * 1_000_000.0;

        println!("\n  {species_name}:");
        println!("    Peak difference     : {:.6} mol/L ({:.2}%)", peak_diff, peak_pct);
        println!("    Ret. time difference: {:.4} s ({:.2} ppm)", rt_diff, rt_ppm);
    }

    if multi_runs.len() >= 2 {
        println!("\n— Competitive —");
        let euler = &multi_runs[0];
        let rk4   = &multi_runs[1];

        for s in 0..n_species {
            let peak_diff = (rk4.peak_concentrations[s] - euler.peak_concentrations[s]).abs();
            let peak_pct  = (peak_diff / euler.peak_concentrations[s]) * 100.0;
            let rt_diff   = (rk4.retention_times[s] - euler.retention_times[s]).abs();
            let rt_ppm    = (rt_diff / euler.retention_times[s]) * 1_000_000.0;

            println!("\n  {} (competitive):", species_names[s]);
            println!("    Peak difference     : {:.6} mol/L ({:.2}%)", peak_diff, peak_pct);
            println!("    Ret. time difference: {:.4} s ({:.2} ppm)", rt_diff, rt_ppm);
        }
    }

    // ── Peak validation ───────────────────────────────────────────────────────

    print_section("Peak Validation");

    println!("— Solo —");
    for run in &solo_runs {
        let is_peak = run.peak_concentration > 1e-4
            && run.final_concentration < run.peak_concentration * 0.1;
        let status = if is_peak { "✅ PEAK" } else { "❌ PLATEAU / INCOMPLETE" };
        println!("  {:<12} {:<14} : {}", run.species_name, run.solver_name, status);
    }

    println!("\n— Competitive —");
    for run in &multi_runs {
        for s in 0..n_species {
            let peak   = run.peak_concentrations[s];
            // Retrieve final outlet value for this species
            let per_sp = outlet_multi(&run.result, n_points, n_species);
            let final_c = *per_sp[s].last().unwrap_or(&0.0);
            let is_peak = peak > 1e-4 && final_c < peak * 0.1;
            let status = if is_peak { "✅ PEAK" } else { "❌ PLATEAU / INCOMPLETE" };
            println!("  {:<12} {:<14} : {}", species_names[s], run.solver_name, status);
        }
    }

    // ── Competition effect ────────────────────────────────────────────────────

    print_section("Competition Effect — Retention Time Shift (Euler)");

    // Compare solo Euler vs multi Euler (same solver → no numerical bias)
    let euler_multi = multi_runs.iter().find(|r| r.solver_name == "Euler").unwrap();

    println!("{:<12} {:>12} {:>12} {:>12}",
             "Species", "Solo (s)", "Mixed (s)", "Shift (s)");
    println!("{:-<50}", "");

    for s in 0..n_species {
        let solo_euler = solo_runs.iter()
            .find(|r| r.species_name == species_names[s] && r.solver_name == "Euler")
            .unwrap();
        let rt_solo  = solo_euler.retention_time_secs;
        let rt_mixed = euler_multi.retention_times[s];
        let shift    = rt_mixed - rt_solo;

        println!("{:<12} {:>12.1} {:>12.1} {:>+12.1}", species_names[s], rt_solo, rt_mixed, shift);
    }

    println!();
    for s in 0..n_species {
        let solo_euler = solo_runs.iter()
            .find(|r| r.species_name == species_names[s] && r.solver_name == "Euler")
            .unwrap();
        let rt_solo  = solo_euler.retention_time_secs;
        let rt_mixed = euler_multi.retention_times[s];
        let shift    = rt_mixed - rt_solo;

        let effect = match shift {
            sh if sh.abs() < 1.0 => "no significant shift (competition negligible at these concentrations)",
            sh if sh > 0.0       => "elutes LATER in mixture (competitor occupies shared sites)",
            _                    => "elutes EARLIER in mixture (displaced by stronger competitor)",
        };
        println!("  {} : {}", species_names[s], effect);
    }

    println!("\nDone.");
    Ok(())
}
