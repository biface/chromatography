//! Δt convergence visualisation (issue #55).
//!
//! Run with `cargo run --release --example stiffness_convergence` then
//! `cargo run --release --bin plot_stiffness_convergence`; output lands in
//! `target/plots/stiffness_convergence.svg`.
//!
//! Reads `stiffness_convergence.json` from the system temp directory — the
//! same file `examples/stiffness_convergence.rs` writes to.
//!
//! # Visual elements
//!
//! 1. **Two curves, log-log**: $R_{sf}$(Euler, RK4) vs Δt, one per case
//! 2. **Markers at each swept resolution** (×1, ×2, ×5, ×10, ×25, ×50 the
//!    baseline `n_steps`)
//!
//! A case whose curve stays high over a wider Δt range before dropping is
//! the stiffer one — it needs a finer step to reach the same Euler/RK4
//! agreement the other case reaches at a coarser step.
//!
//! # Cargo.toml
//!
//! ```toml
//! [[bin]]
//! name = "plot_stiffness_convergence"
//! path = "tools/plot_stiffness_convergence.rs"
//!
//! [dependencies]
//! plotters   = "0.3"
//! serde_json = "1"
//! anyhow     = "1"
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use plotters::prelude::*;
use serde_json::Value;

// =================================================================================================
// Data
// =================================================================================================

struct CasePoint {
    delta_t_s: f64,
    rsf: f64,
}

fn report_path() -> PathBuf {
    std::env::temp_dir().join("stiffness_convergence.json")
}

fn read_report(path: &Path) -> anyhow::Result<Vec<(String, Vec<CasePoint>)>> {
    let content = fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "cannot read {}: {e}\nRun first: cargo run --release --example stiffness_convergence",
            path.display()
        )
    })?;
    let root: Value = serde_json::from_str(&content)?;
    let obj = root
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{}: expected a top-level JSON object", path.display()))?;

    let mut cases: Vec<(String, Vec<CasePoint>)> = obj
        .iter()
        .map(|(name, points)| {
            let points = points
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("case '{name}': expected an array of points"))?
                .iter()
                .map(|p| {
                    let delta_t_s =
                        p.get("delta_t_s").and_then(Value::as_f64).ok_or_else(|| {
                            anyhow::anyhow!("case '{name}': point missing 'delta_t_s'")
                        })?;
                    let rsf = p
                        .get("rsf_euler_vs_rk4")
                        .and_then(Value::as_f64)
                        .ok_or_else(|| {
                            anyhow::anyhow!("case '{name}': point missing 'rsf_euler_vs_rk4'")
                        })?;
                    Ok(CasePoint { delta_t_s, rsf })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok((name.clone(), points))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Sort by name for a deterministic legend/color order across runs.
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    // Sort each case's points by Δt ascending, so lines draw left to right.
    for (_, points) in &mut cases {
        points.sort_by(|a, b| a.delta_t_s.partial_cmp(&b.delta_t_s).unwrap());
    }

    Ok(cases)
}

// =================================================================================================
// Plot generation
// =================================================================================================

fn generate_plot(cases: &[(String, Vec<CasePoint>)], output_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let all_dt: Vec<f64> = cases
        .iter()
        .flat_map(|(_, p)| p.iter().map(|pt| pt.delta_t_s))
        .collect();
    let all_rsf: Vec<f64> = cases
        .iter()
        .flat_map(|(_, p)| p.iter().map(|pt| pt.rsf))
        .collect();

    // Manual log10 transform for both axes — no reliance on `.log_scale()`,
    // which isn't used anywhere else in this codebase and can't be verified
    // to compile against the pinned `plotters` version without a local
    // build. `x_log`/`y_log` are what actually get plotted; tick labels are
    // formatted back to the real (non-log) value below.
    let dt_min = all_dt.iter().cloned().fold(f64::INFINITY, f64::min);
    let dt_max = all_dt.iter().cloned().fold(0.0f64, f64::max);
    let rsf_min = all_rsf
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .max(1e-6);
    let rsf_max = all_rsf.iter().cloned().fold(0.0f64, f64::max);

    let x_log_min = dt_min.log10() - 0.15;
    let x_log_max = dt_max.log10() + 0.15;
    let y_log_min = rsf_min.log10() - 0.2;
    let y_log_max = rsf_max.log10() + 0.2;

    let root = SVGBackend::new(output_path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(50)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .caption(
            "Rsf(Euler, RK4) vs step size — stiffness convergence",
            ("sans-serif", 18).into_font(),
        )
        .build_cartesian_2d(x_log_min..x_log_max, y_log_min..y_log_max)?;

    chart
        .configure_mesh()
        .x_desc("Δt (s, log scale)")
        .y_desc("Rsf(Euler, RK4) (log scale)")
        .x_label_formatter(&|x| format!("{:.4}", 10f64.powf(*x)))
        .y_label_formatter(&|y| format!("{:.4}", 10f64.powf(*y)))
        .draw()?;

    let colors = [RED, BLUE, RGBColor(0, 140, 0), RGBColor(148, 0, 211)];

    for (i, (name, points)) in cases.iter().enumerate() {
        let color = colors[i % colors.len()];
        let series: Vec<(f64, f64)> = points
            .iter()
            .map(|p| (p.delta_t_s.log10(), p.rsf.log10()))
            .collect();

        chart
            .draw_series(LineSeries::new(
                series.iter().copied(),
                color.stroke_width(2),
            ))?
            .label(name.replace('_', " "))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });

        chart.draw_series(
            series
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 4, color.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Plot generated: {}", output_path.display());
    Ok(())
}

// =================================================================================================
// Entry point
// =================================================================================================

fn main() -> anyhow::Result<()> {
    let path = report_path();
    println!("Reading {}...", path.display());
    let cases = read_report(&path)?;

    for (name, points) in &cases {
        println!("\n{name}");
        println!("  {:>14} {:>14}", "delta_t_s", "Rsf(Euler,RK4)");
        for p in points {
            println!("  {:>14.6} {:>14.5}", p.delta_t_s, p.rsf);
        }
    }

    let output_path = PathBuf::from("target/plots/stiffness_convergence.svg");
    println!("\nGenerating plot...");
    generate_plot(&cases, &output_path)?;
    Ok(())
}
