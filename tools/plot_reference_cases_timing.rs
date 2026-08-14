//! End-to-end solve time visualisation for the real reference cases (issue #55).
//!
//! Run with `cargo bench --bench langmuir_performance -- bench_reference_cases`
//! then `cargo run --bin plot_reference_cases_timing --release`; output lands
//! in `target/plots/reference_cases_timing.svg`.
//!
//! # Visual elements
//!
//! 1. **Grouped bars**: Euler (blue) and RK4 (red) mean solve time per case,
//!    with 95% CI error bars
//! 2. **Ratio annotation**: RK4/Euler ratio printed above each case group
//!
//! # Cargo.toml
//!
//! ```toml
//! [[bin]]
//! name = "plot_reference_cases_timing"
//! path = "tools/plot_reference_cases_timing.rs"
//!
//! [dependencies]
//! plotters   = "0.3"
//! serde      = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! anyhow     = "1"
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use plotters::prelude::*;
use serde::Deserialize;

// =================================================================================================
// Constants
// =================================================================================================

/// Case names, in the order they should appear on the x-axis — must match
/// `RefCase::name` in `benches/langmuir_performance.rs`.
const CASES: [&str; 2] = ["ascorbic_erythorbic", "glucose_fructose_linear"];

// =================================================================================================
// Criterion JSON deserialisation — same shape as `plot_parallelism_threshold.rs`
// =================================================================================================

#[derive(Debug, Deserialize)]
struct ConfidenceInterval {
    lower_bound: f64,
    upper_bound: f64,
}

#[derive(Debug, Deserialize)]
struct Estimate {
    confidence_interval: ConfidenceInterval,
    point_estimate: f64,
}

#[derive(Debug, Deserialize)]
struct Estimates {
    mean: Estimate,
}

// =================================================================================================
// Data
// =================================================================================================

/// One case's measured Euler and RK4 solve time (milliseconds).
struct CaseTiming {
    case: &'static str,
    euler_ms: f64,
    euler_low_ms: f64,
    euler_high_ms: f64,
    rk4_ms: f64,
    rk4_low_ms: f64,
    rk4_high_ms: f64,
}

fn read_estimates(path: &Path) -> anyhow::Result<Estimates> {
    let content = fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "cannot read {}: {e}\nRun first: cargo bench --bench langmuir_performance -- bench_reference_cases",
            path.display()
        )
    })?;
    Ok(serde_json::from_str(&content)?)
}

fn collect_timings(criterion_dir: &Path) -> anyhow::Result<Vec<CaseTiming>> {
    let group_dir = criterion_dir.join("bench_reference_cases");
    if !group_dir.exists() {
        anyhow::bail!(
            "Criterion directory not found: {}\nRun first: \
             cargo bench --bench langmuir_performance -- bench_reference_cases",
            group_dir.display()
        );
    }

    let mut timings = Vec::with_capacity(CASES.len());
    for &case in &CASES {
        let euler = read_estimates(
            &group_dir
                .join("euler")
                .join(case)
                .join("new")
                .join("estimates.json"),
        )?;
        let rk4 = read_estimates(
            &group_dir
                .join("rk4")
                .join(case)
                .join("new")
                .join("estimates.json"),
        )?;

        timings.push(CaseTiming {
            case,
            euler_ms: euler.mean.point_estimate / 1e6,
            euler_low_ms: euler.mean.confidence_interval.lower_bound / 1e6,
            euler_high_ms: euler.mean.confidence_interval.upper_bound / 1e6,
            rk4_ms: rk4.mean.point_estimate / 1e6,
            rk4_low_ms: rk4.mean.confidence_interval.lower_bound / 1e6,
            rk4_high_ms: rk4.mean.confidence_interval.upper_bound / 1e6,
        });
    }

    Ok(timings)
}

// =================================================================================================
// Plot generation
// =================================================================================================

fn generate_plot(timings: &[CaseTiming], output_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let y_max = timings
        .iter()
        .map(|t| t.euler_high_ms.max(t.rk4_high_ms))
        .fold(0.0_f64, f64::max)
        * 1.25;

    let root = SVGBackend::new(output_path, (1000, 650)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(50)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .caption(
            "Reference cases — Euler vs RK4 end-to-end solve time",
            ("sans-serif", 18).into_font(),
        )
        .build_cartesian_2d(0f64..timings.len() as f64, 0f64..y_max)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_desc("")
        .y_desc("Mean solve time (ms)")
        .x_label_formatter(&|_| String::new())
        .y_label_formatter(&|v| format!("{v:.0} ms"))
        .draw()?;

    // Bar width and per-case slot: each case occupies [i, i+1), Euler in the
    // left half, RK4 in the right half, with a small gap between the two.
    let bar_half_width = 0.35;

    for (i, t) in timings.iter().enumerate() {
        let x0 = i as f64;
        let euler_center = x0 + 0.5 - 0.22;
        let rk4_center = x0 + 0.5 + 0.22;

        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (euler_center - bar_half_width / 2.0, 0.0),
                (euler_center + bar_half_width / 2.0, t.euler_ms),
            ],
            BLUE.mix(0.7).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (euler_center, t.euler_low_ms),
                (euler_center, t.euler_high_ms),
            ],
            ShapeStyle {
                color: BLUE.mix(0.9).to_rgba(),
                filled: false,
                stroke_width: 2,
            },
        )))?;

        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (rk4_center - bar_half_width / 2.0, 0.0),
                (rk4_center + bar_half_width / 2.0, t.rk4_ms),
            ],
            RED.mix(0.7).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(rk4_center, t.rk4_low_ms), (rk4_center, t.rk4_high_ms)],
            ShapeStyle {
                color: RED.mix(0.9).to_rgba(),
                filled: false,
                stroke_width: 2,
            },
        )))?;

        // Case name, below the axis.
        chart.draw_series(std::iter::once(Text::new(
            t.case.replace('_', " "),
            (x0 + 0.5, -y_max * 0.03),
            ("sans-serif", 13).into_font().color(&BLACK),
        )))?;

        // Ratio annotation, above the taller bar.
        let ratio = t.rk4_ms / t.euler_ms;
        let top = t.euler_high_ms.max(t.rk4_high_ms);
        chart.draw_series(std::iter::once(Text::new(
            format!("×{ratio:.2}"),
            (x0 + 0.5, top + y_max * 0.04),
            ("sans-serif", 13).into_font().color(&BLACK),
        )))?;
    }

    // Legend (manual — no data series carries a `.label()` in this layout).
    chart.draw_series(std::iter::once(Rectangle::new(
        [
            (0.02 * timings.len() as f64, y_max * 0.94),
            (0.06 * timings.len() as f64, y_max * 0.98),
        ],
        BLUE.mix(0.7).filled(),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "Euler",
        (0.08 * timings.len() as f64, y_max * 0.96),
        ("sans-serif", 12).into_font(),
    )))?;
    chart.draw_series(std::iter::once(Rectangle::new(
        [
            (0.22 * timings.len() as f64, y_max * 0.94),
            (0.26 * timings.len() as f64, y_max * 0.98),
        ],
        RED.mix(0.7).filled(),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "RK4",
        (0.28 * timings.len() as f64, y_max * 0.96),
        ("sans-serif", 12).into_font(),
    )))?;

    root.present()?;
    println!("Plot generated: {}", output_path.display());
    Ok(())
}

// =================================================================================================
// Entry point
// =================================================================================================

fn main() -> anyhow::Result<()> {
    // Standalone binary variant (Variant B) — see
    // `chrom_rs::output::visualization::fonts` for why this call is needed
    // before any chart renders.
    chrom_rs::output::register_fonts();

    let criterion_dir = PathBuf::from("target/criterion");
    let output_path = PathBuf::from("target/plots/reference_cases_timing.svg");

    println!("Reading Criterion data...");
    let timings = collect_timings(&criterion_dir)?;

    println!(
        "\n{:<24} {:>12} {:>12} {:>10}",
        "case", "euler (ms)", "rk4 (ms)", "ratio"
    );
    println!("{:-<60}", "");
    for t in &timings {
        println!(
            "{:<24} {:>12.3} {:>12.3} {:>9.2}×",
            t.case,
            t.euler_ms,
            t.rk4_ms,
            t.rk4_ms / t.euler_ms
        );
    }

    println!("\nGenerating plot...");
    generate_plot(&timings, &output_path)?;
    Ok(())
}

// =================================================================================================
// Unit tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cases_list_matches_bench_group() {
        // Sanity check only — the real source of truth is
        // `benches/langmuir_performance.rs`'s `RefCase::ascorbic_erythorbic`/
        // `glucose_fructose_linear`. This just guards against a silent typo
        // in `CASES` above.
        assert_eq!(CASES, ["ascorbic_erythorbic", "glucose_fructose_linear"]);
    }
}
