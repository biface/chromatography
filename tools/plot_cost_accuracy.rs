//! Cost-vs-accuracy visualisation: Euler/RK4 solve time against
//! $R_{sf}$(Euler, RK4), per validated case (issue #55).
//!
//! Run with `cargo run --example validation_report` then
//! `cargo run --bin plot_cost_accuracy --release`; output lands in
//! `target/plots/cost_accuracy.svg`.
//!
//! Reads `validation_report.json` from the system temp directory — the same
//! file `examples/validation_report.rs` writes to, and the same path
//! (`std::env::temp_dir()`) it prints on completion.
//!
//! # Visual elements
//!
//! 1. **Grouped bars**: Euler (blue) and RK4 (red) mean solve time per case
//! 2. **$R_{sf}$ annotation**: printed above each case group — the accuracy
//!    "bought" by the extra RK4 cost
//!
//! # Cargo.toml
//!
//! ```toml
//! [[bin]]
//! name = "plot_cost_accuracy"
//! path = "tools/plot_cost_accuracy.rs"
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

/// One case's timing and accuracy figures, read from `validation_report.json`.
struct CaseCostAccuracy {
    case: String,
    euler_ms: f64,
    rk4_ms: f64,
    rsf: f64,
}

fn report_path() -> PathBuf {
    std::env::temp_dir().join("validation_report.json")
}

fn read_report(path: &Path) -> anyhow::Result<Vec<CaseCostAccuracy>> {
    let content = fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "cannot read {}: {e}\nRun first: cargo run --example validation_report",
            path.display()
        )
    })?;
    let root: Value = serde_json::from_str(&content)?;

    let cases = root.get("cases").and_then(Value::as_array).ok_or_else(|| {
        anyhow::anyhow!(
            "{}: missing or malformed top-level 'cases' array",
            path.display()
        )
    })?;

    cases
        .iter()
        .map(|c| {
            let case = c
                .get("case")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("case entry missing 'case' name"))?
                .to_string();
            let rsf = c
                .get("rsf_euler_vs_rk4")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow::anyhow!("case '{case}' missing 'rsf_euler_vs_rk4'"))?;
            let timing = c
                .get("solve_time_ms")
                .ok_or_else(|| anyhow::anyhow!("case '{case}' missing 'solve_time_ms' — regenerate the report with the current validation_report.rs"))?;
            let euler_ms = timing
                .get("euler")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow::anyhow!("case '{case}' missing 'solve_time_ms.euler'"))?;
            let rk4_ms = timing
                .get("rk4")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow::anyhow!("case '{case}' missing 'solve_time_ms.rk4'"))?;
            Ok(CaseCostAccuracy { case, euler_ms, rk4_ms, rsf })
        })
        .collect()
}

// =================================================================================================
// Plot generation
// =================================================================================================

fn generate_plot(data: &[CaseCostAccuracy], output_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let y_max = data
        .iter()
        .map(|d| d.euler_ms.max(d.rk4_ms))
        .fold(0.0_f64, f64::max)
        * 1.3;

    let root = SVGBackend::new(output_path, (1100, 650)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(50)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .caption(
            "Solve time vs Rsf(Euler, RK4) — cost of RK4's extra accuracy",
            ("sans-serif", 18).into_font(),
        )
        .build_cartesian_2d(0f64..data.len() as f64, 0f64..y_max)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_desc("")
        .y_desc("Mean solve time (ms)")
        .x_label_formatter(&|_| String::new())
        .y_label_formatter(&|v| format!("{v:.0} ms"))
        .draw()?;

    let bar_half_width = 0.35;

    for (i, d) in data.iter().enumerate() {
        let x0 = i as f64;
        let euler_center = x0 + 0.5 - 0.22;
        let rk4_center = x0 + 0.5 + 0.22;

        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (euler_center - bar_half_width / 2.0, 0.0),
                (euler_center + bar_half_width / 2.0, d.euler_ms),
            ],
            BLUE.mix(0.7).filled(),
        )))?;
        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (rk4_center - bar_half_width / 2.0, 0.0),
                (rk4_center + bar_half_width / 2.0, d.rk4_ms),
            ],
            RED.mix(0.7).filled(),
        )))?;

        chart.draw_series(std::iter::once(Text::new(
            d.case.replace('_', " "),
            (x0 + 0.5, -y_max * 0.03),
            ("sans-serif", 13).into_font().color(&BLACK),
        )))?;

        let top = d.euler_ms.max(d.rk4_ms);
        chart.draw_series(std::iter::once(Text::new(
            format!("Rsf = {:.4}", d.rsf),
            (x0 + 0.5, top + y_max * 0.05),
            ("sans-serif", 13).into_font().color(&RGBColor(80, 80, 80)),
        )))?;
    }

    // Legend (manual, same convention as `plot_reference_cases_timing.rs`).
    let w = data.len() as f64;
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.02 * w, y_max * 0.94), (0.06 * w, y_max * 0.98)],
        BLUE.mix(0.7).filled(),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "Euler",
        (0.08 * w, y_max * 0.96),
        ("sans-serif", 12).into_font(),
    )))?;
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.22 * w, y_max * 0.94), (0.26 * w, y_max * 0.98)],
        RED.mix(0.7).filled(),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "RK4",
        (0.28 * w, y_max * 0.96),
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
    let path = report_path();
    println!("Reading {}...", path.display());
    let data = read_report(&path)?;

    println!(
        "\n{:<24} {:>12} {:>12} {:>10}",
        "case", "euler (ms)", "rk4 (ms)", "Rsf"
    );
    println!("{:-<62}", "");
    for d in &data {
        println!(
            "{:<24} {:>12.3} {:>12.3} {:>10.4}",
            d.case, d.euler_ms, d.rk4_ms, d.rsf
        );
    }

    let output_path = PathBuf::from("target/plots/cost_accuracy.svg");
    println!("\nGenerating plot...");
    generate_plot(&data, &output_path)?;
    Ok(())
}

// =================================================================================================
// Unit tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_report_parses_expected_shape() {
        let json = r#"{
            "cases": [
                {
                    "case": "tfa_linear",
                    "rsf_euler_vs_rk4": 0.01,
                    "solve_time_ms": {"euler": 5.0, "rk4": 20.0, "rk4_over_euler_ratio": 4.0}
                }
            ]
        }"#;
        let path = std::env::temp_dir().join("chrom_rs_test_cost_accuracy.json");
        fs::write(&path, json).unwrap();
        let data = read_report(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(data.len(), 1);
        assert_eq!(data[0].case, "tfa_linear");
        assert!((data[0].euler_ms - 5.0).abs() < 1e-9);
        assert!((data[0].rk4_ms - 20.0).abs() < 1e-9);
        assert!((data[0].rsf - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_read_report_missing_solve_time_errors() {
        let json = r#"{"cases": [{"case": "x", "rsf_euler_vs_rk4": 0.01}]}"#;
        let path = std::env::temp_dir().join("chrom_rs_test_cost_accuracy_missing.json");
        fs::write(&path, json).unwrap();
        let result = read_report(&path);
        fs::remove_file(&path).ok();

        assert!(result.is_err());
    }
}
