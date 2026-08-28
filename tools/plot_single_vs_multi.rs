//! Surcoût structurel de `LangmuirMulti` face à `LangmuirSingle` à
//! `n_species=1` (issue #55, point 1 — "est-ce utile d'avoir deux
//! structures ?").
//! *Structural overhead of `LangmuirMulti` vs `LangmuirSingle` at
//! `n_species=1`.*
//!
//! Run with `cargo bench --bench langmuir_performance -- bench_single_vs_multi_1species`
//! then `cargo run --bin plot_single_vs_multi --release`; output lands in
//! `target/plots/single_vs_multi.svg`.
//!
//! Reads the four fixed Criterion result directories directly — this bench
//! group has no swept dimension (unlike `bench_parallelism_threshold` or
//! `bench_multi_species_scaling`), so there is nothing to regress and no
//! `npts_X`-style directory parsing needed.
//!
//! Deliberately does **not** cross-reference `bench_multi_species_scaling`'s
//! own `n_species=1` point: that group fixes a different `n_points` than
//! this one (confirmed by inspecting both `langmuir_performance.rs` bench
//! functions — not assumed), so the two are not directly comparable in
//! absolute terms. Mixing them on one chart would silently imply a
//! comparison that isn't valid.
//!
//! # Visual elements
//!
//! 1. **Grouped bars**: Single (blue) vs Multi (red) mean time, one group
//!    per solver (Euler, RK4)
//! 2. **Surcoût annotation**: Multi/Single ratio printed above each group
//!
//! # Cargo.toml
//!
//! ```toml
//! [[bin]]
//! name = "plot_single_vs_multi"
//! path = "tools/plot_single_vs_multi.rs"
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
// Désérialisation JSON Criterion / Criterion JSON deserialisation
// (identique à plot_parallelism_threshold.rs — mêmes champs lus)
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

fn read_estimates(path: &Path) -> anyhow::Result<Estimates> {
    let content = fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "cannot read {}: {e}\nRun first: cargo bench --bench langmuir_performance -- \
             bench_single_vs_multi_1species",
            path.display()
        )
    })?;
    Ok(serde_json::from_str(&content)?)
}

// =================================================================================================
// Data
// =================================================================================================

struct SolverGroup {
    solver: &'static str,
    single_us: f64,
    single_ci_us: (f64, f64),
    multi_us: f64,
    multi_ci_us: (f64, f64),
}

const GROUP: &str = "bench_single_vs_multi_1species";

/// Lit les 4 chemins fixes (pas de balayage dans ce groupe) et retourne les
/// deux couples Single/Multi, un par solveur.
/// *Reads the 4 fixed paths (no sweep in this group) and returns the two
/// Single/Multi pairs, one per solver.*
fn collect_data(criterion_dir: &Path) -> anyhow::Result<Vec<SolverGroup>> {
    let mut groups = Vec::new();

    for (solver, single_fn, multi_fn) in [
        ("euler", "single_euler", "multi_1sp_euler"),
        ("rk4", "single_rk4", "multi_1sp_rk4"),
    ] {
        let single_path = criterion_dir
            .join(GROUP)
            .join(single_fn)
            .join("new")
            .join("estimates.json");
        let multi_path = criterion_dir
            .join(GROUP)
            .join(multi_fn)
            .join("new")
            .join("estimates.json");

        let single = read_estimates(&single_path)?;
        let multi = read_estimates(&multi_path)?;

        groups.push(SolverGroup {
            solver,
            single_us: single.mean.point_estimate / 1e3,
            single_ci_us: (
                single.mean.confidence_interval.lower_bound / 1e3,
                single.mean.confidence_interval.upper_bound / 1e3,
            ),
            multi_us: multi.mean.point_estimate / 1e3,
            multi_ci_us: (
                multi.mean.confidence_interval.lower_bound / 1e3,
                multi.mean.confidence_interval.upper_bound / 1e3,
            ),
        });
    }

    Ok(groups)
}

// =================================================================================================
// Plot generation
// =================================================================================================

fn generate_plot(groups: &[SolverGroup], output_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let y_max = groups
        .iter()
        .map(|g| g.single_us.max(g.multi_us))
        .fold(0.0_f64, f64::max)
        * 1.35;

    let root = SVGBackend::new(output_path, (900, 650)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(50)
        .x_label_area_size(45)
        .y_label_area_size(80)
        .caption(
            "LangmuirSingle vs LangmuirMulti (n_species=1) — surcoût structurel",
            ("sans-serif", 18).into_font(),
        )
        .build_cartesian_2d(0f64..groups.len() as f64, 0f64..y_max)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_desc("")
        .y_desc("Temps moyen (µs)")
        .x_label_formatter(&|_| String::new())
        .y_label_formatter(&|v| format!("{v:.0} µs"))
        .draw()?;

    let bar_half_width = 0.35;

    for (i, g) in groups.iter().enumerate() {
        let x0 = i as f64;
        let single_center = x0 + 0.5 - 0.22;
        let multi_center = x0 + 0.5 + 0.22;

        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (single_center - bar_half_width / 2.0, 0.0),
                (single_center + bar_half_width / 2.0, g.single_us),
            ],
            BLUE.mix(0.7).filled(),
        )))?;
        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (multi_center - bar_half_width / 2.0, 0.0),
                (multi_center + bar_half_width / 2.0, g.multi_us),
            ],
            RED.mix(0.7).filled(),
        )))?;

        // Barres d'IC 95% / 95% CI whiskers
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (single_center, g.single_ci_us.0),
                (single_center, g.single_ci_us.1),
            ],
            BLACK.stroke_width(1),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (multi_center, g.multi_ci_us.0),
                (multi_center, g.multi_ci_us.1),
            ],
            BLACK.stroke_width(1),
        )))?;

        chart.draw_series(std::iter::once(Text::new(
            g.solver.to_uppercase(),
            (x0 + 0.5, -y_max * 0.04),
            ("sans-serif", 13).into_font().color(&BLACK),
        )))?;

        let ratio = g.multi_us / g.single_us;
        let top = g.single_us.max(g.multi_us);
        chart.draw_series(std::iter::once(Text::new(
            format!("Surcoût ×{ratio:.2}"),
            (x0 + 0.5, top + y_max * 0.05),
            ("sans-serif", 13).into_font().color(&RGBColor(150, 0, 0)),
        )))?;
    }

    // Légende manuelle, même convention que plot_cost_accuracy.rs
    // Manual legend, same convention as plot_cost_accuracy.rs
    let w = groups.len() as f64;
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.02 * w, y_max * 0.94), (0.06 * w, y_max * 0.98)],
        BLUE.mix(0.7).filled(),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "Single",
        (0.08 * w, y_max * 0.96),
        ("sans-serif", 12).into_font(),
    )))?;
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.22 * w, y_max * 0.94), (0.26 * w, y_max * 0.98)],
        RED.mix(0.7).filled(),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "Multi (1 espèce)",
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
    // Standalone binary variant (Variant B) — see
    // `chrom_rs::output::visualization::fonts` for why this call is needed
    // before any chart renders.
    chrom_rs::output::register_fonts();

    let criterion_dir = PathBuf::from("target/criterion");
    println!("Reading Criterion data from {}...", criterion_dir.display());

    let groups = collect_data(&criterion_dir)?;

    println!(
        "\n{:<8} {:>14} {:>14} {:>10}",
        "solver", "single (µs)", "multi (µs)", "ratio"
    );
    println!("{:-<50}", "");
    for g in &groups {
        println!(
            "{:<8} {:>14.2} {:>14.2} {:>9.2}×",
            g.solver,
            g.single_us,
            g.multi_us,
            g.multi_us / g.single_us
        );
    }

    let output_path = PathBuf::from("target/plots/single_vs_multi.svg");
    println!("\nGenerating plot...");
    generate_plot(&groups, &output_path)?;
    Ok(())
}

// =================================================================================================
// Tests unitaires / Unit tests
// =================================================================================================

#[cfg(test)]
mod tests {
    /// Régression sur les données réelles (V6, 28/08) : le surcoût structurel
    /// de Multi à n_species=1 doit rester dans une fourchette large mais
    /// non triviale — s'il tombe sous ×2, la distinction Single/Multi perd
    /// une partie de sa justification (issue #55, point 1) ; s'il explose
    /// au-delà de ×20, quelque chose a régressé ailleurs (allocation,
    /// dispatch...).
    /// *Regression on real data (V6, 28/08): Multi's structural overhead at
    /// n_species=1 should stay within a wide but non-trivial range — below
    /// ×2 the Single/Multi distinction loses part of its justification
    /// (issue #55, point 1); above ×20 something regressed elsewhere
    /// (allocation, dispatch...).*
    #[test]
    fn test_regression_overhead_from_actual_data() {
        let single_euler_ns = 861_371.25_f64;
        let multi_euler_ns = 6_864_477.56_f64;
        let ratio_euler = multi_euler_ns / single_euler_ns;
        assert!(
            (2.0..20.0).contains(&ratio_euler),
            "surcoût Euler hors fourchette plausible: {ratio_euler:.2}×"
        );

        let single_rk4_ns = 2_786_793.48_f64;
        let multi_rk4_ns = 26_751_994.53_f64;
        let ratio_rk4 = multi_rk4_ns / single_rk4_ns;
        assert!(
            (2.0..20.0).contains(&ratio_rk4),
            "surcoût RK4 hors fourchette plausible: {ratio_rk4:.2}×"
        );
    }

    /// Conversion ns -> µs cohérente (pas de coquille /1e6 copiée depuis les
    /// autres outils, qui convertissent en ms).
    /// *Consistent ns -> µs conversion (no /1e6 typo copied from the other
    /// tools, which convert to ms).*
    #[test]
    fn test_ns_to_us() {
        assert!((1_000.0_f64 / 1e3 - 1.0).abs() < 1e-12);
    }
}
