//! Diagramme travail-précision : ordre de convergence mesuré vs théorique
//! (issue #55, point 4 — "analyse par rapport aux grandeurs théoriques").
//! *Work-precision diagram: measured vs theoretical convergence order.*
//!
//! Run with `cargo run --release --example stiffness_convergence` then
//! `cargo run --release --bin plot_convergence_order -- [--case=NAME]`;
//! output lands in `target/plots/convergence_order_<case>.svg`.
//!
//! Reads `stiffness_convergence.json` from the system temp directory — the
//! same file `examples/stiffness_convergence.rs` writes to, and the same
//! file `plot_stiffness_convergence.rs` reads. This tool answers a
//! different question than that one: `plot_stiffness_convergence.rs` shows
//! `Rsf(Euler, RK4)` (Euler's error, RK4 taken as reference) across the 3
//! validation cases. This tool isolates a *single* case and adds RK4's own
//! self-convergence (`rsf_rk4_self_vs_next`) plus the two theoretical
//! reference slopes side by side, to answer: does RK4 actually show 4th
//! order convergence here, or not?
//!
//! # CLI
//!
//! `--case=<name>` selects the case (default: `ascorbic_erythorbic`).
//! Available names are whatever keys exist in `stiffness_convergence.json`
//! (currently `ascorbic_erythorbic`, `erythorbic_alone`,
//! `glucose_fructose_linear`) — an invalid name prints the available list
//! and exits, it does not fall back silently.
//!
//! # Visual elements
//!
//! 1. **Euler measured** (blue): `rsf_euler_vs_rk4` vs `n_steps`, log-log
//! 2. **RK4 measured** (orange): `rsf_rk4_self_vs_next` vs `n_steps`, log-log
//! 3. **Theoretical order-1 slope** (blue, dashed): anchored on Euler's
//!    first point, `err = err_0 × (n/n_0)^-1`
//! 4. **Theoretical order-4 slope** (orange, dashed): anchored on RK4's
//!    first point, `err = err_0 × (n/n_0)^-4`
//!
//! A method that matches its theoretical order tracks its own reference
//! slope. Euler is expected to. RK4's measured curve visibly failing to
//! approach its order-4 reference is the open question this tool exists
//! to keep visible, not to resolve — see the article's §4.
//!
//! # Cargo.toml
//!
//! ```toml
//! [[bin]]
//! name = "plot_convergence_order"
//! path = "tools/plot_convergence_order.rs"
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
// Constantes / Constants
// =================================================================================================

const DEFAULT_CASE: &str = "ascorbic_erythorbic";

// =================================================================================================
// Data
// =================================================================================================

/// Un palier de résolution temporelle pour un cas donné.
/// *One temporal-resolution step for a given case.*
#[derive(Debug, Clone)]
struct StepPoint {
    n_steps: usize,
    /// Erreur d'Euler relative à RK4 / *Euler's error relative to RK4*
    rsf_euler_vs_rk4: Option<f64>,
    /// Auto-convergence de RK4 / *RK4's own self-convergence*
    rsf_rk4_self_vs_next: Option<f64>,
}

fn report_path() -> PathBuf {
    std::env::temp_dir().join("stiffness_convergence.json")
}

/// Lit `stiffness_convergence.json` et retourne les noms de cas disponibles
/// ainsi que les points du cas demandé.
/// *Reads `stiffness_convergence.json` and returns the available case names
/// plus the requested case's points.*
fn read_case(path: &Path, case: &str) -> anyhow::Result<Vec<StepPoint>> {
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

    let Some(points) = obj.get(case) else {
        let available: Vec<&str> = obj.keys().map(String::as_str).collect();
        anyhow::bail!(
            "case '{case}' not found in {}.\nAvailable cases: {}",
            path.display(),
            available.join(", ")
        );
    };

    let mut points: Vec<StepPoint> = points
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("case '{case}': expected an array of points"))?
        .iter()
        .map(|p| {
            let n_steps = p
                .get("n_steps")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow::anyhow!("case '{case}': point missing 'n_steps'"))?
                as usize;
            let rsf_euler_vs_rk4 = p.get("rsf_euler_vs_rk4").and_then(Value::as_f64);
            let rsf_rk4_self_vs_next = p.get("rsf_rk4_self_vs_next").and_then(Value::as_f64);
            Ok(StepPoint {
                n_steps,
                rsf_euler_vs_rk4,
                rsf_rk4_self_vs_next,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    points.sort_by_key(|p| p.n_steps);
    Ok(points)
}

// =================================================================================================
// Pente théorique / Theoretical slope
// =================================================================================================

/// Génère une droite `err = err_0 × (n / n_0)^-order` ancrée sur le premier
/// point de la série fournie.
/// *Generates a line `err = err_0 × (n / n_0)^-order` anchored on the
/// series' first point.*
fn theoretical_slope(anchor_n: f64, anchor_err: f64, order: f64, ns: &[f64]) -> Vec<(f64, f64)> {
    ns.iter()
        .map(|&n| (n, anchor_err * (n / anchor_n).powf(-order)))
        .collect()
}

/// Ordre empirique entre deux points consécutifs :
/// `p = ln(err1/err2) / ln(n2/n1)`.
/// *Empirical order between two consecutive points.*
fn empirical_order(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    (p1.1 / p2.1).ln() / (p2.0 / p1.0).ln()
}

// =================================================================================================
// CLI
// =================================================================================================

/// Extrait `--case=NAME` (ou `--case NAME`) des arguments, sans dépendance
/// supplémentaire (aucun de ces outils n'utilise `dynamic-cli`, réservé à
/// la CLI applicative principale — voir `src/cli/`).
/// *Extracts `--case=NAME` (or `--case NAME`) from arguments, no extra
/// dependency (none of these tools use `dynamic-cli`, reserved for the main
/// application CLI — see `src/cli/`).*
fn parse_case_arg(args: &[String]) -> Option<String> {
    for (i, arg) in args.iter().enumerate() {
        if let Some(v) = arg.strip_prefix("--case=") {
            return Some(v.to_string());
        }
        if arg == "--case" {
            return args.get(i + 1).cloned();
        }
    }
    None
}

// =================================================================================================
// Plot generation
// =================================================================================================

fn generate_plot(case: &str, points: &[StepPoint], output_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let euler_xy: Vec<(f64, f64)> = points
        .iter()
        .filter_map(|p| p.rsf_euler_vs_rk4.map(|e| (p.n_steps as f64, e)))
        .collect();
    let rk4_xy: Vec<(f64, f64)> = points
        .iter()
        .filter_map(|p| p.rsf_rk4_self_vs_next.map(|e| (p.n_steps as f64, e)))
        .collect();

    anyhow::ensure!(euler_xy.len() >= 2, "case '{case}': not enough Euler points to plot");
    anyhow::ensure!(rk4_xy.len() >= 2, "case '{case}': not enough RK4 points to plot");

    let ref_order1 = theoretical_slope(
        euler_xy[0].0,
        euler_xy[0].1,
        1.0,
        &euler_xy.iter().map(|p| p.0).collect::<Vec<_>>(),
    );
    let ref_order4 = theoretical_slope(
        rk4_xy[0].0,
        rk4_xy[0].1,
        4.0,
        &rk4_xy.iter().map(|p| p.0).collect::<Vec<_>>(),
    );

    let all_n: Vec<f64> = euler_xy.iter().chain(&rk4_xy).map(|p| p.0).collect();
    let all_err: Vec<f64> = euler_xy
        .iter()
        .chain(&rk4_xy)
        .chain(&ref_order4)
        .map(|p| p.1)
        .collect();

    // Transformation log10 manuelle — même convention que
    // `plot_stiffness_convergence.rs` (pas de `.log_scale()`, non vérifiable
    // sans compilation locale contre la version de `plotters` figée ici).
    // *Manual log10 transform — same convention as
    // `plot_stiffness_convergence.rs` (no `.log_scale()`, unverifiable
    // without a local build against the pinned `plotters` version).*
    let n_min = all_n.iter().cloned().fold(f64::INFINITY, f64::min);
    let n_max = all_n.iter().cloned().fold(0.0f64, f64::max);
    let err_min = all_err.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-12);
    let err_max = all_err.iter().cloned().fold(0.0f64, f64::max);

    let x_log_min = n_min.log10() - 0.1;
    let x_log_max = n_max.log10() + 0.1;
    let y_log_min = err_min.log10() - 0.3;
    let y_log_max = err_max.log10() + 0.3;

    let root = SVGBackend::new(output_path, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(50)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .caption(
            format!("Diagramme travail-précision — {case} — ordre observé vs théorique"),
            ("sans-serif", 18).into_font(),
        )
        .build_cartesian_2d(x_log_min..x_log_max, y_log_min..y_log_max)?;

    chart
        .configure_mesh()
        .x_desc("n_steps (log)")
        .y_desc("Erreur (Rsf, log)")
        .x_label_formatter(&|x| format!("{:.0}", 10f64.powf(*x)))
        .y_label_formatter(&|y| format!("{:.2e}", 10f64.powf(*y)))
        .draw()?;

    let log_pts = |pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
        pts.iter().map(|&(n, e)| (n.log10(), e.log10())).collect()
    };

    let euler_log = log_pts(&euler_xy);
    let rk4_log = log_pts(&rk4_xy);
    let ref1_log = log_pts(&ref_order1);
    let ref4_log = log_pts(&ref_order4);

    chart
        .draw_series(LineSeries::new(euler_log.iter().copied(), BLUE.stroke_width(2)))?
        .label("Euler mesuré")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));
    chart.draw_series(euler_log.iter().map(|&(x, y)| Circle::new((x, y), 4, BLUE.filled())))?;

    chart
        .draw_series(LineSeries::new(rk4_log.iter().copied(), RGBColor(217, 95, 2).stroke_width(2)))?
        .label("RK4 mesuré")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(217, 95, 2).stroke_width(2))
        });
    chart.draw_series(
        rk4_log
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 4, RGBColor(217, 95, 2).filled())),
    )?;

    chart
        .draw_series(LineSeries::new(
            ref1_log.iter().copied(),
            BLUE.mix(0.5).stroke_width(1),
        ))?
        .label("pente théorique ordre 1 (Euler)")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.5).stroke_width(1))
        });

    chart
        .draw_series(LineSeries::new(
            ref4_log.iter().copied(),
            RGBColor(217, 95, 2).mix(0.5).stroke_width(1),
        ))?
        .label("pente théorique ordre 4 (RK4)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                RGBColor(217, 95, 2).mix(0.5).stroke_width(1),
            )
        });

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
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
    // Standalone binary variant (Variant B) — see
    // `chrom_rs::output::visualization::fonts` for why this call is needed
    // before any chart renders.
    chrom_rs::output::register_fonts();

    let args: Vec<String> = std::env::args().collect();
    let case = parse_case_arg(&args).unwrap_or_else(|| DEFAULT_CASE.to_string());

    let path = report_path();
    println!("Reading {} (case: {case})...", path.display());
    let points = read_case(&path, &case)?;

    println!(
        "\n{:>10} {:>16} {:>16}",
        "n_steps", "Rsf(Euler,RK4)", "Rsf(RK4 self)"
    );
    for p in &points {
        println!(
            "{:>10} {:>16} {:>16}",
            p.n_steps,
            p.rsf_euler_vs_rk4.map(|v| format!("{v:.6}")).unwrap_or_else(|| "—".into()),
            p.rsf_rk4_self_vs_next.map(|v| format!("{v:.6}")).unwrap_or_else(|| "—".into()),
        );
    }

    println!("\nOrdre empirique entre paliers consécutifs / Empirical order between consecutive steps:");
    let euler_xy: Vec<(f64, f64)> = points
        .iter()
        .filter_map(|p| p.rsf_euler_vs_rk4.map(|e| (p.n_steps as f64, e)))
        .collect();
    let rk4_xy: Vec<(f64, f64)> = points
        .iter()
        .filter_map(|p| p.rsf_rk4_self_vs_next.map(|e| (p.n_steps as f64, e)))
        .collect();
    print!("  Euler p≈[");
    for w in euler_xy.windows(2) {
        print!("{:.2} ", empirical_order(w[0], w[1]));
    }
    println!("]  (théorique: 1.0)");
    print!("  RK4   p≈[");
    for w in rk4_xy.windows(2) {
        print!("{:.2} ", empirical_order(w[0], w[1]));
    }
    println!("]  (théorique: 4.0)");

    let output_path = PathBuf::from(format!("target/plots/convergence_order_{case}.svg"));
    println!("\nGenerating plot...");
    generate_plot(&case, &points, &output_path)?;
    Ok(())
}

// =================================================================================================
// Tests unitaires / Unit tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_case_arg ───────────────────────────────────────────────────

    #[test]
    fn test_parse_case_arg_equals_form() {
        let args = vec!["bin".to_string(), "--case=glucose_fructose_linear".to_string()];
        assert_eq!(parse_case_arg(&args), Some("glucose_fructose_linear".to_string()));
    }

    #[test]
    fn test_parse_case_arg_space_form() {
        let args = vec!["bin".to_string(), "--case".to_string(), "erythorbic_alone".to_string()];
        assert_eq!(parse_case_arg(&args), Some("erythorbic_alone".to_string()));
    }

    #[test]
    fn test_parse_case_arg_absent_returns_none() {
        let args = vec!["bin".to_string()];
        assert_eq!(parse_case_arg(&args), None);
    }

    // ── theoretical_slope ────────────────────────────────────────────────

    /// Une pente d'ordre 1 doit exactement doubler l'erreur quand n_steps
    /// est divisé par deux.
    /// *An order-1 slope must exactly double the error when n_steps is
    /// halved.*
    #[test]
    fn test_theoretical_slope_order1() {
        let ns = vec![1000.0, 2000.0, 4000.0];
        let slope = theoretical_slope(1000.0, 0.1, 1.0, &ns);
        assert!((slope[1].1 - 0.05).abs() < 1e-9, "attendu 0.05, obtenu {}", slope[1].1);
        assert!((slope[2].1 - 0.025).abs() < 1e-9, "attendu 0.025, obtenu {}", slope[2].1);
    }

    /// Une pente d'ordre 4 doit diviser l'erreur par 16 quand n_steps double.
    /// *An order-4 slope must divide the error by 16 when n_steps doubles.*
    #[test]
    fn test_theoretical_slope_order4() {
        let ns = vec![1000.0, 2000.0];
        let slope = theoretical_slope(1000.0, 1.0, 4.0, &ns);
        assert!((slope[1].1 - 1.0 / 16.0).abs() < 1e-9, "attendu {}, obtenu {}", 1.0 / 16.0, slope[1].1);
    }

    // ── empirical_order ──────────────────────────────────────────────────

    #[test]
    fn test_empirical_order_recovers_known_order() {
        // err = 1 / n^2 : l'ordre empirique doit retrouver 2.0
        // *err = 1 / n^2: empirical order must recover 2.0*
        let p1 = (100.0, 1.0 / 100f64.powi(2));
        let p2 = (200.0, 1.0 / 200f64.powi(2));
        let order = empirical_order(p1, p2);
        assert!((order - 2.0).abs() < 1e-9, "attendu 2.0, obtenu {order}");
    }

    /// Régression sur les données réelles (V6, 28/08, ascorbic_erythorbic) :
    /// Euler doit rester proche de l'ordre 1 théorique ; RK4 doit rester très
    /// loin de l'ordre 4 théorique. Ce test doit échouer si l'écart RK4 se
    /// résorbe un jour — c'est le signal qu'il faut réviser le texte de
    /// l'article plutôt que documenter un résultat négatif obsolète.
    /// *Regression on real data (V6, 28/08, ascorbic_erythorbic): Euler
    /// should stay close to theoretical order 1; RK4 should stay far from
    /// theoretical order 4. This test should fail if the RK4 gap ever
    /// closes — that's the signal to revise the article's text rather than
    /// document a stale negative result.*
    #[test]
    fn test_regression_order_gap_from_actual_data() {
        // n_steps=4000 -> 8000, cas ascorbic_erythorbic, session V6 (28/08)
        let euler_p1 = (4000.0, 0.022105344448755806);
        let euler_p2 = (8000.0, 0.010926143614532034);
        let euler_order = empirical_order(euler_p1, euler_p2);
        assert!(
            (0.5..2.5).contains(&euler_order),
            "Euler devrait rester proche de l'ordre 1, obtenu {euler_order:.2}"
        );

        let rk4_p1 = (4000.0, 0.00016801798192234821);
        let rk4_p2 = (8000.0, 0.00015387198980721913);
        let rk4_order = empirical_order(rk4_p1, rk4_p2);
        assert!(
            rk4_order < 2.5,
            "RK4 ne devrait PAS approcher l'ordre 4 avec le protocole actuel, obtenu {rk4_order:.2} \
             — si ce test échoue parce que rk4_order a monté, c'est une bonne nouvelle scientifique, \
             pas un bug : réviser §4 de l'article."
        );
    }
}
