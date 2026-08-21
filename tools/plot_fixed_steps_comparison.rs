//! Comparaison adaptatif vs pas fixe — isole la loi O(n³) de la confusion
//! avec le nombre de pas de temps adaptatif.
//! *Adaptive vs fixed-steps comparison — isolates the O(n³) law from the
//! confound with the adaptive time-step count.*
//!
//! # Pourquoi ce tool / *Why this tool*
//!
//! `bench_multi_species_scaling` et `bench_species_response_curve` calculent
//! `n_steps` par tirage à chaque `n_species` (`safe_nsteps_for_multi`), donc
//! le temps mesuré varie avec deux facteurs à la fois : la taille du
//! problème (n_species → O(n³) attendu) ET le nombre de pas (qui varie
//! aussi avec n_species). Les groupes `_fixed_steps` (ajout 2026-08-17)
//! rejouent la même grille à `n_steps` fixe : la comparaison des deux
//! révèle la part de la variation qui vient du seul n_steps, et donne un
//! exposant α mesuré en régime "pur" O(n³) découplé de ce facteur.
//! *`bench_multi_species_scaling` and `bench_species_response_curve`
//! compute `n_steps` by draw at each `n_species` (`safe_nsteps_for_multi`),
//! so measured time varies with two factors at once: problem size
//! (n_species → expected O(n³)) AND step count (which also varies with
//! n_species). The `_fixed_steps` groups (added 2026-08-17) replay the same
//! grid at fixed `n_steps`: comparing the two reveals how much of the
//! variation comes from n_steps alone, and gives a "pure" O(n³) exponent
//! decoupled from that factor.*
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench langmuir_performance -- bench_multi_species_scaling
//! cargo bench --bench langmuir_performance -- bench_species_response_curve
//! cargo run --bin plot_fixed_steps_comparison --release
//! ```
//!
//! Produit trois fichiers dans `target/plots/` :
//! *Produces three files under `target/plots/`:*
//! - `multi_species_scaling_fixed_vs_adaptive.svg`
//! - `species_response_curve_fixed_vs_adaptive_euler.svg`
//! - `species_response_curve_fixed_vs_adaptive_rk4.svg`
//!
//! Chaque graphique superpose adaptatif et fixe sur les mêmes axes (pas
//! deux images séparées) — c'est la comparaison directe qui montre l'effet
//! du confondant, pas un avant/après juxtaposé.
//! *Each chart overlays adaptive and fixed on the same axes (not two
//! separate images) — that is the direct comparison that shows the
//! confound's effect, not a side-by-side before/after.*

use std::fs;
use std::path::{Path, PathBuf};

use plotters::prelude::*;
use serde::Deserialize;

// =================================================================================================
// Désérialisation JSON Criterion — identique à plot_parallelism_threshold.rs
// Criterion JSON deserialisation — identical to plot_parallelism_threshold.rs
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
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

// =================================================================================================
// Structures de données / Data structures
// =================================================================================================

#[derive(Debug, Clone)]
struct DataPoint {
    n: usize,
    time_ms: f64,
    time_low_ms: f64,
    time_high_ms: f64,
}

// =================================================================================================
// Lecture des données / Data reading
// =================================================================================================

/// Extrait n depuis un nom de répertoire Criterion avec le préfixe donné
/// (ex. `parse_n("n_species_50", "n_species_")` → `Some(50)`).
/// *Extracts n from a Criterion directory name with the given prefix.*
fn parse_n(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse().ok()
}

/// Collecte les points de mesure sous `target/criterion/<group>/<solver>/`.
/// *Collects measurement points under `target/criterion/<group>/<solver>/`.*
fn collect_points(
    criterion_dir: &Path,
    group: &str,
    solver: &str,
    id_prefix: &str,
) -> anyhow::Result<Vec<DataPoint>> {
    let dir = criterion_dir.join(group).join(solver);

    if !dir.exists() {
        anyhow::bail!(
            "Répertoire Criterion introuvable / Criterion directory not found: {}\n\
             Lancez d'abord / Run first: cargo bench --bench langmuir_performance -- {}",
            dir.display(),
            group
        );
    }

    let mut points = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(n) = parse_n(&name.to_string_lossy(), id_prefix) else {
            continue;
        };

        let estimates_path = entry.path().join("new").join("estimates.json");
        if !estimates_path.exists() {
            eprintln!(
                "[WARN] Fichier manquant / Missing file: {}",
                estimates_path.display()
            );
            continue;
        }

        let est = read_estimates(&estimates_path)?;
        points.push(DataPoint {
            n,
            time_ms: est.mean.point_estimate / 1e6,
            time_low_ms: est.mean.confidence_interval.lower_bound / 1e6,
            time_high_ms: est.mean.confidence_interval.upper_bound / 1e6,
        });
    }

    if points.is_empty() {
        anyhow::bail!("Aucune donnée trouvée / No data found in {}", dir.display());
    }

    points.sort_by_key(|p| p.n);
    Ok(points)
}

// =================================================================================================
// Régression log-log / Log-log regression — identique à plot_parallelism_threshold.rs
// =================================================================================================

#[derive(Debug)]
struct LogLogRegression {
    alpha: f64,
    log_a: f64,
}

impl LogLogRegression {
    fn fit(points: &[(f64, f64)]) -> Option<Self> {
        let valid: Vec<(f64, f64)> = points
            .iter()
            .filter(|&&(x, y)| x > 0.0 && y > 0.0)
            .map(|&(x, y)| (x.ln(), y.ln()))
            .collect();

        if valid.len() < 2 {
            return None;
        }

        let n = valid.len() as f64;
        let sum_x: f64 = valid.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = valid.iter().map(|(_, y)| y).sum();
        let sum_xx: f64 = valid.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = valid.iter().map(|(x, y)| x * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return None;
        }

        let alpha = (n * sum_xy - sum_x * sum_y) / denom;
        let log_a = (sum_y - alpha * sum_x) / n;

        Some(Self { alpha, log_a })
    }

    fn predict(&self, n: f64) -> f64 {
        self.log_a.exp() * n.powf(self.alpha)
    }
}

// =================================================================================================
// Génération du graphique superposé / Overlaid plot generation
// =================================================================================================

/// Trace adaptatif et fixe superposés sur les mêmes axes, avec régression
/// log-log de chaque série et l'exposant α affiché dans la légende.
/// *Plots adaptive and fixed overlaid on the same axes, with a log-log
/// regression of each series and the α exponent shown in the legend.*
#[allow(clippy::too_many_arguments)]
fn generate_comparison_plot(
    adaptive: &[DataPoint],
    fixed: &[DataPoint],
    title: &str,
    x_desc: &str,
    output_path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let reg_adaptive = LogLogRegression::fit(
        &adaptive
            .iter()
            .map(|p| (p.n as f64, p.time_ms))
            .collect::<Vec<_>>(),
    );
    let reg_fixed = LogLogRegression::fit(
        &fixed
            .iter()
            .map(|p| (p.n as f64, p.time_ms))
            .collect::<Vec<_>>(),
    );

    let x_max = adaptive
        .iter()
        .chain(fixed.iter())
        .map(|p| p.n)
        .max()
        .unwrap_or(50) as f64;
    let y_max = adaptive
        .iter()
        .chain(fixed.iter())
        .map(|p| p.time_high_ms)
        .fold(0.0_f64, f64::max)
        * 1.15;

    let root = SVGBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let alpha_line = format!(
        "α adaptatif={:.2}  α fixe={:.2}  (théorique O(n³)=3.00)",
        reg_adaptive.as_ref().map(|r| r.alpha).unwrap_or(f64::NAN),
        reg_fixed.as_ref().map(|r| r.alpha).unwrap_or(f64::NAN),
    );

    let mut chart = ChartBuilder::on(&root)
        .margin(50)
        .x_label_area_size(55)
        .y_label_area_size(80)
        .caption(
            format!("{title}\n{alpha_line}"),
            ("sans-serif", 16).into_font(),
        )
        .build_cartesian_2d(0f64..x_max * 1.05, 0f64..y_max)?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("Temps moyen / Mean time (ms)")
        .x_label_formatter(&|v| format!("{}", *v as usize))
        .y_label_formatter(&|v| format!("{:.1} ms", v))
        .draw()?;

    // ── Série adaptative (bleu) avec barres d'erreur ────────────────────────
    // ── Adaptive series (blue) with error bars ──────────────────────────────
    let cap = x_max * 0.004;
    for p in adaptive {
        let x = p.n as f64;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, p.time_low_ms), (x, p.time_high_ms)],
            ShapeStyle {
                color: BLUE.mix(0.5).to_rgba(),
                filled: false,
                stroke_width: 1,
            },
        )))?;
        for &y_cap in &[p.time_low_ms, p.time_high_ms] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x - cap, y_cap), (x + cap, y_cap)],
                ShapeStyle {
                    color: BLUE.mix(0.5).to_rgba(),
                    filled: false,
                    stroke_width: 1,
                },
            )))?;
        }
    }

    chart
        .draw_series(LineSeries::new(
            adaptive.iter().map(|p| (p.n as f64, p.time_ms)),
            BLUE.stroke_width(2),
        ))?
        .label("Adaptatif (n_steps variable) / Adaptive (variable n_steps)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

    // ── Série fixe (orange) avec barres d'erreur ────────────────────────────
    // ── Fixed series (orange) with error bars ───────────────────────────────
    let orange = RGBColor(230, 126, 34);
    for p in fixed {
        let x = p.n as f64;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, p.time_low_ms), (x, p.time_high_ms)],
            ShapeStyle {
                color: orange.mix(0.5).to_rgba(),
                filled: false,
                stroke_width: 1,
            },
        )))?;
        for &y_cap in &[p.time_low_ms, p.time_high_ms] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x - cap, y_cap), (x + cap, y_cap)],
                ShapeStyle {
                    color: orange.mix(0.5).to_rgba(),
                    filled: false,
                    stroke_width: 1,
                },
            )))?;
        }
    }

    chart
        .draw_series(LineSeries::new(
            fixed.iter().map(|p| (p.n as f64, p.time_ms)),
            orange.stroke_width(2),
        ))?
        .label("Fixe, n_steps=500 / Fixed, n_steps=500")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], orange.stroke_width(2)));

    // ── Régressions log-log extrapolées, en pointillés fins ─────────────────
    // ── Extrapolated log-log regressions, thin dashed ───────────────────────
    if let Some(ref r) = reg_adaptive {
        let n0 = adaptive.first().map(|p| p.n as f64).unwrap_or(1.0);
        let pts: Vec<(f64, f64)> = (0..=200)
            .map(|i| {
                let x = n0 + (x_max - n0) * i as f64 / 200.0;
                (x, r.predict(x))
            })
            .filter(|&(_, y)| y <= y_max)
            .collect();
        chart.draw_series(pts.windows(2).map(|w| {
            PathElement::new(
                vec![w[0], w[1]],
                ShapeStyle {
                    color: BLUE.mix(0.5).to_rgba(),
                    filled: false,
                    stroke_width: 1,
                },
            )
        }))?;
    }
    if let Some(ref r) = reg_fixed {
        let n0 = fixed.first().map(|p| p.n as f64).unwrap_or(1.0);
        let pts: Vec<(f64, f64)> = (0..=200)
            .map(|i| {
                let x = n0 + (x_max - n0) * i as f64 / 200.0;
                (x, r.predict(x))
            })
            .filter(|&(_, y)| y <= y_max)
            .collect();
        chart.draw_series(pts.windows(2).map(|w| {
            PathElement::new(
                vec![w[0], w[1]],
                ShapeStyle {
                    color: orange.mix(0.5).to_rgba(),
                    filled: false,
                    stroke_width: 1,
                },
            )
        }))?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.92))
        .border_style(RGBColor(180, 180, 180))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    println!(
        "✅ Graphique généré / Plot generated: {}",
        output_path.display()
    );
    Ok(())
}

/// Imprime un tableau récapitulatif comparatif adaptatif/fixe sur stdout.
/// *Prints an adaptive/fixed comparative summary table to stdout.*
fn print_comparison_table(label: &str, adaptive: &[DataPoint], fixed: &[DataPoint]) {
    println!("\n{label}");
    println!(
        "{:<8} {:<14} {:<14} {:<10}",
        "n", "adaptatif (ms)", "fixe (ms)", "écart"
    );
    println!("{:-<50}", "");
    for a in adaptive {
        if let Some(f) = fixed.iter().find(|f| f.n == a.n) {
            let ecart_pct = 100.0 * (a.time_ms - f.time_ms) / f.time_ms;
            println!(
                "{:<8} {:<14.3} {:<14.3} {:>+8.1}%",
                a.n, a.time_ms, f.time_ms, ecart_pct
            );
        }
    }
}

// =================================================================================================
// Point d'entrée / Entry point
// =================================================================================================

fn main() -> anyhow::Result<()> {
    chrom_rs::output::register_fonts();

    let criterion_dir = PathBuf::from("target/criterion");
    let plots_dir = PathBuf::from("target/plots");

    // ── Comparaison 1 : bench_multi_species_scaling (euler seul) ───────────
    println!("📂 Lecture bench_multi_species_scaling / bench_multi_species_scaling_fixed_steps...");
    let adaptive_mss = collect_points(
        &criterion_dir,
        "bench_multi_species_scaling",
        "euler",
        "n_species_",
    )?;
    let fixed_mss = collect_points(
        &criterion_dir,
        "bench_multi_species_scaling_fixed_steps",
        "euler",
        "n_species_",
    )?;
    print_comparison_table(
        "bench_multi_species_scaling (euler)",
        &adaptive_mss,
        &fixed_mss,
    );
    generate_comparison_plot(
        &adaptive_mss,
        &fixed_mss,
        "bench_multi_species_scaling — adaptatif vs n_steps fixe (euler)",
        "n_species",
        &plots_dir.join("multi_species_scaling_fixed_vs_adaptive.svg"),
    )?;

    // ── Comparaison 2 : bench_species_response_curve, euler ────────────────
    println!("\n📂 Lecture bench_species_response_curve_small / _fixed_steps (euler)...");
    let adaptive_src_euler = collect_points(
        &criterion_dir,
        "bench_species_response_curve_small",
        "euler",
        "n_sp_",
    )?;
    let fixed_src_euler = collect_points(
        &criterion_dir,
        "bench_species_response_curve_fixed_steps",
        "euler",
        "n_sp_",
    )?;
    print_comparison_table(
        "bench_species_response_curve (euler)",
        &adaptive_src_euler,
        &fixed_src_euler,
    );
    generate_comparison_plot(
        &adaptive_src_euler,
        &fixed_src_euler,
        "bench_species_response_curve — adaptatif vs n_steps fixe (euler)",
        "n_species",
        &plots_dir.join("species_response_curve_fixed_vs_adaptive_euler.svg"),
    )?;

    // ── Comparaison 3 : bench_species_response_curve, rk4 ───────────────────
    println!("\n📂 Lecture bench_species_response_curve_small / _fixed_steps (rk4)...");
    let adaptive_src_rk4 = collect_points(
        &criterion_dir,
        "bench_species_response_curve_small",
        "rk4",
        "n_sp_",
    )?;
    let fixed_src_rk4 = collect_points(
        &criterion_dir,
        "bench_species_response_curve_fixed_steps",
        "rk4",
        "n_sp_",
    )?;
    print_comparison_table(
        "bench_species_response_curve (rk4)",
        &adaptive_src_rk4,
        &fixed_src_rk4,
    );
    generate_comparison_plot(
        &adaptive_src_rk4,
        &fixed_src_rk4,
        "bench_species_response_curve — adaptatif vs n_steps fixe (rk4)",
        "n_species",
        &plots_dir.join("species_response_curve_fixed_vs_adaptive_rk4.svg"),
    )?;

    Ok(())
}

// =================================================================================================
// Tests unitaires / Unit tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_n_species_prefix() {
        assert_eq!(parse_n("n_species_50", "n_species_"), Some(50));
        assert_eq!(parse_n("n_sp_18", "n_species_"), None); // préfixe différent / different prefix
    }

    #[test]
    fn test_parse_n_sp_prefix() {
        assert_eq!(parse_n("n_sp_18", "n_sp_"), Some(18));
        assert_eq!(parse_n("euler", "n_sp_"), None);
    }

    #[test]
    fn test_parse_n_invalid() {
        assert_eq!(parse_n("n_species_abc", "n_species_"), None);
        assert_eq!(parse_n("", "n_species_"), None);
    }

    #[test]
    fn test_regression_perfect_cubic() {
        let pts: Vec<(f64, f64)> = (1..=8)
            .map(|i| {
                let x = i as f64 * 10.0;
                (x, x.powi(3))
            })
            .collect();
        let reg = LogLogRegression::fit(&pts).unwrap();
        assert!(
            (reg.alpha - 3.0).abs() < 1e-6,
            "Exposant attendu 3.0, obtenu {}",
            reg.alpha
        );
    }

    #[test]
    fn test_regression_insufficient_points() {
        assert!(LogLogRegression::fit(&[]).is_none());
        assert!(LogLogRegression::fit(&[(10.0, 5.0)]).is_none());
    }
}
