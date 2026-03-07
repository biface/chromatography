//! Visualisation du seuil de parallélisme Rayon
//! *Rayon parallelism threshold visualisation*
//!
//! # Éléments visuels / *Visual elements*
//!
//! 1. **Courbe mesurée** : série (bleue) + parallèle (rouge) avec barres d'erreur IC 95%
//! 2. **Courbe théorique O(n²)** : pointillés gris, calée sur le premier point série
//! 3. **Régression log-log** : droite de régression sur chaque régime avec exposant mesuré
//! 4. **Annotation de cassure** : gain mesuré entre npts=499 et npts=500
//!
//! *1. **Measured curve**: serial (blue) + parallel (red) with 95% CI error bars*
//! *2. **Theoretical O(n²) curve**: grey dashes, anchored on the first serial point*
//! *3. **Log-log regression**: regression line per regime with measured exponent*
//! *4. **Breakpoint annotation**: measured speedup between npts=499 and npts=500*
//!
//! # Utilisation / *Usage*
//!
//! ```bash
//! cargo bench --bench langmuir_performance -- bench_parallelism_threshold
//! cargo run --bin plot_parallelism_threshold --release
//! # → target/plots/parallelism_threshold.svg
//! ```
//!
//! # Cargo.toml
//!
//! ```toml
//! [[bin]]
//! name = "plot_parallelism_threshold"
//! path = "tools/plot_parallelism_threshold.rs"
//!
//! [dependencies]
//! plotters  = "0.3"
//! serde     = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! anyhow    = "1"
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use plotters::prelude::*;
use serde::Deserialize;

// =================================================================================================
// Constantes
// Constants
// =================================================================================================

// n_species fixé dans bench_parallelism_threshold
// n_species fixed in bench_parallelism_threshold
const N_SPECIES: usize = 2;

// Seuil de parallélisme (doit correspondre à parallel_threshold() dans chrom-rs)
// Parallelism threshold (must match parallel_threshold() in chrom-rs)
const PARALLEL_THRESHOLD_OPS: usize = 1000;

// n_points au seuil : 1000 / 2 = 500
const THRESHOLD_N_POINTS: usize = PARALLEL_THRESHOLD_OPS / N_SPECIES;

// =================================================================================================
// Désérialisation JSON Criterion / Criterion JSON deserialisation
// =================================================================================================

/// Intervalle de confiance Criterion (nanosecondes)
/// *Criterion confidence interval (nanoseconds)*
#[derive(Debug, Deserialize)]
struct ConfidenceInterval {
    lower_bound: f64,
    upper_bound: f64,
}

/// Estimation statistique Criterion
/// *Criterion statistical estimate*
#[derive(Debug, Deserialize)]
struct Estimate {
    confidence_interval: ConfidenceInterval,
    /// Estimation ponctuelle en nanosecondes / *Point estimate in nanoseconds*
    point_estimate: f64,
}

/// Contenu de `estimates.json`
/// *Content of `estimates.json`*
#[derive(Debug, Deserialize)]
struct Estimates {
    mean: Estimate,
}

// =================================================================================================
// Structures de données / Data structures
// =================================================================================================

/// Régime de calcul
/// *Computation regime*
#[derive(Debug, Clone, PartialEq)]
enum Regime {
    /// ops < PARALLEL_THRESHOLD_OPS
    Serial,
    /// ops ≥ PARALLEL_THRESHOLD_OPS
    Parallel,
}

/// Point de mesure extrait des JSON Criterion
/// *Measurement point extracted from Criterion JSON files*
#[derive(Debug, Clone)]
struct DataPoint {
    n_points: usize,
    time_ms: f64,
    time_low_ms: f64,
    time_high_ms: f64,
    regime: Regime,
}

// =================================================================================================
// Lecture des données / Data reading
// =================================================================================================

/// Extrait n_points depuis le nom de répertoire Criterion (ex. "npts_500" → 500)
/// *Extracts n_points from Criterion directory name (e.g. "npts_500" → 500)*
fn parse_n_points(name: &str) -> Option<usize> {
    name.strip_prefix("npts_")?.parse().ok()
}

/// Lit un fichier `estimates.json` et retourne les estimations
/// *Reads an `estimates.json` file and returns estimates*
fn read_estimates(path: &Path) -> anyhow::Result<Estimates> {
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

/// Collecte tous les points de mesure depuis `target/criterion/bench_parallelism_threshold/euler/`
/// *Collects all measurement points from `target/criterion/bench_parallelism_threshold/euler/`*
fn collect_data_points(criterion_dir: &Path) -> anyhow::Result<Vec<DataPoint>> {
    let euler_dir = criterion_dir
        .join("bench_parallelism_threshold")
        .join("euler");

    if !euler_dir.exists() {
        anyhow::bail!(
            "Répertoire Criterion introuvable / Criterion directory not found: {}\n\
             Lancez d'abord / Run first: \
             cargo bench --bench langmuir_performance -- bench_parallelism_threshold",
            euler_dir.display()
        );
    }

    let mut points = Vec::new();

    for entry in fs::read_dir(&euler_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(n_points) = parse_n_points(&name.to_string_lossy()) else {
            continue;
        };

        let estimates_path = entry.path().join("new").join("estimates.json");
        if !estimates_path.exists() {
            eprintln!("[WARN] Fichier manquant / Missing file: {}", estimates_path.display());
            continue;
        }

        let est = read_estimates(&estimates_path)?;
        let ops = n_points * N_SPECIES;

        points.push(DataPoint {
            n_points,
            time_ms:      est.mean.point_estimate / 1e6,
            time_low_ms:  est.mean.confidence_interval.lower_bound / 1e6,
            time_high_ms: est.mean.confidence_interval.upper_bound / 1e6,
            regime: if ops >= PARALLEL_THRESHOLD_OPS { Regime::Parallel } else { Regime::Serial },
        });
    }

    if points.is_empty() {
        anyhow::bail!("Aucune donnée trouvée / No data found in {}", euler_dir.display());
    }

    points.sort_by_key(|p| p.n_points);
    Ok(points)
}

// =================================================================================================
// Régression log-log / Log-log regression
// =================================================================================================

/// Résultat d'une régression log-log : exposant et coefficient
/// *Log-log regression result: exponent and coefficient*
///
/// Modèle : `t = a × n^α`  →  `log(t) = log(a) + α × log(n)`
/// *Model: `t = a × n^α`  →  `log(t) = log(a) + α × log(n)`*
#[derive(Debug)]
struct LogLogRegression {
    /// Exposant mesuré (pente dans l'espace log-log) / *Measured exponent (log-log slope)*
    alpha: f64,
    /// Coefficient (ordonnée à l'origine dans l'espace log-log)
    /// *Coefficient (log-log intercept)*
    log_a: f64,
}

impl LogLogRegression {
    /// Calcule la régression par moindres carrés ordinaires sur log(x), log(y)
    /// *Computes OLS regression on log(x), log(y)*
    ///
    /// Retourne `None` si moins de 2 points valides.
    /// *Returns `None` if fewer than 2 valid points.*
    fn fit(points: &[(f64, f64)]) -> Option<Self> {
        // Filtrer les valeurs non positives (log non défini)
        // Filter non-positive values (log undefined)
        let valid: Vec<(f64, f64)> = points
            .iter()
            .filter(|&&(x, y)| x > 0.0 && y > 0.0)
            .map(|&(x, y)| (x.ln(), y.ln()))
            .collect();

        if valid.len() < 2 {
            return None;
        }

        let n = valid.len() as f64;
        let sum_x:  f64 = valid.iter().map(|(x, _)| x).sum();
        let sum_y:  f64 = valid.iter().map(|(_, y)| y).sum();
        let sum_xx: f64 = valid.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = valid.iter().map(|(x, y)| x * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return None;
        }

        // Pente = exposant α / Slope = exponent α
        let alpha = (n * sum_xy - sum_x * sum_y) / denom;
        // Ordonnée à l'origine = log(a) / Intercept = log(a)
        let log_a = (sum_y - alpha * sum_x) / n;

        Some(Self { alpha, log_a })
    }

    /// Prédit le temps pour un n_points donné
    /// *Predicts time for a given n_points value*
    fn predict(&self, n: f64) -> f64 {
        self.log_a.exp() * n.powf(self.alpha)
    }
}

// =================================================================================================
// Génération du graphique / Plot generation
// =================================================================================================

/// Génère le graphique SVG avec courbe théorique, régression et annotation de cassure
/// *Generates the SVG plot with theoretical curve, regression, and breakpoint annotation*
fn generate_plot(points: &[DataPoint], output_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // ── Séparation série / parallèle ──────────────────────────────────────
    let serial:   Vec<&DataPoint> = points.iter().filter(|p| p.regime == Regime::Serial).collect();
    let parallel: Vec<&DataPoint> = points.iter().filter(|p| p.regime == Regime::Parallel).collect();

    // ── Régression log-log ────────────────────────────────────────────────
    // Régression sur les points série (la loi O(n²) théorique est ici)
    // Regression on serial points (theoretical O(n²) law is here)
    let serial_xy: Vec<(f64, f64)> = serial.iter()
        .map(|p| (p.n_points as f64, p.time_ms))
        .collect();
    let parallel_xy: Vec<(f64, f64)> = parallel.iter()
        .map(|p| (p.n_points as f64, p.time_ms))
        .collect();

    let reg_serial   = LogLogRegression::fit(&serial_xy);
    let reg_parallel = LogLogRegression::fit(&parallel_xy);

    // ── Courbe O(n²) théorique calée sur le premier point série ───────────
    // ── Theoretical O(n²) curve anchored on the first serial point ────────
    //
    // t_théo(n) = t_mesuré(n₀) × (n/n₀)²
    // On ne fait aucune hypothèse sur le coefficient absolu — on teste la forme.
    // No assumption on the absolute coefficient — we test only the shape.
    let (n0, t0) = serial.first()
        .map(|p| (p.n_points as f64, p.time_ms))
        .unwrap_or((50.0, 1.0));

    // ── Bornes des axes ───────────────────────────────────────────────────
    let x_max = points.iter().map(|p| p.n_points).max().unwrap_or(5000) as f64;

    // y_max global sur tous les points : les courbes théorique et de
    // régression sont extrapolées jusqu'à x_max, leur intersection naturelle
    // avec y_max révèle ce que coûterait le régime série sans Rayon.
    //
    // Global y_max over all points: theoretical and regression curves are
    // extrapolated up to x_max; their natural intersection with y_max
    // reveals what the serial regime would cost without Rayon.
    let y_max = points.iter().map(|p| p.time_high_ms).fold(0.0_f64, f64::max) * 1.15;

    // ── Backend SVG ───────────────────────────────────────────────────────
    let root = SVGBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(50)
        .x_label_area_size(55)
        .y_label_area_size(80)
        .caption(
            format!(
                "Seuil de parallélisme Rayon — n_species={N_SPECIES} fixé\n\
                 Rayon Parallelism Threshold — n_species={N_SPECIES} fixed \
                 (threshold: ops={PARALLEL_THRESHOLD_OPS} → n_points={THRESHOLD_N_POINTS})"
            ),
            ("sans-serif", 16).into_font(),
        )
        .build_cartesian_2d(0f64..x_max * 1.05, 0f64..y_max)?;

    chart
        .configure_mesh()
        .x_desc("n_points (points spatiaux / spatial points)")
        .y_desc("Temps moyen / Mean time (ms)")
        .x_label_formatter(&|v| format!("{}", *v as usize))
        .y_label_formatter(&|v| format!("{:.0} ms", v))
        .draw()?;

    // ── Zones de fond série / parallèle ───────────────────────────────────
    let tx = THRESHOLD_N_POINTS as f64;
    chart.draw_series(std::iter::once(
        Rectangle::new([(0.0, 0.0), (tx, y_max)], BLUE.mix(0.04).filled())
    ))?;
    chart.draw_series(std::iter::once(
        Rectangle::new([(tx, 0.0), (x_max * 1.05, y_max)], RED.mix(0.04).filled())
    ))?;

    // ── Ligne verticale du seuil ──────────────────────────────────────────
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(tx, 0.0), (tx, y_max)],
        ShapeStyle { color: GREEN.mix(0.7).to_rgba(), filled: false, stroke_width: 2 },
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        format!("Seuil / Threshold\nn_points={THRESHOLD_N_POINTS}"),
        (tx + x_max * 0.01, y_max * 0.92),
        ("sans-serif", 11).into_font().color(&GREEN.mix(0.8)),
    )))?;

    // ── Courbe O(n²) théorique (pointillés gris) ──────────────────────────
    // ── Theoretical O(n²) curve (grey dashes) ─────────────────────────────
    //
    // Tracée uniquement sur le domaine série car la loi O(n²) = coût total
    // serie (n_points × n_steps ∝ n_points²) ne s'applique pas en parallèle.
    // Drawn only over the serial domain: O(n²) = total serial cost
    // (n_points × n_steps ∝ n_points²) does not apply in the parallel regime.
    // x_serial_max sert uniquement au domaine de la régression série.
    // La courbe O(n²) est extrapolée jusqu'à x_max : son clip naturel par
    // y_max montre visuellement où le régime série sans Rayon dépasserait
    // le coût parallèle mesuré.
    //
    // x_serial_max is used only for the serial regression domain.
    // The O(n²) curve is extrapolated up to x_max: its natural clip by
    // y_max visually shows where the serial regime without Rayon would
    // exceed the measured parallel cost.
    let x_serial_max = serial.last().map(|p| p.n_points as f64).unwrap_or(tx);
    let theory_points: Vec<(f64, f64)> = (0..=400)
        .map(|i| {
            let x = n0 + (x_max - n0) * i as f64 / 400.0;
            let y = t0 * (x / n0).powi(2);
            (x, y)
        })
        .filter(|&(_, y)| y <= y_max)
        .collect();

    chart.draw_series(
        theory_points.windows(2).map(|w| {
            PathElement::new(
                vec![w[0], w[1]],
                ShapeStyle {
                    color: RGBColor(150, 150, 150).mix(0.8).to_rgba(),
                    filled: false,
                    stroke_width: 2,
                },
            )
        })
    )?.label("O(n²) théorique / theoretical")
      .legend(|(x, y)| PathElement::new(
          vec![(x, y), (x + 20, y)],
          ShapeStyle { color: RGBColor(150, 150, 150).to_rgba(), filled: false, stroke_width: 2 }
      ));

    // ── Droite de régression série (pointillés bleus foncés) ──────────────
    // ── Serial regression line (dark blue dashes) ─────────────────────────
    if let Some(ref reg) = reg_serial {
        // Extrapolée jusqu'à x_max comme la courbe O(n²) théorique.
        // Extrapolated up to x_max like the theoretical O(n²) curve.
        let reg_pts: Vec<(f64, f64)> = (0..=400)
            .map(|i| {
                let x = n0 + (x_max - n0) * i as f64 / 400.0;
                (x, reg.predict(x))
            })
            .filter(|&(_, y)| y > 0.0 && y <= y_max)
            .collect();

        chart.draw_series(
            reg_pts.windows(2).map(|w| {
                PathElement::new(
                    vec![w[0], w[1]],
                    ShapeStyle {
                        color: RGBColor(0, 0, 180).mix(0.6).to_rgba(),
                        filled: false,
                        stroke_width: 2,
                    },
                )
            })
        )?.label(format!("Régression série / Serial regression: O(n^{:.2})", reg.alpha))
          .legend(|(x, y)| PathElement::new(
              vec![(x, y), (x + 20, y)],
              ShapeStyle { color: RGBColor(0, 0, 180).mix(0.6).to_rgba(), filled: false, stroke_width: 2 }
          ));
    }

    // ── Droite de régression parallèle (pointillés rouges foncés) ─────────
    // ── Parallel regression line (dark red dashes) ────────────────────────
    if let Some(ref reg) = reg_parallel {
        let x_par_min  = parallel.first().map(|p| p.n_points as f64).unwrap_or(tx);
        let x_par_max  = parallel.last().map(|p| p.n_points as f64).unwrap_or(x_max);

        let reg_pts: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                let x = x_par_min + (x_par_max - x_par_min) * i as f64 / 100.0;
                (x, reg.predict(x))
            })
            .filter(|&(_, y)| y > 0.0 && y <= y_max)
            .collect();

        chart.draw_series(
            reg_pts.windows(2).map(|w| {
                PathElement::new(
                    vec![w[0], w[1]],
                    ShapeStyle {
                        color: RGBColor(180, 0, 0).mix(0.6).to_rgba(),
                        filled: false,
                        stroke_width: 2,
                    },
                )
            })
        )?.label(format!("Régression parallèle / Parallel regression: O(n^{:.2})", reg.alpha))
          .legend(|(x, y)| PathElement::new(
              vec![(x, y), (x + 20, y)],
              ShapeStyle { color: RGBColor(180, 0, 0).mix(0.6).to_rgba(), filled: false, stroke_width: 2 }
          ));
    }

    // ── Barres d'erreur IC 95% ─────────────────────────────────────────────
    // ── 95% CI error bars ─────────────────────────────────────────────────
    for p in points {
        let x   = p.n_points as f64;
        let col = if p.regime == Regime::Serial { BLUE.mix(0.4) } else { RED.mix(0.4) };
        let cap = x_max * 0.004;

        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, p.time_low_ms), (x, p.time_high_ms)],
            ShapeStyle { color: col.to_rgba(), filled: false, stroke_width: 1 },
        )))?;
        for &y_cap in &[p.time_low_ms, p.time_high_ms] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x - cap, y_cap), (x + cap, y_cap)],
                ShapeStyle { color: col.to_rgba(), filled: false, stroke_width: 1 },
            )))?;
        }
    }

    // ── Courbe série mesurée (bleue) ──────────────────────────────────────
    let serial_curve: Vec<(f64, f64)> = serial.iter()
        .map(|p| (p.n_points as f64, p.time_ms))
        .collect();
    chart.draw_series(LineSeries::new(serial_curve.clone(), BLUE.stroke_width(3)))?
        .label("Série mesurée / Measured serial")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    chart.draw_series(serial_curve.iter().map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled())))?;

    // ── Courbe parallèle mesurée (rouge) ──────────────────────────────────
    let parallel_curve: Vec<(f64, f64)> = parallel.iter()
        .map(|p| (p.n_points as f64, p.time_ms))
        .collect();
    chart.draw_series(LineSeries::new(parallel_curve.clone(), RED.stroke_width(3)))?
        .label("Parallèle mesuré / Measured parallel")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    chart.draw_series(parallel_curve.iter().map(|&(x, y)| Circle::new((x, y), 5, RED.filled())))?;

    // ── Annotation du gain à la cassure ───────────────────────────────────
    // ── Breakpoint gain annotation ────────────────────────────────────────
    //
    // Cherche les deux points de part et d'autre du seuil (499 et 500).
    // Seeks the two points on either side of the threshold (499 and 500).
    let before = points.iter().find(|p| p.n_points == THRESHOLD_N_POINTS - 1);
    let after  = points.iter().find(|p| p.n_points == THRESHOLD_N_POINTS);

    if let (Some(b), Some(a)) = (before, after) {
        let speedup = b.time_ms / a.time_ms;

        // Flèche entre les deux points
        // Arrow between the two points
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(b.n_points as f64, b.time_ms), (a.n_points as f64, a.time_ms)],
            ShapeStyle { color: RGBColor(0, 150, 0).to_rgba(), filled: false, stroke_width: 3 },
        )))?;

        // Label du gain
        // Gain label
        chart.draw_series(std::iter::once(Text::new(
            format!("Gain Rayon\n{:.2}×", speedup),
            (tx + x_max * 0.02, (b.time_ms + a.time_ms) / 2.0),
            ("sans-serif", 13).into_font().color(&RGBColor(0, 130, 0)),
        )))?;
    }


    // ── Légende ───────────────────────────────────────────────────────────
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.92))
        .border_style(RGBColor(180, 180, 180))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    println!("✅ Graphique généré / Plot generated: {}", output_path.display());
    Ok(())
}

// =================================================================================================
// Point d'entrée / Entry point
// =================================================================================================

fn main() -> anyhow::Result<()> {
    let criterion_dir = PathBuf::from("target/criterion");
    let output_path   = PathBuf::from("target/plots/parallelism_threshold.svg");

    println!("📂 Lecture des données Criterion / Reading Criterion data...");

    let points = collect_data_points(&criterion_dir)?;

    // ── Tableau récapitulatif ─────────────────────────────────────────────
    println!("\n{:<12} {:<10} {:<20} {:<12} {:<12}", "n_points", "ops", "régime", "ms", "±IC_ms");
    println!("{:-<68}", "");
    for p in &points {
        let regime = if p.regime == Regime::Serial { "série" } else { "PARALLÈLE" };
        println!(
            "{:<12} {:<10} {:<20} {:<12.3} ±{:.3}",
            p.n_points, p.n_points * N_SPECIES, regime,
            p.time_ms, (p.time_high_ms - p.time_low_ms) / 2.0
        );
    }

    // ── Régressions ───────────────────────────────────────────────────────
    let serial_xy:   Vec<(f64, f64)> = points.iter().filter(|p| p.regime == Regime::Serial)
        .map(|p| (p.n_points as f64, p.time_ms)).collect();
    let parallel_xy: Vec<(f64, f64)> = points.iter().filter(|p| p.regime == Regime::Parallel)
        .map(|p| (p.n_points as f64, p.time_ms)).collect();

    println!("\n📐 Régressions log-log / Log-log regressions:");
    if let Some(r) = LogLogRegression::fit(&serial_xy) {
        println!(
            "   Série    : t ∝ n^{:.3}  (théorique O(n²) = n^2.000)  écart={:+.3}",
            r.alpha, r.alpha - 2.0
        );
    }
    if let Some(r) = LogLogRegression::fit(&parallel_xy) {
        println!("   Parallèle: t ∝ n^{:.3}", r.alpha);
    }

    // ── Gain à la cassure ─────────────────────────────────────────────────
    let before = points.iter().find(|p| p.n_points == THRESHOLD_N_POINTS - 1);
    let after  = points.iter().find(|p| p.n_points == THRESHOLD_N_POINTS);
    if let (Some(b), Some(a)) = (before, after) {
        println!(
            "\n🚀 Gain au seuil / Speedup at threshold (n={} → n={}): {:.2}×",
            b.n_points, a.n_points, b.time_ms / a.time_ms
        );
    }

    println!("\n🎨 Génération du graphique / Generating plot...");
    generate_plot(&points, &output_path)?;
    Ok(())
}

// =================================================================================================
// Tests unitaires / Unit tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_n_points ────────────────────────────────────────────────────

    #[test]
    fn test_parse_n_points_valid() {
        assert_eq!(parse_n_points("npts_500"), Some(500));
        assert_eq!(parse_n_points("npts_50"),  Some(50));
    }

    #[test]
    fn test_parse_n_points_invalid() {
        assert_eq!(parse_n_points("euler"),    None);
        assert_eq!(parse_n_points("npts_abc"), None);
        assert_eq!(parse_n_points(""),         None);
    }

    // ── LogLogRegression ─────────────────────────────────────────────────

    /// Régression sur y = x² doit retrouver α = 2.0 exactement
    /// *Regression on y = x² must recover α = 2.0 exactly*
    #[test]
    fn test_regression_perfect_quadratic() {
        let pts: Vec<(f64, f64)> = (1..=10).map(|i| {
            let x = i as f64 * 50.0;
            (x, x * x)
        }).collect();
        let reg = LogLogRegression::fit(&pts).unwrap();
        assert!((reg.alpha - 2.0).abs() < 1e-6,
            "Exposant attendu 2.0, obtenu {}", reg.alpha);
    }

    /// Régression sur y = x³ doit retrouver α = 3.0
    /// *Regression on y = x³ must recover α = 3.0*
    #[test]
    fn test_regression_perfect_cubic() {
        let pts: Vec<(f64, f64)> = (1..=8).map(|i| {
            let x = i as f64 * 10.0;
            (x, x.powi(3))
        }).collect();
        let reg = LogLogRegression::fit(&pts).unwrap();
        assert!((reg.alpha - 3.0).abs() < 1e-6,
            "Exposant attendu 3.0, obtenu {}", reg.alpha);
    }

    /// Moins de 2 points → None (pas de régression possible)
    /// *Fewer than 2 points → None (regression not possible)*
    #[test]
    fn test_regression_insufficient_points() {
        assert!(LogLogRegression::fit(&[]).is_none());
        assert!(LogLogRegression::fit(&[(10.0, 5.0)]).is_none());
    }

    /// predict() est cohérent avec les données de la régression
    /// *predict() is consistent with the regression data*
    #[test]
    fn test_regression_predict_consistency() {
        let pts = vec![(100.0, 10.0), (200.0, 40.0), (400.0, 160.0)]; // α=2
        let reg = LogLogRegression::fit(&pts).unwrap();
        let pred_200 = reg.predict(200.0);
        // La prédiction sur un point d'entraînement doit être proche
        // Prediction on a training point must be close
        assert!((pred_200 - 40.0).abs() / 40.0 < 0.05,
            "Prédiction trop éloignée : {} ≠ 40.0", pred_200);
    }

    // ── Régimes ───────────────────────────────────────────────────────────

    /// La cassure est exactement à n_points = THRESHOLD_N_POINTS
    /// *The breakpoint is exactly at n_points = THRESHOLD_N_POINTS*
    #[test]
    fn test_threshold_boundary() {
        let ops_before = (THRESHOLD_N_POINTS - 1) * N_SPECIES;
        let ops_at     = THRESHOLD_N_POINTS       * N_SPECIES;
        assert!(ops_before < PARALLEL_THRESHOLD_OPS);
        assert!(ops_at    >= PARALLEL_THRESHOLD_OPS);
    }

    // ── Conversion ────────────────────────────────────────────────────────

    #[test]
    fn test_ns_to_ms() {
        assert!((1_000_000.0_f64 / 1e6 - 1.0).abs() < 1e-12);
    }

    /// Les données de ta mesure réelle servent de test de régression
    /// *Your actual measurement data serves as a regression test*
    #[test]
    fn test_speedup_regression_from_actual_data() {
        let time_499 = 117.98_f64;
        let time_500 =  38.13_f64;
        let speedup  = time_499 / time_500;
        assert!(speedup > 2.0 && speedup < 6.0,
            "Gain improbable / Implausible speedup: {speedup:.2}");
    }
}
