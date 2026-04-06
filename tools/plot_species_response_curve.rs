//! Visualisation de la courbe de réponse n_species (bench_species_response_curve)
//! *Species response curve visualisation (bench_species_response_curve)*
//!
//! # Éléments visuels / *Visual elements*
//!
//! 1. **Courbes mesurées** : Euler (bleu) + RK4 (rouge) avec barres d'erreur IC 95%
//! 2. **Courbe théorique O(n³)** : pointillés gris, calée sur le premier point (n=2)
//! 3. **Régression log-log** : exposant mesuré sur Euler et RK4 séparément
//! 4. **Ligne du seuil de parallélisme** : verte, à n_species = 10 (ops = 1000)
//! 5. **Annotation du ratio RK4/Euler** : mesuré sur chaque point
//!
//! *1. **Measured curves**: Euler (blue) + RK4 (red) with 95% CI error bars*
//! *2. **Theoretical O(n³) curve**: grey dashes, anchored on first point (n=2)*
//! *3. **Log-log regression**: measured exponent on Euler and RK4 separately*
//! *4. **Parallelism threshold line**: green, at n_species=10 (ops=1000)*
//! *5. **RK4/Euler ratio annotation**: measured at each point*
//!
//! # Utilisation / *Usage*
//!
//! ```bash
//! cargo bench --bench langmuir_performance -- bench_species_response_curve
//! cargo run --bin plot_species_response_curve --release
//! # → target/plots/species_response_curve.svg
//! ```
//!
//! # Cargo.toml
//!
//! ```toml
//! [[bin]]
//! name = "plot_species_response_curve"
//! path = "tools/plot_species_response_curve.rs"
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use plotters::prelude::*;
use serde::Deserialize;

// =================================================================================================
// Constantes / Constants
// =================================================================================================

// n_points fixé dans bench_species_response_curve
// n_points fixed in bench_species_response_curve
const N_POINTS: usize = 100;

// Seuil de parallélisme : ops = n_points × n_species ≥ 1000 → n_species ≥ 10
// Parallelism threshold: ops = n_points × n_species ≥ 1000 → n_species ≥ 10
const PARALLEL_THRESHOLD_OPS: usize = 1000;
const THRESHOLD_N_SPECIES: usize = PARALLEL_THRESHOLD_OPS / N_POINTS; // = 10

// n_species de référence pour le calage de la courbe O(n³)
// Reference n_species for O(n³) curve anchoring
const N_REF: usize = 2;

// =================================================================================================
// Désérialisation JSON Criterion / Criterion JSON deserialisation
// =================================================================================================

#[derive(Debug, Deserialize)]
struct ConfidenceInterval {
    lower_bound: f64,
    upper_bound: f64,
}

#[derive(Debug, Deserialize)]
struct Estimate {
    confidence_interval: ConfidenceInterval,
    /// Estimation ponctuelle en nanosecondes / *Point estimate in nanoseconds*
    point_estimate: f64,
}

#[derive(Debug, Deserialize)]
struct Estimates {
    mean: Estimate,
}

// =================================================================================================
// Structures de données / Data structures
// =================================================================================================

/// Identifiant de solveur
/// *Solver identifier*
#[derive(Debug, Clone, PartialEq)]
enum Solver {
    Euler,
    Rk4,
}

/// Point de mesure extrait des JSON Criterion
/// *Measurement point extracted from Criterion JSON files*
#[derive(Debug, Clone)]
struct DataPoint {
    n_species: usize,
    solver: Solver,
    time_ms: f64,
    time_low_ms: f64,
    time_high_ms: f64,
}

// =================================================================================================
// Lecture des données / Data reading
// =================================================================================================

/// Extrait n_species depuis le nom de sous-répertoire (ex. "n_sp_10" → 10)
/// *Extracts n_species from subdirectory name (e.g. "n_sp_10" → 10)*
fn parse_n_species(name: &str) -> Option<usize> {
    name.strip_prefix("n_sp_")?.parse().ok()
}

/// Lit un fichier `estimates.json`
/// *Reads an `estimates.json` file*
fn read_estimates(path: &Path) -> anyhow::Result<Estimates> {
    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

/// Collecte tous les points pour un solveur donné
/// *Collects all points for a given solver*
fn collect_solver_points(solver_dir: &Path, solver: Solver) -> anyhow::Result<Vec<DataPoint>> {
    if !solver_dir.exists() {
        return Ok(vec![]);
    }

    let mut points = Vec::new();

    for entry in fs::read_dir(solver_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(n_species) = parse_n_species(&name.to_string_lossy()) else {
            continue;
        };

        // Criterion peut écrire dans "new/" ou directement dans le répertoire
        // Criterion may write into "new/" or directly in the directory
        let estimates_path = entry.path().join("new").join("estimates.json");
        if !estimates_path.exists() {
            eprintln!(
                "[WARN] Fichier manquant / Missing: {}",
                estimates_path.display()
            );
            continue;
        }

        let est = read_estimates(&estimates_path)?;
        let ops = N_POINTS * n_species;

        points.push(DataPoint {
            n_species,
            solver: solver.clone(),
            time_ms: est.mean.point_estimate / 1e6,
            time_low_ms: est.mean.confidence_interval.lower_bound / 1e6,
            time_high_ms: est.mean.confidence_interval.upper_bound / 1e6,
        });
    }

    points.sort_by_key(|p| p.n_species);
    Ok(points)
}

/// Collecte Euler et RK4 depuis `target/criterion/<group_name>/`
/// *Collects Euler and RK4 from `target/criterion/<group_name>/`*
///
/// Accepte plusieurs noms de groupe pour compatibilité avec l'ancienne
/// (`bench_species_response_curve`) et la nouvelle architecture
/// (`bench_species_response_curve_small`, `bench_species_response_curve_large`).
///
/// *Accepts several group names for compatibility with both the old
/// (`bench_species_response_curve`) and new split architecture.*
fn collect_all_points(criterion_dir: &Path) -> anyhow::Result<Vec<DataPoint>> {
    // Groupes à essayer dans l'ordre de préférence
    // Groups to try in order of preference
    let group_names = [
        // Architecture scindée (nouvelle) / Split architecture (new)
        "bench_species_response_curve_small",
        "bench_species_response_curve_large",
        // Architecture monolithique (ancienne) / Monolithic architecture (old)
        "bench_species_response_curve",
    ];

    let mut all_points: Vec<DataPoint> = Vec::new();
    let mut found_any = false;

    for group_name in group_names {
        let group_dir = criterion_dir.join(group_name);
        if !group_dir.exists() {
            continue;
        }
        found_any = true;

        let mut euler = collect_solver_points(&group_dir.join("euler"), Solver::Euler)?;
        let mut rk4 = collect_solver_points(&group_dir.join("rk4"), Solver::Rk4)?;

        all_points.append(&mut euler);
        all_points.append(&mut rk4);
    }

    if !found_any {
        anyhow::bail!(
            "Aucun répertoire Criterion trouvé / No Criterion directory found.\n\
             Lancez d'abord / Run first:\n\
             cargo bench --bench langmuir_performance -- bench_species_response_curve"
        );
    }

    if all_points.is_empty() {
        anyhow::bail!("Aucune donnée trouvée / No data found");
    }

    // Dédoublonnage par (n_species, solver) si les deux architectures coexistent
    // Deduplication by (n_species, solver) if both architectures coexist
    all_points.sort_by(|a, b| {
        a.n_species
            .cmp(&b.n_species)
            .then_with(|| format!("{:?}", a.solver).cmp(&format!("{:?}", b.solver)))
    });
    all_points.dedup_by(|a, b| a.n_species == b.n_species && a.solver == b.solver);

    Ok(all_points)
}

// =================================================================================================
// Régression log-log / Log-log regression
// =================================================================================================

/// Résultat d'une régression log-log : t = a × n^α
/// *Log-log regression result: t = a × n^α*
#[derive(Debug)]
struct LogLogRegression {
    /// Exposant mesuré (pente log-log) / *Measured exponent (log-log slope)*
    alpha: f64,
    /// log(a) pour prediction / *log(a) for prediction*
    log_a: f64,
}

impl LogLogRegression {
    /// Régression OLS sur les paires (log x, log y)
    /// *OLS regression on pairs (log x, log y)*
    fn fit(pts: &[(f64, f64)]) -> Option<Self> {
        let valid: Vec<(f64, f64)> = pts
            .iter()
            .filter(|&&(x, y)| x > 0.0 && y > 0.0)
            .map(|&(x, y)| (x.ln(), y.ln()))
            .collect();

        if valid.len() < 2 {
            return None;
        }

        let n = valid.len() as f64;
        let sum_x = valid.iter().map(|(x, _)| x).sum::<f64>();
        let sum_y = valid.iter().map(|(_, y)| y).sum::<f64>();
        let sum_xx = valid.iter().map(|(x, _)| x * x).sum::<f64>();
        let sum_xy = valid.iter().map(|(x, y)| x * y).sum::<f64>();

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
// Génération du graphique / Plot generation
// =================================================================================================

/// Génère le graphique SVG complet
/// *Generates the complete SVG plot*
fn generate_plot(all_points: &[DataPoint], output_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let euler_pts: Vec<&DataPoint> = all_points
        .iter()
        .filter(|p| p.solver == Solver::Euler)
        .collect();
    let rk4_pts: Vec<&DataPoint> = all_points
        .iter()
        .filter(|p| p.solver == Solver::Rk4)
        .collect();

    // ── Régressions log-log ───────────────────────────────────────────────
    let euler_xy: Vec<(f64, f64)> = euler_pts
        .iter()
        .map(|p| (p.n_species as f64, p.time_ms))
        .collect();
    let rk4_xy: Vec<(f64, f64)> = rk4_pts
        .iter()
        .map(|p| (p.n_species as f64, p.time_ms))
        .collect();

    let reg_euler = LogLogRegression::fit(&euler_xy);
    let reg_rk4 = LogLogRegression::fit(&rk4_xy);

    // ── Courbe O(n³) théorique calée sur Euler à n_species = N_REF ────────
    // ── Theoretical O(n³) curve anchored on Euler at n_species = N_REF ────
    //
    // t_théo(n) = t_euler(N_REF) × (n / N_REF)³
    // Ce calage isole la forme de la loi (exposant) du coefficient absolu.
    // This anchoring isolates the law shape (exponent) from the absolute coefficient.
    let t_ref = euler_pts
        .iter()
        .find(|p| p.n_species == N_REF)
        .map(|p| p.time_ms)
        .unwrap_or_else(|| euler_pts.first().map(|p| p.time_ms).unwrap_or(1.0));
    let n_ref_f = N_REF as f64;

    // ── Bornes des axes ───────────────────────────────────────────────────
    let x_max = all_points.iter().map(|p| p.n_species).max().unwrap_or(100) as f64;
    let y_max = all_points
        .iter()
        .map(|p| p.time_high_ms)
        .fold(0.0_f64, f64::max)
        * 1.15;

    // ── Backend SVG ───────────────────────────────────────────────────────
    let root = SVGBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(50)
        .x_label_area_size(55)
        .y_label_area_size(90)
        .caption(
            format!(
                "Courbe de réponse n_species — n_points={N_POINTS} fixé\n\
                 Species Response Curve — n_points={N_POINTS} fixed \
                 (parallelism threshold: ops={PARALLEL_THRESHOLD_OPS} → n_species={THRESHOLD_N_SPECIES})"
            ),
            ("sans-serif", 15).into_font(),
        )
        .build_cartesian_2d(0f64..x_max * 1.05, 0f64..y_max)?;

    chart
        .configure_mesh()
        .x_desc("n_species (nombre d'espèces / number of species)")
        .y_desc("Temps moyen / Mean time (ms)")
        .x_label_formatter(&|v| format!("{}", *v as usize))
        .y_label_formatter(&|v| format!("{:.0} ms", v))
        .draw()?;

    // ── Zones de fond : série (bleu pâle) + parallèle (rouge pâle) ────────
    let tx = THRESHOLD_N_SPECIES as f64;
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (tx, y_max)],
        BLUE.mix(0.04).filled(),
    )))?;
    chart.draw_series(std::iter::once(Rectangle::new(
        [(tx, 0.0), (x_max * 1.05, y_max)],
        RED.mix(0.04).filled(),
    )))?;

    // ── Ligne verticale du seuil de parallélisme ──────────────────────────
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(tx, 0.0), (tx, y_max)],
        ShapeStyle {
            color: GREEN.mix(0.7).to_rgba(),
            filled: false,
            stroke_width: 2,
        },
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        format!("Seuil / Threshold\nn_species={THRESHOLD_N_SPECIES}"),
        (tx + x_max * 0.01, y_max * 0.94),
        ("sans-serif", 10).into_font().color(&GREEN.mix(0.8)),
    )))?;

    // ── Courbe O(n³) théorique (pointillés gris) ──────────────────────────
    // ── Theoretical O(n³) curve (grey dashes) ─────────────────────────────
    let theory_curve: Vec<(f64, f64)> = (0..=200)
        .map(|i| {
            let n = n_ref_f + (x_max - n_ref_f) * i as f64 / 200.0;
            let y = t_ref * (n / n_ref_f).powi(3);
            (n, y)
        })
        .filter(|&(_, y)| y <= y_max)
        .collect();

    chart
        .draw_series(theory_curve.windows(2).map(|w| {
            PathElement::new(
                vec![w[0], w[1]],
                ShapeStyle {
                    color: RGBColor(150, 150, 150).mix(0.8).to_rgba(),
                    filled: false,
                    stroke_width: 2,
                },
            )
        }))?
        .label("O(n³) théorique / theoretical")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle {
                    color: RGBColor(150, 150, 150).to_rgba(),
                    filled: false,
                    stroke_width: 2,
                },
            )
        });

    // ── Droite de régression Euler ─────────────────────────────────────────
    if let Some(ref reg) = reg_euler {
        let n_min = euler_pts.first().map(|p| p.n_species as f64).unwrap_or(2.0);
        let reg_curve: Vec<(f64, f64)> = (0..=200)
            .map(|i| {
                let n = n_min + (x_max - n_min) * i as f64 / 200.0;
                (n, reg.predict(n))
            })
            .filter(|&(_, y)| y > 0.0 && y <= y_max)
            .collect();

        chart
            .draw_series(reg_curve.windows(2).map(|w| {
                PathElement::new(
                    vec![w[0], w[1]],
                    ShapeStyle {
                        color: RGBColor(0, 0, 180).mix(0.55).to_rgba(),
                        filled: false,
                        stroke_width: 2,
                    },
                )
            }))?
            .label(format!("Régr. Euler: O(n^{:.2})", reg.alpha))
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle {
                        color: RGBColor(0, 0, 180).mix(0.55).to_rgba(),
                        filled: false,
                        stroke_width: 2,
                    },
                )
            });
    }

    // ── Droite de régression RK4 ──────────────────────────────────────────
    if let Some(ref reg) = reg_rk4 {
        let n_min = rk4_pts.first().map(|p| p.n_species as f64).unwrap_or(2.0);
        let reg_curve: Vec<(f64, f64)> = (0..=200)
            .map(|i| {
                let n = n_min + (x_max - n_min) * i as f64 / 200.0;
                (n, reg.predict(n))
            })
            .filter(|&(_, y)| y > 0.0 && y <= y_max)
            .collect();

        chart
            .draw_series(reg_curve.windows(2).map(|w| {
                PathElement::new(
                    vec![w[0], w[1]],
                    ShapeStyle {
                        color: RGBColor(180, 0, 0).mix(0.55).to_rgba(),
                        filled: false,
                        stroke_width: 2,
                    },
                )
            }))?
            .label(format!("Régr. RK4: O(n^{:.2})", reg.alpha))
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle {
                        color: RGBColor(180, 0, 0).mix(0.55).to_rgba(),
                        filled: false,
                        stroke_width: 2,
                    },
                )
            });
    }

    // ── Barres d'erreur IC 95% ────────────────────────────────────────────
    for p in all_points {
        let x = p.n_species as f64;
        let col = if p.solver == Solver::Euler {
            BLUE.mix(0.35)
        } else {
            RED.mix(0.35)
        };
        let cap = x_max * 0.003;

        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, p.time_low_ms), (x, p.time_high_ms)],
            ShapeStyle {
                color: col.to_rgba(),
                filled: false,
                stroke_width: 1,
            },
        )))?;
        for &y_cap in &[p.time_low_ms, p.time_high_ms] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x - cap, y_cap), (x + cap, y_cap)],
                ShapeStyle {
                    color: col.to_rgba(),
                    filled: false,
                    stroke_width: 1,
                },
            )))?;
        }
    }

    // ── Courbe Euler mesurée (bleue) ──────────────────────────────────────
    let euler_curve: Vec<(f64, f64)> = euler_pts
        .iter()
        .map(|p| (p.n_species as f64, p.time_ms))
        .collect();
    chart
        .draw_series(LineSeries::new(euler_curve.clone(), BLUE.stroke_width(3)))?
        .label("Euler mesuré / Measured Euler")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    chart.draw_series(
        euler_curve
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled())),
    )?;

    // ── Courbe RK4 mesurée (rouge) ────────────────────────────────────────
    let rk4_curve: Vec<(f64, f64)> = rk4_pts
        .iter()
        .map(|p| (p.n_species as f64, p.time_ms))
        .collect();
    chart
        .draw_series(LineSeries::new(rk4_curve.clone(), RED.stroke_width(3)))?
        .label("RK4 mesuré / Measured RK4")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    chart.draw_series(
        rk4_curve
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, RED.filled())),
    )?;

    // ── Annotation du seuil de parallélisme ───────────────────────────────
    // ── Parallelism threshold annotation ─────────────────────────────────
    //
    // Si les deux solveurs ont des mesures à n=9 et n=10, on annote le gain
    // obtenu grâce au parallélisme Rayon.
    // If both solvers have measurements at n=9 and n=10, annotate the Rayon gain.
    let euler_before = euler_pts
        .iter()
        .find(|p| p.n_species == THRESHOLD_N_SPECIES - 1);
    let euler_after = euler_pts
        .iter()
        .find(|p| p.n_species == THRESHOLD_N_SPECIES);

    if let (Some(eb), Some(ea)) = (euler_before, euler_after) {
        // Rapport attendu sans parallélisme : (10/9)³ ≈ 1.37×
        // Expected ratio without parallelism: (10/9)³ ≈ 1.37×
        let expected_ratio =
            (THRESHOLD_N_SPECIES as f64 / (THRESHOLD_N_SPECIES - 1) as f64).powi(3);
        let actual_ratio = ea.time_ms / eb.time_ms;
        let par_effect = actual_ratio / expected_ratio;

        chart.draw_series(std::iter::once(Text::new(
            format!(
                "Rayon actif ici\nratio mesuré: {:.2}×\nO(n³) prédit: {:.2}×\neffet parallèle: {:.2}×",
                actual_ratio, expected_ratio, par_effect
            ),
            (tx + x_max * 0.01, y_max * 0.55),
            ("sans-serif", 10).into_font().color(&RGBColor(0, 120, 0)),
        )))?;

        // Flèche entre n=9 et n=10 sur la courbe Euler
        // Arrow between n=9 and n=10 on the Euler curve
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                ((THRESHOLD_N_SPECIES - 1) as f64, eb.time_ms),
                (THRESHOLD_N_SPECIES as f64, ea.time_ms),
            ],
            ShapeStyle {
                color: RGBColor(0, 140, 0).to_rgba(),
                filled: false,
                stroke_width: 3,
            },
        )))?;
    }

    // ── Légende ───────────────────────────────────────────────────────────
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

// =================================================================================================
// Point d'entrée / Entry point
// =================================================================================================

fn main() -> anyhow::Result<()> {
    let criterion_dir = PathBuf::from("target/criterion");
    let output_path = PathBuf::from("target/plots/species_response_curve.svg");

    println!("📂 Lecture des données Criterion / Reading Criterion data...");

    let all_points = collect_all_points(&criterion_dir)?;

    let euler_pts: Vec<&DataPoint> = all_points
        .iter()
        .filter(|p| p.solver == Solver::Euler)
        .collect();
    let rk4_pts: Vec<&DataPoint> = all_points
        .iter()
        .filter(|p| p.solver == Solver::Rk4)
        .collect();

    // ── Tableau récapitulatif ─────────────────────────────────────────────
    println!(
        "\n{:<12} {:<10} {:<10} {:<12} {:<12} {:<10}",
        "n_species", "ops", "régime", "Euler_ms", "RK4_ms", "ratio"
    );
    println!("{:-<68}", "");

    let n_species_list: Vec<usize> = {
        let mut ns: Vec<usize> = all_points.iter().map(|p| p.n_species).collect();
        ns.sort();
        ns.dedup();
        ns
    };

    for &n_sp in &n_species_list {
        let ops = N_POINTS * n_sp;
        let regime = if ops >= PARALLEL_THRESHOLD_OPS {
            "PARALLÈLE"
        } else {
            "série"
        };
        let e_time = euler_pts
            .iter()
            .find(|p| p.n_species == n_sp)
            .map(|p| p.time_ms);
        let r_time = rk4_pts
            .iter()
            .find(|p| p.n_species == n_sp)
            .map(|p| p.time_ms);

        let ratio_str = match (e_time, r_time) {
            (Some(e), Some(r)) => format!("{:.2}×", r / e),
            _ => "N/A".to_string(),
        };

        println!(
            "{:<12} {:<10} {:<10} {:<12} {:<12} {:<10}",
            n_sp,
            ops,
            regime,
            e_time.map(|t| format!("{:.2}", t)).unwrap_or("-".into()),
            r_time.map(|t| format!("{:.2}", t)).unwrap_or("-".into()),
            ratio_str,
        );
    }

    // ── Régressions log-log ───────────────────────────────────────────────
    let euler_xy: Vec<(f64, f64)> = euler_pts
        .iter()
        .map(|p| (p.n_species as f64, p.time_ms))
        .collect();
    let rk4_xy: Vec<(f64, f64)> = rk4_pts
        .iter()
        .map(|p| (p.n_species as f64, p.time_ms))
        .collect();

    println!("\n📐 Régressions log-log / Log-log regressions (théorique O(n³) = 3.000):");
    if let Some(r) = LogLogRegression::fit(&euler_xy) {
        println!(
            "   Euler : t ∝ n^{:.3}  (écart / deviation: {:+.3})",
            r.alpha,
            r.alpha - 3.0
        );
    }
    if let Some(r) = LogLogRegression::fit(&rk4_xy) {
        println!(
            "   RK4   : t ∝ n^{:.3}  (écart / deviation: {:+.3})",
            r.alpha,
            r.alpha - 3.0
        );
    }

    println!("\n🎨 Génération du graphique / Generating plot...");
    generate_plot(&all_points, &output_path)?;
    Ok(())
}

// =================================================================================================
// Tests unitaires / Unit tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_n_species ───────────────────────────────────────────────────

    #[test]
    fn test_parse_n_species_valid() {
        assert_eq!(parse_n_species("n_sp_10"), Some(10));
        assert_eq!(parse_n_species("n_sp_100"), Some(100));
        assert_eq!(parse_n_species("n_sp_2"), Some(2));
    }

    #[test]
    fn test_parse_n_species_invalid() {
        assert_eq!(parse_n_species("euler"), None);
        assert_eq!(parse_n_species("n_sp_abc"), None);
        assert_eq!(parse_n_species(""), None);
        assert_eq!(parse_n_species("npts_100"), None);
    }

    // ── Régression log-log ────────────────────────────────────────────────

    /// Régression sur y = x³ doit retrouver α = 3.0
    /// *Regression on y = x³ must recover α = 3.0*
    #[test]
    fn test_regression_cubic() {
        let pts: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let x = i as f64 * 5.0;
                (x, x.powi(3))
            })
            .collect();
        let reg = LogLogRegression::fit(&pts).unwrap();
        assert!((reg.alpha - 3.0).abs() < 1e-6, "α={}", reg.alpha);
    }

    /// Moins de 2 points → None
    #[test]
    fn test_regression_too_few_points() {
        assert!(LogLogRegression::fit(&[]).is_none());
        assert!(LogLogRegression::fit(&[(5.0, 10.0)]).is_none());
    }

    /// Valeurs nulles ou négatives sont ignorées sans panique
    /// *Zero or negative values are ignored without panicking*
    #[test]
    fn test_regression_filters_nonpositive() {
        let pts = vec![(0.0, 10.0), (-1.0, 5.0), (10.0, 100.0), (20.0, 800.0)];
        // Seulement 2 points valides : (10, 100) et (20, 800) → α ≈ 3
        // Only 2 valid points: (10, 100) and (20, 800) → α ≈ 3
        let reg = LogLogRegression::fit(&pts).unwrap();
        assert!((reg.alpha - 3.0).abs() < 0.01, "α={}", reg.alpha);
    }

    // ── Seuil de parallélisme ─────────────────────────────────────────────

    /// Le seuil est bien à n_species = 10 avec n_points = 100
    /// *The threshold is at n_species = 10 with n_points = 100*
    #[test]
    fn test_threshold_at_n_species_10() {
        assert_eq!(THRESHOLD_N_SPECIES, 10);
        assert!((9 * N_POINTS) < PARALLEL_THRESHOLD_OPS);
        assert!((10 * N_POINTS) >= PARALLEL_THRESHOLD_OPS);
    }

    // ── Courbe O(n³) ─────────────────────────────────────────────────────

    /// La courbe théorique O(n³) calée sur (N_REF, t_ref) est exacte au point de calage
    /// *The O(n³) curve anchored at (N_REF, t_ref) is exact at the anchoring point*
    #[test]
    fn test_theory_curve_exact_at_anchor() {
        let t_ref = 1.5_f64;
        let n_ref = N_REF as f64;
        let n_test = n_ref;
        let t_theo = t_ref * (n_test / n_ref).powi(3);
        assert!((t_theo - t_ref).abs() < 1e-12);
    }

    /// Ratio O(n³) entre n=2 et n=10 : (10/2)³ = 125
    /// *O(n³) ratio between n=2 and n=10: (10/2)³ = 125*
    #[test]
    fn test_theory_ratio_2_to_10() {
        let ratio = (10.0_f64 / N_REF as f64).powi(3);
        assert!((ratio - 125.0).abs() < 1e-6);
    }
}
