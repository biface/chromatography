//! Visualisation du balayage de stabilité CFL (`examples/cfl_stability_scan.rs`).
//! *Visualisation of the CFL stability scan (`examples/cfl_stability_scan.rs`).*
//!
//! Run with `cargo run --release --example cfl_stability_scan` then
//! `cargo run --release --bin plot_cfl_stability`; output lands in
//! `target/plots/cfl_stability.svg`.
//!
//! Reads `cfl_stability_scan.json` from the system temp directory — the
//! same file `examples/cfl_stability_scan.rs` writes to.
//!
//! # Visual elements
//!
//! Deux panneaux empilés, un par solveur (Euler=rouge, RK4=bleu, même
//! convention de couleurs que `tools/plot_stiffness_convergence.rs`) :
//!
//! 1. **Ordre de grandeur du profil final** — `log10(max(|min|, |max|))`
//!    vs CFL (log). Montre où chaque solveur bascule d'un profil
//!    physiquement raisonnable vers des valeurs qui explosent.
//! 2. **`tail_ratio`** — fraction de la norme non résorbée en fin de
//!    simulation, vs CFL (log). Reste proche de 0 tant que la simulation
//!    est saine, saute vers 1 à l'instabilité.
//!
//! Les points au verdict `OK` sont des cercles pleins de rayon 4 ; les
//! points signalés (`SUSPECT (...)`ou `NaN/Inf`) sont des cercles pleins
//! de rayon 7 — même famille de marqueur que le reste du dépôt (voir
//! `tools/plot_stiffness_convergence.rs`), pas de nouvelle forme
//! introduite pour rester sur des primitives déjà vérifiées ailleurs dans
//! ce dépôt.
//! *Two stacked panels, one per solver (Euler=red, RK4=blue, same color
//! convention as `tools/plot_stiffness_convergence.rs`):*
//!
//! *1. **Magnitude of the final profile** — `log10(max(|min|, |max|))` vs
//!    CFL (log). Shows where each solver flips from a physically
//!    reasonable profile to blown-up values.*
//! *2. **`tail_ratio`** — fraction of the norm left unresolved at the end
//!    of the simulation, vs CFL (log). Stays near 0 while the run is
//!    healthy, jumps to 1 at instability.*
//!
//! *`OK`-verdict points are solid circles of radius 4; flagged points
//! (`SUSPECT (...)` or `NaN/Inf`) are solid circles of radius 7 — same
//! marker family as the rest of the repository (see
//! `tools/plot_stiffness_convergence.rs`), no new shape introduced so as
//! to stay on primitives already verified elsewhere in this repository.*
//!
//! # Cargo.toml
//!
//! ```toml
//! [[bin]]
//! name = "plot_cfl_stability"
//! path = "tools/plot_cfl_stability.rs"
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

struct Row {
    cfl: f64,
    /// `log10(max(|final_min|, |final_max|))`. `None` si `final_min`/`final_max`
    /// sont absents du JSON (cas `all_finite=false` sérialisé — voir la note
    /// dans `examples/cfl_stability_scan.rs`: serde_json sérialise NaN/Inf en
    /// `null`, jamais observé en pratique jusqu'à CFL=50 mais géré ici sans
    /// paniquer).
    /// *`None` if `final_min`/`final_max` are absent from the JSON (the
    /// `all_finite=false` case serialized — see the note in
    /// `examples/cfl_stability_scan.rs`: serde_json serializes NaN/Inf as
    /// `null`, never observed in practice up to CFL=50 but handled here
    /// without panicking.)*
    magnitude_log10: Option<f64>,
    tail_ratio: f64,
    suspect: bool,
}

fn report_path() -> PathBuf {
    std::env::temp_dir().join("cfl_stability_scan.json")
}

fn read_report(path: &Path) -> anyhow::Result<Vec<(String, Vec<Row>)>> {
    let content = fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "cannot read {}: {e}\nRun first: cargo run --release --example cfl_stability_scan",
            path.display()
        )
    })?;
    let root: Value = serde_json::from_str(&content)?;
    let rows_json = root
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("{}: expected top-level 'rows' array", path.display()))?;

    let mut by_solver: Vec<(String, Vec<Row>)> = Vec::new();

    for r in rows_json {
        let cfl = r
            .get("cfl")
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("row missing 'cfl'"))?;
        let solver = r
            .get("solver")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("row missing 'solver'"))?
            .to_string();
        let verdict = r
            .get("verdict")
            .and_then(Value::as_str)
            .unwrap_or("verdict manquant");
        let suspect = verdict != "OK";

        let final_min = r.get("final_min").and_then(Value::as_f64);
        let final_max = r.get("final_max").and_then(Value::as_f64);
        let magnitude_log10 = match (final_min, final_max) {
            (Some(mn), Some(mx)) => Some(mn.abs().max(mx.abs()).max(1e-300).log10()),
            _ => {
                eprintln!(
                    "cfl={cfl} solver={solver}: final_min/final_max absents du JSON \
                     (NaN/Inf) — point omis du panneau magnitude / \
                     point omitted from the magnitude panel"
                );
                None
            }
        };

        let tail_ratio = r.get("tail_ratio").and_then(Value::as_f64).unwrap_or(0.0);

        let row = Row {
            cfl,
            magnitude_log10,
            tail_ratio,
            suspect,
        };

        // `.iter_mut().find()+push()` dans un `match` est un cas connu que
        // le borrow checker rejette (pas résolu par NLL, cf. RFC Polonius) ;
        // on cherche l'index d'abord (emprunt immutable qui se termine),
        // puis on mute — aucune double-emprunt.
        // `.iter_mut().find()+push()` inside a `match` is a known case
        // rejected by the borrow checker (not solved by NLL, cf. the
        // Polonius RFC); find the index first (immutable borrow that ends),
        // then mutate — no double borrow.
        match by_solver.iter().position(|(name, _)| *name == solver) {
            Some(i) => by_solver[i].1.push(row),
            None => by_solver.push((solver, vec![row])),
        }
    }

    // Ordre alphabétique ⇒ euler avant rk4, cohérent avec la palette
    // [RED, BLUE] ci-dessous (euler=rouge, rk4=bleu).
    // Alphabetical order ⇒ euler before rk4, consistent with the
    // [RED, BLUE] palette below (euler=red, rk4=blue).
    by_solver.sort_by(|a, b| a.0.cmp(&b.0));
    for (_, rows) in &mut by_solver {
        rows.sort_by(|a, b| a.cfl.partial_cmp(&b.cfl).unwrap());
    }

    Ok(by_solver)
}

// =================================================================================================
// Plot generation
// =================================================================================================

const COLORS: [RGBColor; 2] = [RED, BLUE];

fn generate_plot(by_solver: &[(String, Vec<Row>)], output_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let all_cfl: Vec<f64> = by_solver
        .iter()
        .flat_map(|(_, rows)| rows.iter().map(|r| r.cfl))
        .collect();
    let cfl_min = all_cfl.iter().cloned().fold(f64::INFINITY, f64::min);
    let cfl_max = all_cfl.iter().cloned().fold(0.0f64, f64::max);
    let x_log_min = cfl_min.log10() - 0.1;
    let x_log_max = cfl_max.log10() + 0.1;

    let all_mag: Vec<f64> = by_solver
        .iter()
        .flat_map(|(_, rows)| rows.iter().filter_map(|r| r.magnitude_log10))
        .collect();
    let mag_min = all_mag.iter().cloned().fold(f64::INFINITY, f64::min) - 5.0;
    let mag_max = all_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 5.0;

    let root = SVGBackend::new(output_path, (1000, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((2, 1));

    // ── Panneau 1 : ordre de grandeur du profil final ──────────────────
    // ── Panel 1: magnitude of the final profile ─────────────────────────
    {
        let mut chart = ChartBuilder::on(&areas[0])
            .margin(30)
            .x_label_area_size(45)
            .y_label_area_size(70)
            .caption(
                "Ordre de grandeur du profil final vs CFL — bench_cfl_stability",
                ("sans-serif", 18).into_font(),
            )
            .build_cartesian_2d(x_log_min..x_log_max, mag_min..mag_max)?;

        chart
            .configure_mesh()
            .x_desc("CFL (log scale)")
            .y_desc("log10(max(|min|, |max|))")
            .x_label_formatter(&|x| format!("{:.1}", 10f64.powf(*x)))
            .draw()?;

        for (i, (name, rows)) in by_solver.iter().enumerate() {
            let color = COLORS[i % COLORS.len()];
            // Construits ensemble en un seul passage pour rester alignés —
            // filtrer `series` séparément puis `zip` avec `rows` non filtré
            // désaligne silencieusement dès qu'une ligne a
            // `magnitude_log10=None` (cas NaN/Inf).
            // Built together in a single pass to stay aligned — filtering
            // `series` separately then `zip`-ing against unfiltered `rows`
            // silently misaligns as soon as one row has
            // `magnitude_log10=None` (NaN/Inf case).
            let points: Vec<(&Row, f64, f64)> = rows
                .iter()
                .filter_map(|r| r.magnitude_log10.map(|m| (r, r.cfl.log10(), m)))
                .collect();

            chart
                .draw_series(LineSeries::new(
                    points.iter().map(|&(_, x, y)| (x, y)),
                    color.stroke_width(2),
                ))?
                .label(name.clone())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                });

            for &(r, x, y) in &points {
                let radius = if r.suspect { 7 } else { 4 };
                chart.draw_series(std::iter::once(Circle::new((x, y), radius, color.filled())))?;
            }
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    // ── Panneau 2 : tail_ratio ───────────────────────────────────────────
    // ── Panel 2: tail_ratio ───────────────────────────────────────────────
    {
        let mut chart = ChartBuilder::on(&areas[1])
            .margin(30)
            .x_label_area_size(45)
            .y_label_area_size(70)
            .caption(
                "Fraction de la norme non résorbée en fin de simulation vs CFL",
                ("sans-serif", 18).into_font(),
            )
            .build_cartesian_2d(x_log_min..x_log_max, -0.05..1.10)?;

        chart
            .configure_mesh()
            .x_desc("CFL (log scale)")
            .y_desc("tail_ratio")
            .x_label_formatter(&|x| format!("{:.1}", 10f64.powf(*x)))
            .draw()?;

        for (i, (name, rows)) in by_solver.iter().enumerate() {
            let color = COLORS[i % COLORS.len()];
            let series: Vec<(f64, f64)> =
                rows.iter().map(|r| (r.cfl.log10(), r.tail_ratio)).collect();

            chart
                .draw_series(LineSeries::new(
                    series.iter().copied(),
                    color.stroke_width(2),
                ))?
                .label(name.clone())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                });

            for (r, (x, y)) in rows.iter().zip(series.iter().copied()) {
                let radius = if r.suspect { 7 } else { 4 };
                chart.draw_series(std::iter::once(Circle::new((x, y), radius, color.filled())))?;
            }
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

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

    let path = report_path();
    println!("Reading {}...", path.display());
    let by_solver = read_report(&path)?;

    for (name, rows) in &by_solver {
        println!("\n{name}");
        println!(
            "  {:>6} {:>14} {:>10}  verdict",
            "CFL", "log10(mag)", "tail_ratio"
        );
        for r in rows {
            let mag_str = r
                .magnitude_log10
                .map(|m| format!("{m:.2}"))
                .unwrap_or_else(|| "—".to_string());
            println!(
                "  {:>6.1} {:>14} {:>10.4}  {}",
                r.cfl,
                mag_str,
                r.tail_ratio,
                if r.suspect { "SUSPECT" } else { "OK" }
            );
        }
    }

    let output_path = PathBuf::from("target/plots/cfl_stability.svg");
    println!("\nGenerating plot...");
    generate_plot(&by_solver, &output_path)?;
    Ok(())
}
