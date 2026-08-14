//! # chrom-rs — binary entry point
//!
//! Delegates entirely to the [`cli`](chrom_rs::cli) layer built on
//! `dynamic-cli`. All simulation logic lives in the library crate.
//!
//! # Usage
//!
//! ```text
//! chrom-rs run [--project-dir <dir>]
//!              --model    <file.yml>              (or --source model    file=<file.yml>)
//!              --scenario <file.yml>              (or --source scenario file=<file.yml>)
//!              --solver   <file.yml>              (or --source solver   file=<file.yml>)
//!              [--output-csv   <file.csv>]        (or --output csv  file=<file.csv>)
//!              [--output-plot  <file.png|.svg>]   (or --output png  file=<file.png>,
//!                                                      --output svg  file=<file.svg>)
//!              [--export-json  <file.json>]       (or --output json file=<file.json>)
//!
//! chrom-rs check [--project-dir <dir>]
//!                [--source model    file=<file.yml>]
//!                [--source scenario file=<file.yml>]
//!                [--source solver   file=<file.yml>]
//! ```
//!
//! `run` accepts either the legacy scalar options or the repeatable
//! `--source`/`--output` syntax for each role — never both for the same
//! role in the same invocation (aliases: `simulate`, `solve`). `check`
//! only has the repeatable syntax, and every role is optional: with none
//! given it just lists the project directory's config-like files; with
//! any given, it validates exactly those.

fn main() {
    // Standalone binary variant (Variant B) — see
    // `src/output/visualization/fonts.rs` for why this is needed before
    // anything else runs: `ab_glyph` has no name-based font discovery, so
    // "sans-serif" must be registered explicitly before any chart renders.
    chrom_rs::output::register_fonts();

    let result = chrom_rs::cli::build_app()
        .map_err(|e| format!("initialisation error: {e}"))
        .and_then(|app| app.run().map_err(|e| format!("runtime error: {e}")));

    if let Err(e) = result {
        eprintln!("chrom-rs: {e}");
        std::process::exit(1);
    }
}
