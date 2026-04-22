//! # chrom-rs — binary entry point
//!
//! Delegates entirely to the [`cli`](chrom_rs::cli) layer built on
//! `dynamic-cli`. All simulation logic lives in the library crate.
//!
//! # Usage
//!
//! ```text
//! chrom-rs run --model model.yml --scenario scenario.yml --solver solver.yml
//!              [--project-dir <dir>]
//!              [--output-csv results.csv]
//!              [--output-plot chromatogram.png]
//!              [--export-json results.json]
//! ```

fn main() {
    let result = chrom_rs::cli::build_app()
        .map_err(|e| format!("initialisation error: {e}"))
        .and_then(|app| app.run().map_err(|e| format!("runtime error: {e}")));

    if let Err(e) = result {
        eprintln!("chrom-rs: {e}");
        std::process::exit(1);
    }
}
