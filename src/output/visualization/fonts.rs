//! Embedded fallback font — standalone binary variant, no system font
//! database required at runtime.
//!
//! `plotters`' default feature set (`ttf`) resolves font family names like
//! `"sans-serif"` via `font-kit`, which on Linux goes through `fontconfig`/
//! `freetype` — real dynamic-library dependencies at *runtime*, not just at
//! build time (confirmed via `cargo tree -e normal -i font-kit`: `chrom-rs
//! → plotters → font-kit`, not a `criterion`/dev-only artifact). For a
//! "download and run" distributable binary, that's a dependency the target
//! machine may not have installed.
//!
//! This crate instead builds `plotters` with the `ab_glyph` feature — a
//! pure-Rust font renderer with zero system dependency — and embeds one
//! specific font directly into the compiled binary via `include_bytes!`.
//! The trade-off: `ab_glyph` has no name-based system font *discovery* the
//! way `fontconfig` does, so every family name a chart uses (this crate
//! only ever uses `"sans-serif"`) must be explicitly registered with the
//! embedded font's bytes before any chart is rendered — every public
//! `plot_*` function in this crate does that itself, on entry, guarded by
//! `std::sync::Once` so it's cheap after the first call; see
//! [`register_fonts`] for why this isn't left to callers to remember.
//!
//! # Font and license
//!
//! [DejaVu Sans](https://dejavu-fonts.github.io/), Regular weight only —
//! the Bitstream Vera Fonts license (see `assets/fonts/LICENSE-DEJAVU.txt`) permits
//! embedding and redistribution in compiled form; the only restriction is
//! that a *modified* font can't keep the "Bitstream"/"Vera" names, which
//! doesn't apply here since the font is embedded unmodified.
//!
//! Only the `Normal` style is registered. `plotters` falls back to
//! `FontStyle::Normal` for any unregistered style under the same family
//! name (see `plotters`' `ab_glyph` backend, `FontMap::get_fallback`), so
//! there's no need to bundle a Bold/Oblique variant unless a chart actually
//! requests one — none does today.

use std::sync::Once;

use plotters::style::{FontStyle, register_font};

/// DejaVu Sans, Regular — embedded so `"sans-serif"` resolves without a
/// system font database. See the module-level doc comment for the license.
static DEJAVU_SANS: &[u8] = include_bytes!("../../../assets/fonts/DejaVuSans.ttf");

/// Guards the actual registration so it only runs once per process, no
/// matter how many call sites invoke [`register_fonts`] — see that
/// function's doc comment.
static REGISTER_ONCE: Once = Once::new();

/// Registers the embedded font under the `"sans-serif"` family — the name
/// every chart in this crate uses via `.into_font()`.
///
/// Every public `plot_*` function in this crate calls this at its own
/// entry point, so callers never *need* to call it themselves — including
/// `#[test]` functions that call `plot_chromatogram` and friends directly,
/// which is exactly the case that broke before this used `std::sync::Once`
/// (see the module-level doc comment: `main()` in `src/main.rs` and the
/// `tools/plot_*.rs` binaries called this explicitly, but the test suite
/// calls the plotting functions directly and never went through `main()`
/// at all, so `"sans-serif"` was never registered — `FontError::FontUnavailable`
/// at test time). Calling it explicitly at the top of `main()` as well is
/// harmless — `std::sync::Once` makes every call after the first a no-op —
/// and left in place there mainly as documentation of intent.
///
/// # Panics
///
/// Panics (on the first call only) if the embedded font bytes fail to
/// parse as a valid TrueType font. This can only happen if
/// `assets/fonts/DejaVuSans.ttf` is replaced with an invalid file — the
/// bytes are fixed at compile time via `include_bytes!`, so this is a
/// build-time invariant, not a runtime condition that depends on anything
/// the caller controls.
pub fn register_fonts() {
    REGISTER_ONCE.call_once(|| {
        if register_font("sans-serif", FontStyle::Normal, DEJAVU_SANS).is_err() {
            panic!(
                "embedded assets/fonts/DejaVuSans.ttf is not a valid TTF — this should never fail"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_fonts_does_not_panic() {
        register_fonts();
    }

    #[test]
    fn test_register_fonts_is_idempotent() {
        register_fonts();
        register_fonts();
    }
}
