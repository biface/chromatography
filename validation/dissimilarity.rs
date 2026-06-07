//! Surface resolution and dissimilarity criterion
//!
//! # Background
//!
//! The **surface resolution** $R_{sf}$ (Eq. 7.1, Felinger & Guiochon) compares two
//! normalised chromatograms $Y_1(t)$ and $Y_2(t)$ by integrating the absolute
//! difference of their normalised profiles:
//!
//! $$R_{sf}(Y_1, Y_2) = \int_0^\infty |Y_1(t) - Y_2(t)|\, dt$$
//!
//! where each $Y_k$ is normalised so that $\int Y_k\, dt = 1$.
//!
//! The **dissimilarity criterion** $\Delta^{M_2}_{M_1}$ then compares the outlet
//! profiles produced by two different models (or implementations) for the same
//! physical case:
//!
//! $$\Delta^{M_2}_{M_1} = R_{sf}\!\left(Y^{M_1}_A,\, Y^{M_2}_A\right)$$
//!
//! Interpretation thresholds (linear adsorption, from Figure 7.1):
//! - $\Delta < 0.05$ → chromatograms are **indistinguishable**
//! - $\Delta < 0.10$ → models are **scientifically equivalent**
//! - $\Delta \geq 0.10$ → significant divergence — investigation required
//!
//! # Implementation notes
//!
//! The two signals being compared may come from different solvers with different
//! time grids.  This module handles heterogeneous grids by building a merged time
//! axis and interpolating each signal linearly onto it before integration.
//!
//! All functions operate on plain slices and return `f64`; they have no dependency
//! on the rest of chrom-rs and can be used in isolation.

// =================================================================================================
// Normalisation
// =================================================================================================

/// Normalise a concentration profile so that its integral over time equals 1.
///
/// Uses the trapezoidal rule to compute the area $S = \int c(t)\, dt$ and
/// returns $Y_i = c_i / S$ for each sample.
///
/// # Arguments
///
/// * `times`          — time grid \[s\], strictly increasing, length $\geq 2$
/// * `concentrations` — sampled concentrations \[mol/L\], same length as `times`
///
/// # Panics
///
/// Panics if `times` and `concentrations` have different lengths or if the
/// computed area is zero (signal is identically zero).
///
/// # Returns
///
/// Normalised profile $Y$ such that $\int Y\, dt \approx 1$.
pub fn normalize(times: &[f64], concentrations: &[f64]) -> Vec<f64> {
    assert_eq!(
        times.len(),
        concentrations.len(),
        "times and concentrations must have the same length"
    );
    let area = trapezoid(times, concentrations);
    assert!(
        area.abs() > f64::EPSILON,
        "cannot normalise a zero signal (area = {area})"
    );
    concentrations.iter().map(|c| c / area).collect()
}

// =================================================================================================
// Trapezoidal integration
// =================================================================================================

/// Integrate $f$ over `times` using the trapezoidal rule.
///
/// $$\int_{t_0}^{t_n} f(t)\, dt \approx \sum_{i=0}^{n-1} \frac{f_i + f_{i+1}}{2} \cdot \Delta t_i$$
///
/// # Arguments
///
/// * `times`  — abscissas, strictly increasing
/// * `values` — ordinates, same length as `times`
pub fn trapezoid(times: &[f64], values: &[f64]) -> f64 {
    assert_eq!(times.len(), values.len());
    times
        .windows(2)
        .zip(values.windows(2))
        .map(|(t, v)| 0.5 * (v[0] + v[1]) * (t[1] - t[0]))
        .sum()
}

// =================================================================================================
// Linear interpolation
// =================================================================================================

/// Linearly interpolate `values` (sampled at `times`) at a new point `t`.
///
/// Clamps to the boundary value outside the original range.
///
/// # Arguments
///
/// * `times`  — original abscissas, strictly increasing
/// * `values` — original ordinates
/// * `t`      — query point
fn interpolate_at(times: &[f64], values: &[f64], t: f64) -> f64 {
    // Clamp outside range
    if t <= times[0] {
        return values[0];
    }
    if t >= *times.last().unwrap() {
        return *values.last().unwrap();
    }
    // Binary search for the bracketing interval
    let idx = times.partition_point(|&x| x <= t).saturating_sub(1);
    let idx = idx.min(times.len() - 2);
    let dt = times[idx + 1] - times[idx];
    if dt < f64::EPSILON {
        return values[idx];
    }
    let alpha = (t - times[idx]) / dt;
    values[idx] * (1.0 - alpha) + values[idx + 1] * alpha
}

/// Resample `values` (on `src_times`) onto a new time grid `dst_times`.
///
/// Each output point is computed by linear interpolation; boundary values are
/// clamped.
fn resample(src_times: &[f64], src_values: &[f64], dst_times: &[f64]) -> Vec<f64> {
    dst_times
        .iter()
        .map(|&t| interpolate_at(src_times, src_values, t))
        .collect()
}

// =================================================================================================
// Merged time grid
// =================================================================================================

/// Build a merged, sorted, deduplicated time grid from two grids.
///
/// Points closer than `tol` are considered identical and merged.
fn merge_grids(t1: &[f64], t2: &[f64], tol: f64) -> Vec<f64> {
    let mut merged: Vec<f64> = t1.iter().chain(t2.iter()).copied().collect();
    merged.sort_by(|a, b| a.partial_cmp(b).unwrap());
    merged.dedup_by(|a, b| (*a - *b).abs() < tol);
    merged
}

// =================================================================================================
// Surface resolution / dissimilarity criterion
// =================================================================================================

/// Compute the surface resolution $R_{sf}$ between two concentration profiles.
///
/// Both profiles are first normalised to unit area, then resampled onto a
/// merged time grid, and finally integrated by the trapezoidal rule:
///
/// $$R_{sf} = \int |Y_1(t) - Y_2(t)|\, dt$$
///
/// The two signals may have different time grids — they are reconciled by
/// linear interpolation onto the union grid.
///
/// # Arguments
///
/// * `t1`, `c1` — time grid and concentrations for signal 1 (e.g. chrom-rs)
/// * `t2`, `c2` — time grid and concentrations for signal 2 (e.g. Python reference)
///
/// # Returns
///
/// $R_{sf} \in [0, 2]$.  Values below 0.10 indicate scientific equivalence.
///
/// # Panics
///
/// Panics if either signal has zero area (identically zero concentration).
pub fn rsf(t1: &[f64], c1: &[f64], t2: &[f64], c2: &[f64]) -> f64 {
    // Normalise both signals to unit area
    let y1 = normalize(t1, c1);
    let y2 = normalize(t2, c2);

    // Merge the two time grids
    // Tolerance: 1% of the smallest interval in either grid
    let min_dt = t1
        .windows(2)
        .chain(t2.windows(2))
        .map(|w| (w[1] - w[0]).abs())
        .fold(f64::INFINITY, f64::min);
    let tol = min_dt * 0.01;
    let grid = merge_grids(t1, t2, tol);

    // Resample both signals onto the merged grid
    let y1_merged = resample(t1, &y1, &grid);
    let y2_merged = resample(t2, &y2, &grid);

    // Integrate |Y1 - Y2|
    let diff: Vec<f64> = y1_merged
        .iter()
        .zip(y2_merged.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    trapezoid(&grid, &diff)
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
            .collect()
    }

    fn gaussian_signal(times: &[f64], center: f64, sigma: f64) -> Vec<f64> {
        times
            .iter()
            .map(|&t| {
                let x = (t - center) / sigma;
                (-0.5 * x * x).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
            })
            .collect()
    }

    // ── normalize ────────────────────────────────────────────────────────────

    #[test]
    fn normalize_unit_gaussian() {
        // A Gaussian already normalised to unit area should remain so
        let t = linspace(0.0, 100.0, 2000);
        let c = gaussian_signal(&t, 50.0, 5.0);
        let y = normalize(&t, &c);
        let area = trapezoid(&t, &y);
        assert!(
            (area - 1.0).abs() < 1e-4,
            "expected area ≈ 1, got {area:.6}"
        );
    }

    #[test]
    fn normalize_scales_signal() {
        // A signal with area 3.0 should be scaled by 1/3
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let c = vec![0.0, 2.0, 2.0, 0.0]; // trapezoid area = 4.0
        let y = normalize(&t, &c);
        let area = trapezoid(&t, &y);
        assert!((area - 1.0).abs() < 1e-10, "area = {area}");
    }

    // ── trapezoid ────────────────────────────────────────────────────────────

    #[test]
    fn trapezoid_constant_function() {
        // ∫₀¹ 3 dt = 3.0
        let t = linspace(0.0, 1.0, 1000);
        let v: Vec<f64> = t.iter().map(|_| 3.0).collect();
        let result = trapezoid(&t, &v);
        assert!((result - 3.0).abs() < 1e-6, "result = {result}");
    }

    #[test]
    fn trapezoid_linear_function() {
        // ∫₀¹ t dt = 0.5
        let t = linspace(0.0, 1.0, 10_000);
        let v: Vec<f64> = t.clone();
        let result = trapezoid(&t, &v);
        assert!((result - 0.5).abs() < 1e-6, "result = {result}");
    }

    // ── interpolate_at ───────────────────────────────────────────────────────

    #[test]
    fn interpolate_midpoint() {
        let t = vec![0.0, 1.0, 2.0];
        let v = vec![0.0, 1.0, 4.0];
        // Midpoint between t=1 and t=2: linear → 2.5
        let result = interpolate_at(&t, &v, 1.5);
        assert!((result - 2.5).abs() < 1e-10, "result = {result}");
    }

    #[test]
    fn interpolate_clamps_below() {
        let t = vec![1.0, 2.0];
        let v = vec![5.0, 10.0];
        assert_eq!(interpolate_at(&t, &v, 0.0), 5.0);
    }

    #[test]
    fn interpolate_clamps_above() {
        let t = vec![1.0, 2.0];
        let v = vec![5.0, 10.0];
        assert_eq!(interpolate_at(&t, &v, 3.0), 10.0);
    }

    // ── rsf ──────────────────────────────────────────────────────────────────

    #[test]
    fn rsf_identical_signals_is_zero() {
        // Comparing a signal with itself must give exactly 0
        let t = linspace(0.0, 100.0, 500);
        let c = gaussian_signal(&t, 50.0, 5.0);
        let result = rsf(&t, &c, &t, &c);
        assert!(result < 1e-10, "rsf = {result:.2e}");
    }

    #[test]
    fn rsf_disjoint_signals_approaches_two() {
        // Two non-overlapping unit Gaussians → Rsf ≈ 2
        let t = linspace(0.0, 200.0, 5000);
        let c1 = gaussian_signal(&t, 50.0, 2.0);
        let c2 = gaussian_signal(&t, 150.0, 2.0);
        let result = rsf(&t, &c1, &t, &c2);
        assert!(
            (result - 2.0).abs() < 0.01,
            "expected ≈ 2.0, got {result:.4}"
        );
    }

    #[test]
    fn rsf_slightly_shifted_below_threshold() {
        // Two Gaussians with increasing shifts — checks monotonicity of Rsf.
        // A 0.2 σ shift gives Rsf ≈ 0.16, a 1.0 σ shift gives Rsf ≈ 0.63.
        // Both are below 2.0 (disjoint) and ordered: small shift < large shift.
        let t = linspace(0.0, 100.0, 5000);
        let c1 = gaussian_signal(&t, 50.0, 5.0);
        let c2_small = gaussian_signal(&t, 51.0, 5.0); // 0.2 σ shift
        let c2_large = gaussian_signal(&t, 55.0, 5.0); // 1.0 σ shift
        let rsf_small = rsf(&t, &c1, &t, &c2_small);
        let rsf_large = rsf(&t, &c1, &t, &c2_large);
        assert!(
            rsf_small < rsf_large,
            "larger shift should give larger Rsf: {rsf_small:.4} vs {rsf_large:.4}"
        );
        assert!(rsf_small < 0.5, "rsf = {rsf_small:.4} should be < 0.5");
        assert!(rsf_large < 1.0, "rsf = {rsf_large:.4} should be < 1.0");
    }

    #[test]
    fn rsf_heterogeneous_grids() {
        // Same Gaussian, different sampling densities — result must remain < 1e-3
        let t1 = linspace(0.0, 100.0, 200);
        let t2 = linspace(0.0, 100.0, 5000);
        let c1 = gaussian_signal(&t1, 50.0, 5.0);
        let c2 = gaussian_signal(&t2, 50.0, 5.0);
        let result = rsf(&t1, &c1, &t2, &c2);
        assert!(
            result < 1e-3,
            "rsf on heterogeneous grids = {result:.4e}, expected < 1e-3"
        );
    }

    #[test]
    fn rsf_symmetric() {
        // Rsf(Y1, Y2) == Rsf(Y2, Y1)
        let t = linspace(0.0, 100.0, 1000);
        let c1 = gaussian_signal(&t, 45.0, 4.0);
        let c2 = gaussian_signal(&t, 55.0, 6.0);
        let r12 = rsf(&t, &c1, &t, &c2);
        let r21 = rsf(&t, &c2, &t, &c1);
        assert!((r12 - r21).abs() < 1e-10, "asymmetry: {r12:.6} vs {r21:.6}");
    }
}
