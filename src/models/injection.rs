//! Temporal injection profiles for chromatography
//!
//! Defines how concentration at the column inlet varies with TIME.
//!
//! # Difference from Spatial Injection
//!
//! - **Spatial injection** (injection.rs): C(z, t=0) - initial distribution in space
//! - **Temporal injection** (this file): C(z=0, t) - inlet concentration over time
//!
//! # Use Case
//!
//! This is what happens in real chromatography:
//! - Sample is injected at inlet (z=0) for a certain duration
//! - Injection can be instantaneous (Dirac) or gradual (Gaussian)
//! - After injection ends, inlet returns to baseline
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::models::TemporalInjection;
//! // Gaussian injection: peak at t=10s, width 3s, height 0.1 mol/L
//! let injection = TemporalInjection::gaussian(10.0, 3.0, 0.1);
//!
//! // At different times:
//! assert!(injection.evaluate(0.0) < 1e-3);      // Before injection (nearly zero)
//! assert!((injection.evaluate(10.0) - 0.1).abs() < 1e-10);     // Peak
//! assert!(injection.evaluate(20.0) < 1e-3);    // After injection (nearly zero)
//! ```

use serde::{Deserialize, Deserializer, Serialize, Serializer};
/// Temporal injection profile at column inlet
///
/// Defines how C(z=0, t) varies with time.
///
/// # Types
///
/// - **Dirac**: Instantaneous injection at a single time point
/// - **Gaussian**: Smooth injection over time with bell-shaped profile
/// - **Rectangle**: Constant injection for a duration
/// - **Custom**: User-defined temporal profile
use std::sync::Arc;

// ==================== Serialisation snapshot (internal) ====================

/// Internal representation for serde serialisation of [`TemporalInjection`].
///
/// Mirrors all serialisable variants of [`TemporalInjection`].
/// [`TemporalInjection::Custom`] is excluded — closures cannot be serialised.
///
/// The `type` field is used as a tag in the JSON/YAML output:
/// ```json
/// { "type": "Dirac", "time": 5.0, "amount": 0.1 }
/// ```
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum TemporalInjectionSnapshot {
    Dirac {
        time: f64,
        amount: f64,
    },
    Gaussian {
        center: f64,
        width: f64,
        peak_concentration: f64,
    },
    Rectangle {
        start: f64,
        end: f64,
        concentration: f64,
    },
    None,
}

/// Temporal injection profile at the column inlet ($z = 0$).
///
/// Defines how the inlet concentration $C(t, z=0)$ varies over time.
/// The solver evaluates the profile at each time step and applies it as
/// a boundary condition.
///
/// All variants except [`TemporalInjection::Custom`] are serialisable and
/// can be specified in `scenario.yml`.
///
/// # Choosing a profile
///
/// | Variant | Use case |
/// |---------|----------|
/// | [`Dirac`](TemporalInjection::Dirac) | Instantaneous pulse (sharp injection) |
/// | [`Gaussian`](TemporalInjection::Gaussian) | Smooth finite-width pulse |
/// | [`Rectangle`](TemporalInjection::Rectangle) | Constant-concentration step injection |
/// | [`Custom`](TemporalInjection::Custom) | Arbitrary closure (not serialisable) |
/// | [`None`](TemporalInjection::None) | No injection (zero inlet concentration) |
pub enum TemporalInjection {
    /// Dirac delta injection at a single time point
    ///
    /// # Parameters
    ///
    /// - `time` : Time of injection \[s\]
    /// - `amount` : Amount injected (integral) \[mol/L·s\]
    ///
    /// # Physics
    ///
    /// Approximated as a narrow Gaussian with σ = dt/2
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::models::TemporalInjection;
    /// // Inject at t=5s
    /// let injection = TemporalInjection::dirac(5.0, 0.1);
    /// ```
    Dirac { time: f64, amount: f64 },

    /// Gaussian (bell-shaped) injection over time
    ///
    /// # Parameters
    ///
    /// - `center` : Peak injection time \[s\]
    /// - `width` : Standard deviation σ \[s\]
    /// - `peak_concentration` : Maximum concentration \[mol/L\]
    ///
    /// # Formula
    ///
    /// ```text
    /// C(t) = C₀ · exp(-((t - μ)² / (2σ²)))
    /// ```
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::models::TemporalInjection;
    /// // Gaussian injection: peak at t=10s, width 3s, max 0.1 mol/L
    /// let injection = TemporalInjection::gaussian(10.0, 3.0, 0.1);
    /// ```
    Gaussian {
        center: f64,
        width: f64,
        peak_concentration: f64,
    },

    /// Rectangular (constant) injection for a duration
    ///
    /// # Parameters
    ///
    /// - `start` : Start time \[s\]
    /// - `end` : End time \[s\]
    /// - `concentration` : Constant concentration during injection \[mol/L\]
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::models::TemporalInjection;
    /// // Constant injection from t=5s to t=15s
    /// let injection = TemporalInjection::rectangle(5.0, 15.0, 0.05);
    /// ```
    Rectangle {
        start: f64,
        end: f64,
        concentration: f64,
    },

    /// Custom temporal profile from user function
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::models::TemporalInjection;
    /// let injection = TemporalInjection::custom(|t| {
    ///     if t < 10.0 { 0.1 * t / 10.0 } else { 0.0 }
    /// });
    /// ```
    Custom(Arc<dyn Fn(f64) -> f64 + Send + Sync>),

    /// No injection (baseline = 0)
    None,
}

// ==================== Manual Clone Implementation ====================

impl Clone for TemporalInjection {
    fn clone(&self) -> Self {
        match self {
            Self::Dirac { time, amount } => Self::Dirac {
                time: *time,
                amount: *amount,
            },
            Self::Gaussian {
                center,
                width,
                peak_concentration,
            } => Self::Gaussian {
                center: *center,
                width: *width,
                peak_concentration: *peak_concentration,
            },
            Self::Rectangle {
                start,
                end,
                concentration,
            } => Self::Rectangle {
                start: *start,
                end: *end,
                concentration: *concentration,
            },
            Self::Custom(f) => Self::Custom(Arc::clone(f)), // ✅ Arc allows cloning
            Self::None => Self::None,
        }
    }
}

// ==================== Manual Debug Implementation ====================

impl std::fmt::Debug for TemporalInjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dirac { time, amount } => f
                .debug_struct("Dirac")
                .field("time", time)
                .field("amount", amount)
                .finish(),
            Self::Gaussian {
                center,
                width,
                peak_concentration,
            } => f
                .debug_struct("Gaussian")
                .field("center", center)
                .field("width", width)
                .field("peak_concentration", peak_concentration)
                .finish(),
            Self::Rectangle {
                start,
                end,
                concentration,
            } => f
                .debug_struct("Rectangle")
                .field("start", start)
                .field("end", end)
                .field("concentration", concentration)
                .finish(),
            Self::Custom(_) => f
                .debug_struct("Custom")
                .field("function", &"<user-defined>")
                .finish(),
            Self::None => f.debug_struct("None").finish(),
        }
    }
}

// ==================== Manual Serialize Implementation ====================

/// Serialises [`TemporalInjection`] to JSON/YAML via an internal snapshot type.
///
/// # Errors
///
/// Returns a serialisation error if called on [`TemporalInjection::Custom`],
/// which holds a closure that cannot be represented in a data format.
impl Serialize for TemporalInjection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let snapshot = match self {
            Self::Dirac { time, amount } => TemporalInjectionSnapshot::Dirac {
                time: *time,
                amount: *amount,
            },
            Self::Gaussian {
                center,
                width,
                peak_concentration,
            } => TemporalInjectionSnapshot::Gaussian {
                center: *center,
                width: *width,
                peak_concentration: *peak_concentration,
            },
            Self::Rectangle {
                start,
                end,
                concentration,
            } => TemporalInjectionSnapshot::Rectangle {
                start: *start,
                end: *end,
                concentration: *concentration,
            },
            Self::None => TemporalInjectionSnapshot::None,
            Self::Custom(_) => {
                return Err(serde::ser::Error::custom(
                    "Temporal Injection::Custom cannot be serialized",
                ));
            }
        };
        snapshot.serialize(serializer)
    }
}

// ==================== Manual Deserialize Implementation ====================

/// Deserializes [`TemporalInjection`] from JSON/YAML via an internal snapshot type.
///
/// [`TemporalInjection::Custom`] cannot be deserialized — it must be constructed
/// programmatically via [`TemporalInjection::custom`].
impl<'de> Deserialize<'de> for TemporalInjection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let snapshot = match TemporalInjectionSnapshot::deserialize(deserializer)? {
            TemporalInjectionSnapshot::Dirac { time, amount } => Self::Dirac { time, amount },
            TemporalInjectionSnapshot::Gaussian {
                center,
                width,
                peak_concentration,
            } => Self::Gaussian {
                center,
                width,
                peak_concentration,
            },
            TemporalInjectionSnapshot::Rectangle {
                start,
                end,
                concentration,
            } => Self::Rectangle {
                start,
                end,
                concentration,
            },
            TemporalInjectionSnapshot::None => Self::None,
        };
        Ok(snapshot)
    }
}

// ==================== Implementation ====================
impl TemporalInjection {
    /// Create a Dirac delta injection
    ///
    /// # Arguments
    ///
    /// * `time` - Time of injection \[s\]
    /// * `amount` - Amount to inject \[mol/L·s\]
    ///
    /// # Note
    ///
    /// In discrete time, this is approximated as a narrow Gaussian
    /// with width = dt/2 to conserve total injected amount.
    pub fn dirac(time: f64, amount: f64) -> Self {
        Self::Dirac { time, amount }
    }

    /// Create a Gaussian temporal injection
    ///
    /// # Arguments
    ///
    /// * `center` - Peak time \[s\]
    /// * `width` - Standard deviation \[s\]
    /// * `peak_concentration` - Maximum concentration \[mol/L\]
    pub fn gaussian(center: f64, width: f64, peak_concentration: f64) -> Self {
        Self::Gaussian {
            center,
            width,
            peak_concentration,
        }
    }

    /// Create a rectangular temporal injection
    ///
    /// # Arguments
    ///
    /// * `start` - Start time \[s\]
    /// * `end` - End time \[s\]
    /// * `concentration` - Constant concentration \[mol/L\]
    pub fn rectangle(start: f64, end: f64, concentration: f64) -> Self {
        assert!(end > start, "Rectangle end must be > start");
        Self::Rectangle {
            start,
            end,
            concentration,
        }
    }

    /// Create a custom temporal injection
    ///
    /// # Arguments
    ///
    /// * `f` - Function that takes time and returns concentration
    pub fn custom<F>(f: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        Self::Custom(Arc::new(f))
    }

    /// Create a "no injection" profile (always returns 0)
    pub fn none() -> Self {
        Self::None
    }

    /// Evaluate the injection profile at a given time
    ///
    /// # Arguments
    ///
    /// * `t` - Current time \[s\]
    ///
    /// # Returns
    ///
    /// Concentration at inlet at time t \[mol/L\]
    pub fn evaluate(&self, t: f64) -> f64 {
        match self {
            Self::Dirac { time, amount } => {
                // Approximate as narrow Gaussian
                // Width chosen to give reasonable discrete representation
                let width = 0.1; // 0.1 second width for Dirac approximation
                let distance = (t - time) / width;
                let peak = amount / (width * (2.0 * std::f64::consts::PI).sqrt());
                let exposant = -(distance * distance) / (2.0 * width * width);

                peak * exposant.exp()
            }

            Self::Gaussian {
                center,
                width,
                peak_concentration,
            } => {
                let distance = (t - center) / width;
                peak_concentration * (-distance * distance / 2.0).exp()
            }

            Self::Rectangle {
                start,
                end,
                concentration,
            } => {
                if t >= *start && t < *end {
                    *concentration
                } else {
                    0.0
                }
            }

            Self::Custom(f) => f(t),

            Self::None => 0.0,
        }
    }

    /// Evaluate at multiple time points
    ///
    /// # Arguments
    ///
    /// * `times` - Vector of time points
    ///
    /// # Returns
    ///
    /// Vector of concentrations at each time
    pub fn evaluate_series(&self, times: &[f64]) -> Vec<f64> {
        times.iter().map(|&t| self.evaluate(t)).collect()
    }
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirac_injection() {
        let injection = TemporalInjection::dirac(10.0, 1.0);

        // Before injection
        assert!(injection.evaluate(0.0) < 0.01);

        // At injection time (should have high concentration)
        assert!(injection.evaluate(10.0) > 1.0);

        // After injection
        assert!(injection.evaluate(20.0) < 0.01);
    }

    #[test]
    fn test_gaussian_injection() {
        let injection = TemporalInjection::gaussian(10.0, 2.0, 0.1);

        // At center
        assert!((injection.evaluate(10.0) - 0.1).abs() < 1e-10);

        // At ±σ should be exp(-0.5) ≈ 0.606 of peak
        let expected = 0.1 * 0.606;
        assert!((injection.evaluate(8.0) - expected).abs() < 0.01);
        assert!((injection.evaluate(12.0) - expected).abs() < 0.01);

        // Far from center
        assert!(injection.evaluate(0.0) < 0.001);
        assert!(injection.evaluate(20.0) < 0.001);
    }

    #[test]
    fn test_rectangle_injection() {
        let injection = TemporalInjection::rectangle(5.0, 15.0, 0.05);

        // Before
        assert_eq!(injection.evaluate(4.0), 0.0);

        // During
        assert_eq!(injection.evaluate(5.0), 0.05);
        assert_eq!(injection.evaluate(10.0), 0.05);
        assert_eq!(injection.evaluate(14.9), 0.05);

        // After
        assert_eq!(injection.evaluate(15.0), 0.0);
    }

    #[test]
    fn test_custom_injection() {
        let injection = TemporalInjection::custom(|t| if t < 10.0 { 0.01 * t } else { 0.0 });

        assert_eq!(injection.evaluate(0.0), 0.0);
        assert_eq!(injection.evaluate(5.0), 0.05);
        assert_eq!(injection.evaluate(10.0), 0.0);
    }

    #[test]
    fn test_none_injection() {
        let injection = TemporalInjection::none();

        assert_eq!(injection.evaluate(0.0), 0.0);
        assert_eq!(injection.evaluate(100.0), 0.0);
    }

    #[test]
    fn test_evaluate_series() {
        let injection = TemporalInjection::gaussian(10.0, 2.0, 0.1);
        let times = vec![0.0, 5.0, 10.0, 15.0, 20.0];
        let values = injection.evaluate_series(&times);

        assert_eq!(values.len(), 5);
        assert!((values[2] - 0.1).abs() < 1e-10); // Peak at t=10
    }

    #[test]
    #[should_panic(expected = "Rectangle end must be > start")]
    fn test_rectangle_invalid() {
        TemporalInjection::rectangle(10.0, 10.0, 0.05);
    }

    #[test]
    fn test_debug_temporal_injection_dirac() {
        let injection = TemporalInjection::dirac(0.0, 10.0);
        let debug = format!("{:?}", injection);

        assert_eq!(debug, "Dirac { time: 0.0, amount: 10.0 }");
    }

    #[test]
    fn test_debug_temporal_injection_gaussian() {
        let injection = TemporalInjection::gaussian(10.0, 2.0, 0.1);
        let debug = format!("{:?}", injection);

        assert_eq!(
            debug,
            "Gaussian { center: 10.0, width: 2.0, peak_concentration: 0.1 }"
        );
    }

    #[test]
    fn test_debug_temporal_injection_rectangle() {
        let injection = TemporalInjection::rectangle(5.0, 9.0, 0.25);
        let debug = format!("{:?}", injection);

        assert_eq!(
            debug,
            "Rectangle { start: 5.0, end: 9.0, concentration: 0.25 }"
        );
    }

    #[test]
    fn test_debug_temporal_injection_custom() {
        let injection = TemporalInjection::custom(|t| t.exp());
        let debug = format!("{:?}", injection);

        assert_eq!(debug, "Custom { function: \"<user-defined>\" }");
    }

    #[test]
    fn test_debug_temporal_injection_none() {
        let injection = TemporalInjection::none();
        let debug = format!("{:?}", injection);

        assert_eq!(debug, "None");
    }

    #[test]
    fn test_clone_temporal_injection_dirac() {
        let injection = TemporalInjection::dirac(10.0, 1.0);
        let clone = injection.clone();

        // Before injection
        assert!(injection.evaluate(0.0) < 0.01);
        assert!(clone.evaluate(0.0) < 0.01);

        // At injection time (should have high concentration)
        assert!(injection.evaluate(10.0) > 1.0);
        assert!(clone.evaluate(10.0) > 1.0);

        // After injection
        assert!(injection.evaluate(20.0) < 0.01);
        assert!(clone.evaluate(20.0) < 0.01);
    }

    #[test]
    fn test_clone_temporal_injection_gaussian() {
        let injection = TemporalInjection::gaussian(10.0, 2.0, 0.1);
        let clone = injection.clone();

        // At center
        assert!((injection.evaluate(10.0) - 0.1).abs() < 1e-10);
        assert!((clone.evaluate(10.0) - 0.1).abs() < 1e-10);

        // At ±σ should be exp(-0.5) ≈ 0.606 of peak
        let expected = 0.1 * 0.606;
        assert!((injection.evaluate(8.0) - expected).abs() < 0.01);
        assert!((injection.evaluate(12.0) - expected).abs() < 0.01);

        assert!((clone.evaluate(8.0) - expected).abs() < 0.01);
        assert!((clone.evaluate(12.0) - expected).abs() < 0.01);

        // Far from center
        assert!(injection.evaluate(0.0) < 0.001);
        assert!(injection.evaluate(20.0) < 0.001);

        assert!(clone.evaluate(0.0) < 0.001);
        assert!(clone.evaluate(20.0) < 0.001);
    }

    #[test]
    fn test_clone_temporal_injection_rectangle() {
        let injection = TemporalInjection::rectangle(5.0, 15.0, 0.05);
        let clone = injection.clone();

        // Before
        assert_eq!(injection.evaluate(4.0), 0.0);
        assert_eq!(clone.evaluate(4.0), 0.0);

        // During
        assert_eq!(injection.evaluate(5.0), 0.05);
        assert_eq!(injection.evaluate(10.0), 0.05);
        assert_eq!(injection.evaluate(14.9), 0.05);

        assert_eq!(clone.evaluate(5.0), 0.05);
        assert_eq!(clone.evaluate(10.0), 0.05);
        assert_eq!(clone.evaluate(14.9), 0.05);

        // After
        assert_eq!(injection.evaluate(15.0), 0.0);
        assert_eq!(clone.evaluate(15.0), 0.0);
    }

    #[test]
    fn test_clone_temporal_injection_custom() {
        let injection = TemporalInjection::custom(|t| if t < 10.0 { 0.01 * t } else { 0.0 });
        let clone = injection.clone();

        assert_eq!(injection.evaluate(0.0), 0.0);
        assert_eq!(clone.evaluate(0.0), 0.0);
        assert_eq!(injection.evaluate(5.0), 0.05);
        assert_eq!(clone.evaluate(5.0), 0.05);
        assert_eq!(injection.evaluate(10.0), 0.0);
        assert_eq!(clone.evaluate(10.0), 0.0);
    }

    #[test]
    fn test_clone_temporal_injection_none() {
        let injection = TemporalInjection::none();
        let clone = injection.clone();

        assert_eq!(clone.evaluate(0.0), 0.0);
        assert_eq!(clone.evaluate(100.0), 0.0);
    }

    // ==================== Serde round-trip tests ====================

    #[test]
    fn test_serialize_dirac() {
        let injection = TemporalInjection::dirac(5.0, 0.1);
        let json = serde_json::to_string(&injection).unwrap();
        assert!(json.contains("\"type\":\"Dirac\""));
        assert!(json.contains("\"time\":5.0"));
        assert!(json.contains("\"amount\":0.1"));
    }

    #[test]
    fn test_serialize_gaussian() {
        let injection = TemporalInjection::gaussian(10.0, 3.0, 0.1);
        let json = serde_json::to_string(&injection).unwrap();
        assert!(json.contains("\"type\":\"Gaussian\""));
        assert!(json.contains("\"center\":10.0"));
    }

    #[test]
    fn test_serialize_rectangle() {
        let injection = TemporalInjection::rectangle(5.0, 15.0, 0.05);
        let json = serde_json::to_string(&injection).unwrap();
        assert!(json.contains("\"type\":\"Rectangle\""));
        assert!(json.contains("\"start\":5.0"));
        assert!(json.contains("\"end\":15.0"));
    }

    #[test]
    fn test_serialize_none() {
        let injection = TemporalInjection::none();
        let json = serde_json::to_string(&injection).unwrap();
        assert!(json.contains("\"type\":\"None\""));
    }

    #[test]
    fn test_serialize_custom_fails() {
        let injection = TemporalInjection::custom(|t| t);
        assert!(serde_json::to_string(&injection).is_err());
    }

    #[test]
    fn test_round_trip_dirac() {
        let original = TemporalInjection::dirac(5.0, 0.1);
        let json = serde_json::to_string(&original).unwrap();
        let restored: TemporalInjection = serde_json::from_str(&json).unwrap();
        assert!((restored.evaluate(5.0) - original.evaluate(5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_round_trip_gaussian() {
        let original = TemporalInjection::gaussian(10.0, 3.0, 0.1);
        let json = serde_json::to_string(&original).unwrap();
        let restored: TemporalInjection = serde_json::from_str(&json).unwrap();
        assert!((restored.evaluate(10.0) - original.evaluate(10.0)).abs() < 1e-10);
    }

    #[test]
    fn test_round_trip_rectangle() {
        let original = TemporalInjection::rectangle(5.0, 15.0, 0.05);
        let json = serde_json::to_string(&original).unwrap();
        let restored: TemporalInjection = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.evaluate(10.0), original.evaluate(10.0));
    }

    #[test]
    fn test_round_trip_none() {
        let original = TemporalInjection::none();
        let json = serde_json::to_string(&original).unwrap();
        let restored: TemporalInjection = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.evaluate(0.0), 0.0);
    }

    #[test]
    fn test_round_trip_yaml() {
        let original = TemporalInjection::dirac(5.0, 0.1);
        let yaml = serde_yaml::to_string(&original).unwrap();
        let restored: TemporalInjection = serde_yaml::from_str(&yaml).unwrap();
        assert!((restored.evaluate(5.0) - original.evaluate(5.0)).abs() < 1e-10);
    }
}
