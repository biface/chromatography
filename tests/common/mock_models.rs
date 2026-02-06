//! Mock physical models for testing
//!
//! These models have known analytical solutions, making them
//! ideal for validating numerical solver accuracy.

use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};

// =================================================================================================
// Exponential Decay: dy/dt = -k*y
// =================================================================================================

/// Exponential decay model: dy/dt = -k*y
///
/// Analytical solution: y(t) = y₀ * exp(-k*t)
///
/// Useful for testing solver accuracy since we know the exact solution.
pub struct ExponentialDecay {
    pub points: usize,
    pub decay_rate: f64,  // k in dy/dt = -k*y
}

impl ExponentialDecay {
    pub fn new(points: usize, decay_rate: f64) -> Self {
        Self { points, decay_rate }
    }

    /// Compute analytical solution at time t
    pub fn analytical_solution(&self, t: f64, y0: f64) -> f64 {
        y0 * (-self.decay_rate * t).exp()
    }
}

impl PhysicalModel for ExponentialDecay {
    fn points(&self) -> usize {
        self.points
    }

    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
        // dy/dt = -k * y
        let mut result = state.clone();

        if let Some(conc) = result.get_mut(PhysicalQuantity::Concentration) {
            conc.apply(|y| -self.decay_rate * y);
        }

        result
    }

    fn setup_initial_state(&self) -> PhysicalState {
        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(self.points, 1.0),
        )
    }

    fn name(&self) -> &str {
        "Exponential Decay"
    }
}

// =================================================================================================
// Constant Growth: dy/dt = c
// =================================================================================================

/// Constant growth model: dy/dt = c
///
/// Analytical solution: y(t) = y₀ + c*t
///
/// Euler is exact for this problem, RK4 should also be exact.
pub struct ConstantGrowth {
    pub points: usize,
    pub growth_rate: f64,
}

impl ConstantGrowth {
    pub fn new(points: usize, growth_rate: f64) -> Self {
        Self { points, growth_rate }
    }

    /// Compute analytical solution at time t
    pub fn analytical_solution(&self, t: f64, y0: f64) -> f64 {
        y0 + self.growth_rate * t
    }
}

impl PhysicalModel for ConstantGrowth {
    fn points(&self) -> usize {
        self.points
    }

    fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {
        // dy/dt = c (constant)
        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(self.points, self.growth_rate),
        )
    }

    fn setup_initial_state(&self) -> PhysicalState {
        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(self.points, 0.0),
        )
    }

    fn name(&self) -> &str {
        "Constant Growth"
    }
}

// =================================================================================================
// Linear Transport: dy/dt = -v * dy/dx (simplified)
// =================================================================================================

/// Linear transport model (simplified)
///
/// This is a placeholder for more complex transport models.
pub struct LinearTransport {
    pub points: usize,
    pub velocity: f64,
}

impl LinearTransport {
    pub fn new(points: usize, velocity: f64) -> Self {
        Self { points, velocity }
    }
}

impl PhysicalModel for LinearTransport {
    fn points(&self) -> usize {
        self.points
    }

    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
        // Placeholder: just decay for now
        let mut result = state.clone();

        if let Some(conc) = result.get_mut(PhysicalQuantity::Concentration) {
            conc.apply(|y| -self.velocity * y * 0.1);
        }

        result
    }

    fn setup_initial_state(&self) -> PhysicalState {
        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(self.points, 1.0),
        )
    }

    fn name(&self) -> &str {
        "Linear Transport"
    }
}

// =================================================================================================
// Tests for Mock Models
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay_analytical() {
        let model = ExponentialDecay::new(5, 0.5);

        // y(0) = 1.0
        assert!((model.analytical_solution(0.0, 1.0) - 1.0).abs() < 1e-10);

        // y(1) = exp(-0.5) ≈ 0.6065
        let y1 = model.analytical_solution(1.0, 1.0);
        assert!((y1 - 0.6065306597).abs() < 1e-6);
    }

    #[test]
    fn test_constant_growth_analytical() {
        let model = ConstantGrowth::new(5, 2.0);

        // y(0) = 0.0
        assert!((model.analytical_solution(0.0, 0.0) - 0.0).abs() < 1e-10);

        // y(5) = 0 + 2*5 = 10.0
        assert!((model.analytical_solution(5.0, 0.0) - 10.0).abs() < 1e-10);
    }
}