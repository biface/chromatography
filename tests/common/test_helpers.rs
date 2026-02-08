//! Helper functions for integration tests

use chrom_rs::physics::{PhysicalState, PhysicalQuantity};
use chrom_rs::solver::{Scenario, DomainBoundaries};
use chrom_rs::physics::PhysicalModel;

/// Assert that two physical states are close (within tolerance)
pub fn assert_states_close(
    state1: &PhysicalState,
    state2: &PhysicalState,
    tolerance: f64,
    message: &str,
) {
    for quantity in [
        PhysicalQuantity::Concentration,
        PhysicalQuantity::Temperature,
        PhysicalQuantity::Pressure,
    ] {
        if let (Some(data1), Some(data2)) = (state1.get(quantity), state2.get(quantity)) {
            let vec1 = data1.as_vector();
            let vec2 = data2.as_vector();

            assert_eq!(vec1.len(), vec2.len(), "{}: Dimension mismatch", message);

            for (i, (&v1, &v2)) in vec1.iter().zip(vec2.iter()).enumerate() {
                let diff = (v1 - v2).abs();
                assert!(
                    diff < tolerance,
                    "{}: Element {} differs by {} (tolerance {})",
                    message, i, diff, tolerance
                );
            }
        }
    }
}

/// Compute L2 norm error between two states
pub fn compute_l2_error(state1: &PhysicalState, state2: &PhysicalState) -> f64 {
    let mut sum_squared_diff = 0.0;
    let mut count = 0;

    for quantity in [
        PhysicalQuantity::Concentration,
        PhysicalQuantity::Temperature,
    ] {
        if let (Some(data1), Some(data2)) = (state1.get(quantity), state2.get(quantity)) {
            let vec1 = data1.as_vector();
            let vec2 = data2.as_vector();

            for (&v1, &v2) in vec1.iter().zip(vec2.iter()) {
                sum_squared_diff += (v1 - v2).powi(2);
                count += 1;
            }
        }
    }

    if count > 0 {
        (sum_squared_diff / count as f64).sqrt()
    } else {
        0.0
    }
}

/// Create a simple scenario for testing
pub fn create_simple_scenario(
    model: Box<dyn PhysicalModel>,
) -> Scenario {
    let initial = model.setup_initial_state();
    let boundaries = DomainBoundaries::temporal(initial);
    Scenario::new(model, boundaries)
}

/// Compute relative error: |actual - expected| / |expected|
pub fn relative_error(actual: f64, expected: f64) -> f64 {
    if expected.abs() < 1e-10 {
        (actual - expected).abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrom_rs::physics::PhysicalData;

    #[test]
    fn test_relative_error() {
        assert!((relative_error(1.0, 1.0) - 0.0).abs() < 1e-10);
        assert!((relative_error(1.1, 1.0) - 0.1).abs() < 1e-10);
        assert!((relative_error(0.9, 1.0) - 0.1).abs() < 1e-10);
    }
}