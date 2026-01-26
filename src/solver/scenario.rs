//! Simulation scenario definition
//!
//! A scenario combines a physical model with boundary conditions.
use crate::physics::traits::PhysicalModel;
use crate::solver::boundary::DomainBoundaries;

/// Simulation scenario
///
/// Defines a specific case to simulate:
/// - Physical model (equations)
/// - Boundary conditions (domain boundaries)
///
/// # Design
///
/// The same scenario can be solved with different numerical methods.
/// This is the "WHAT to solve" (not "HOW to solve").
///
/// # Examples
///
/// ```rust
/// // Define scenario
/// let scenario = Scenario::new(model, boundaries);
///
/// // Solve with different methods
/// let result1 = euler_solver.solve(&scenario, &config1)?;
/// let result2 = newton_solver.solve(&scenario, &config2)?;
/// ```

pub struct Scenario {
    /// Physical model (equations)
    pub model: Box<dyn PhysicalModel>,

    /// Conditions and boundaries
    pub conditions: DomainBoundaries
}

impl Scenario {

    /// Create a scenario
    pub fn new(model: Box<dyn PhysicalModel>, conditions: DomainBoundaries) -> Self {
        Self { model, conditions }
    }

    /// Verifying scenario content (mainly boundaries)
    pub fn validate(&self) -> Result<(), String> {
        self.conditions.validate()
    }

    /// Get model name
    pub fn get_model_name(&self) -> &str {
        self.model.name()
    }

    /// n-dim resolution
    pub fn ndim(&self) -> usize {
        self.conditions.ndim()
    }

    /// spatial dimension
    pub fn sdim(&self) -> usize {
        self.conditions.sdim()
    }

    /// time dependant equations
    pub fn is_time_dependent(&self) -> bool {
        self.conditions.is_time_dependent()
    }
}

impl std::fmt::Debug for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scenario")
        .field("name", &self.get_model_name())
        .field("dimension", &self.ndim())
        .field("spatial dim", &self.sdim())
        .field("is time dependent", &self.is_time_dependent())
        .field("Boundaries / conditions", &self.conditions)
        .finish()
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use nalgebra::DVector;
    use super::*;
    use crate::physics::traits::{PhysicalState, PhysicalQuantity};
    use crate::solver::boundary::DomainBoundaries;

    // Mocking a Physical model
    struct MockModel {
        content: Vec<PhysicalState>
    }

    impl PhysicalModel for MockModel {
        fn points(&self) -> usize {
            10
        }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            if !self.content.is_empty() {
                self.content.last().unwrap().clone() + state.clone()
            } else { 
                state.clone()
            }
        }

        fn setup_initial_state(&self) -> PhysicalState {
            PhysicalState::empty()
        }

        fn name(&self) -> &str {
            "MockModel"
        }
    }

    #[test]
    fn test_scenario_creation() {
        let model = Box::new(MockModel { content: vec![] });
        assert_eq!(model.points(), 10);

        let boundaries = DomainBoundaries::default();
        let scenario = Scenario::new(model, boundaries);
        assert_eq!(scenario.get_model_name(), "MockModel");
    }


}