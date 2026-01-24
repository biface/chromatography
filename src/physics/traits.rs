//! Physical models traits and types
//!
//! This module defines the core API for physical models:
//! - `PhysicalModel`: trait for all physical models
//! - `PhysicalState`: flexible state container
//! - `PhysicalQuantity`: type-safe quantity identifiers

use nalgebra::DVector;
use std::collections::HashMap;

// =================================================================================================
// Physical quantities (Type-safe Identifiers
// =================================================================================================

/// Known physical quantities (type-safe enum)
///
/// # Enum type safety
///
/// If you need to use sizes other than those available in this enumeration, you must create it
/// in order to maintain type safety.
///
/// # Example
/// ```
/// let viscosity = PhysicalQuantity::Custom("Viscosity");
/// let mut state = PhysicalState::empty();
///
/// state.set(
///     viscosity,
///     DVector::from_element(100, 0.0)
/// );
///
///```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicalQuantity {
    /// Concentration (mol/L)
    Concentration,

    /// Temperature (K)
    Temperature,

    /// Pressure (Pa)
    Pressure,

    /// Custom quantity (for use extension)
    Custom(&'static str),

}

// =================================================================================================
// Physical State (Flexible State Container)
// =================================================================================================

/// Physical state of the system
///
/// This structure contains all physical quantities at a given time or iteration.
/// It is flexible since it can store concentration, temperature, pressure, etc.
///
/// # Type Safety
///
/// This structure uses enum `PhysicalQuantity` for quantities instead of strings.
///
/// # Example
/// ```
/// let mut state = PhysicalState::new(PhysicalQuantity::Concentration, concentration_vector);
/// state.set(PhysicalQuantity::Temperature, temperature_vector);
/// ```
#[derive(Debug, Clone)]
pub struct PhysicalState {
    /// Physical quantities stored in a dictionary
    quantities: HashMap<PhysicalQuantity, DVector<f64>>,

    /// Scalar metadata (optional, e.g. energy, mass, etc.)
    metadata: HashMap<String, f64>,
}

impl PhysicalState {
    /// Create a new state with primary quantity
    pub fn new(quantity: PhysicalQuantity, value: DVector<f64>) -> Self {
        let mut quantities = HashMap::new();
        quantities.insert(quantity, value);

        Self {
            quantities,
            metadata: HashMap::new(),
        }
    }

    /// Create an empty state
    pub fn empty() -> Self {
        Self {
            quantities: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get a quantity by type
    pub fn get(&self, quantity: PhysicalQuantity) -> Option<&DVector<f64>> {
        self.quantities.get(&quantity)
    }

    /// Get mutable reference to a quantity
    pub fn get_mut(&mut self, quantity: PhysicalQuantity) -> Option<&mut DVector<f64>> {
        self.quantities.get_mut(&quantity)
    }

    /// Set a quantity
    pub fn set (&mut self, quantity: PhysicalQuantity, value: DVector<f64>) {
        self.quantities.insert(quantity, value);
    }

    /// List of available physical state quantities
    pub fn available_quantities(&self) -> Vec<PhysicalQuantity> {
        self.quantities.keys().cloned().collect()
    }

    /// Get a metadata
    pub fn get_metadata(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).copied()
    }

    /// Set a metadata
    pub fn set_metadata(&mut self, key: String, value: f64) {
        self.metadata.insert(key, value);
    }

}

// Operator overloading for numerical operations

impl std::ops::Add for PhysicalState {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (quantity, value) in rhs.quantities {
            if let Some(existing_value) = self.get_mut(quantity) {
                *existing_value += value;
            } else {
                self.quantities.insert(quantity, value);
            }
        }
        self
    }
}

impl std::ops::Mul<f64> for PhysicalState {

    type Output = Self;
    fn mul(mut self, scalar: f64) -> Self::Output {
        for data in self.quantities.values_mut() {
            *data *= scalar;
        }
        self
    }
}

// ==================================================================================================
// Physical Model Trait
// =================================================================================================

/// Trait for physical models
///
/// # Responsibility
/// Computes the physics equations of a system at a given state.
/// Does NOT solve them (that's the Solver's job).
///
/// The model provides the "physics" (equations), the Solver provides
/// the "numerics" (method to solve them).
///
/// # Stability
/// This trait is STABLE since v0.1.0 and will NEVER be modified.
/// Extensions will use separate optional traits.
///
/// # Mandatory Point
/// All new physical models MUST implement this trait.

pub trait PhysicalModel : Send + Sync {

    /// Number of spatial points
    ///
    /// Used by the solver to allocate vectors
    fn points(&self) -> usize;

    /// Computes the physics at a given state
    ///
    /// # Arguments
    /// * `state` - Current physical state of the system
    ///
    /// # Returns
    /// Result of evaluating the physics (interpretation depends on model type)
    ///
    /// # Physical Interpretation
    ///
    /// For time-dependent models (ODE):
    ///   - Returns right-hand side f(y) of dy/dt = f(y)
    ///   - Solver will integrate this over time using Euler, RK4, etc.
    ///   - Example: chromatography transport-dispersion-adsorption
    ///
    /// For steady-state models (algebraic):
    ///   - Returns residual F(x) to minimize
    ///   - Solver will find x such that F(x) = 0 using Newton-Raphson, etc.
    ///   - Example: equilibrium adsorption
    ///
    /// For PDE models:
    ///   - Returns spatial discretization of differential operators
    ///   - Solver will integrate or solve the resulting system
    ///
    /// # Note
    /// This method encapsulates ALL the physics:
    /// - Isotherms, kinetics, reactions
    /// - Spatial derivatives (finite differences)
    /// - Boundary conditions
    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState;

    /// Creates the initial state for this physical model
    ///
    /// Defines what variables the model tracks (concentration, temperature, etc.)
    /// and their initial spatial distribution.
    fn setup_initial_state(&self) -> PhysicalState;

    /// Name of the model (used to display and logging)

    fn name(&self) -> &str;

    /// Description of the model (option)

    fn description(&self) -> Option<&String> {
        None
    }

}

// =================================================================================================
// Tests
// =================================================================================================
#[cfg(test)]
mod tests {
    use std::ops::Mul;
    use super::*;

    #[test]
    fn test_empty_physical_state() {
        let physics = PhysicalState::empty();

        assert_eq!(physics.quantities.len(), 0);
        assert_eq!(physics.metadata.len(), 0);
    }

    #[test]
    fn test_new_physical_state() {
        let quantity = PhysicalQuantity::Custom("Tesla");
        let physics = PhysicalState::new(
            quantity,
            DVector::from_row_slice(&[1.0, 2.0]),
        );

        assert_eq!(physics.quantities.len(), 1);
        assert_eq!(physics.metadata.len(), 0);
        assert!(physics.available_quantities().contains(&quantity));

        let values = physics.get(quantity).unwrap();

        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_modify_physical_state() {
        let mut physics = PhysicalState::new(
            PhysicalQuantity::Custom("Tesla"),
            DVector::from_row_slice(&[1.0, 2.0]),
        );

        let mut values = physics.get_mut(PhysicalQuantity::Custom("Tesla")).unwrap();
        assert_eq!(values.len(), 2);
        let mut values = values.push(10.0);
        physics.set(PhysicalQuantity::Custom("Tesla"), values);

        assert_eq!(physics.available_quantities().len(), 1);
        assert_eq!(physics.get(PhysicalQuantity::Custom("Tesla")).unwrap().len(), 3);

    }

    #[test]
    fn test_metadata() {
        let mut physics = PhysicalState::new(
            PhysicalQuantity::Custom("Tesla"),
            DVector::from_row_slice(&[1.0, 2.0]),
        );

        physics.set_metadata("molecule".to_string(), 10.0);
        assert_eq!(physics.get_metadata("molecule").unwrap(), 10.0);
    }

    #[test]
    fn test_addition() {
        let state_one = PhysicalState::new(
            PhysicalQuantity::Pressure,
            DVector::from_row_slice(&[780.0, 1024.0]),
        );
        let state_two = PhysicalState::new(
            PhysicalQuantity::Pressure,
            DVector::from_row_slice(&[230.0, -24.0]),
        );
        let false_one = PhysicalState::new(
            PhysicalQuantity::Temperature,
            DVector::from_row_slice(&[0.0, 273.15]),
        );

        let pressure = state_one.clone() + state_two;
        let temperature = false_one + state_one ;

        assert_eq!(pressure.get(PhysicalQuantity::Pressure).unwrap()[0], 1010.0);
        assert_eq!(pressure.get(PhysicalQuantity::Pressure).unwrap()[1], 1000.0);

        assert_ne!(temperature.get(PhysicalQuantity::Temperature).unwrap()[0], 780.0);
        assert_eq!(temperature.get(PhysicalQuantity::Temperature).unwrap()[0], 0.0);
    }

    #[test]
    fn test_multiplication() {
        let mut state_one = PhysicalState::new(
            PhysicalQuantity::Concentration,
            DVector::from_row_slice(&[1.0, 2.0]),
        );

        state_one = state_one * 10.0;

        assert_eq!(state_one.get(PhysicalQuantity::Concentration).unwrap()[0], 10.0);
        assert_eq!(state_one.get(PhysicalQuantity::Concentration).unwrap()[1], 20.0);

        let result = state_one.clone() * 2.0;

        assert_eq!(result.get(PhysicalQuantity::Concentration).unwrap()[0], 20.0);
        assert_eq!(result.get(PhysicalQuantity::Concentration).unwrap()[1], 40.0);
    }

}