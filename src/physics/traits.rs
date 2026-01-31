//! Physical models traits and types
//!
//! This module defines the core API for physical models:
//! - `PhysicalModel`: trait for all physical models
//! - `PhysicalState`: flexible state container with dynamic species management
//! - `PhysicalQuantity`: type-safe quantity identifiers
//!
//! # Design Philosophy
//!
//! The separation of concerns:
//! - **PhysicalData**: Mathematical container (scalar/vector/matrix/array)
//! - **PhysicalState**: Physical semantics (concentration, temperature, etc.)
//! - **PhysicalModel**: Physics equations (isotherms, kinetics, etc.)
//! - **Solver**: Numerical methods (Euler, RK4, etc.)

use std::collections::HashMap;
use crate::physics::data::PhysicalData;

// =================================================================================================
// Physical Quantities (Type-safe Identifiers)
// =================================================================================================

/// Physical quantity identifiers (type-safe enum)
///
/// Provides type-safe keys for the `PhysicalState` HashMap.
///
/// # Extensibility
///
/// Use `Custom` variant for application-specific quantities:
///
/// ```rust
/// use chrom_rs::physics::PhysicalQuantity;
///
/// let viscosity = PhysicalQuantity::Custom("Viscosity");
/// let diffusion = PhysicalQuantity::Custom("DiffusionCoefficient");
/// ```
///
/// # Standard Quantities
///
/// Common quantities are provided as enum variants for convenience
/// and to avoid typos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicalQuantity {
    /// Concentration (mol/L or kg/m³)
    Concentration,

    /// Temperature (K)
    Temperature,

    /// Pressure (Pa)
    Pressure,

    /// Velocity (m/s)
    Velocity,

    /// Custom quantity (for extensibility)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalQuantity;
    ///
    /// let k_langmuir = PhysicalQuantity::Custom("K_langmuir");
    /// let retention_factor = PhysicalQuantity::Custom("RetentionFactor");
    /// ```
    Custom(&'static str),
}

impl std::fmt::Display for PhysicalQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhysicalQuantity::Concentration => write!(f, "Concentration"),
            PhysicalQuantity::Temperature => write!(f, "Temperature"),
            PhysicalQuantity::Pressure => write!(f, "Pressure"),
            PhysicalQuantity::Velocity => write!(f, "Velocity"),
            PhysicalQuantity::Custom(name) => write!(f, "{}", name),
        }
    }
}

// =================================================================================================
// Physical State (Flexible State Container)
// =================================================================================================

/// Physical state of the system
///
/// Container for all physical quantities with support for:
/// - Multiple quantities (concentration, temperature, pressure, etc.)
/// - Dynamic species management (add/remove species on the fly)
/// - Scalar metadata (total mass, energy, etc.)
///
/// # Storage
///
/// Uses `HashMap<PhysicalQuantity, PhysicalData>` where `PhysicalData` can be:
/// - **Scalar**: Uniform value (temperature = 298 K everywhere)
/// - **Vector**: 1D spatial profile (100 points, 1 species)
/// - **Matrix**: Multi-species or 2D spatial (100 points × 3 species)
/// - **Array**: 3D+ multidimensional (spatial + species + time)
///
/// # Species Management
///
/// For multi-component systems (e.g., chromatography), concentration is stored
/// as Matrix[n_points, n_species]:
///
/// ```text
/// Concentration Matrix [100 points × 3 species]:
/// Point 0:  [c₁, c₂, c₃]
/// Point 1:  [c₁, c₂, c₃]
/// ...
/// Point 99: [c₁, c₂, c₃]
/// ```
///
/// # Examples
///
/// ```rust
/// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
///
/// // Empty state
/// let mut state = PhysicalState::empty();
///
/// // Add uniform temperature
/// state.set(
///     PhysicalQuantity::Temperature,
///     PhysicalData::from_scalar(298.15)
/// );
///
/// // Add concentration profile (1 species)
/// state.set(
///     PhysicalQuantity::Concentration,
///     PhysicalData::uniform_vector(100, 1.0)
/// );
/// ```
#[derive(Debug, Clone)]
pub struct PhysicalState {
    /// Physical quantities stored as HashMap
    ///
    /// Key: Type of quantity (Concentration, Temperature, etc.)
    /// Value: Data container (Scalar, Vector, Matrix, or Array)
    pub quantities: HashMap<PhysicalQuantity, PhysicalData>,

    /// Scalar metadata (optional)
    ///
    /// Use for system-level properties like total mass, energy, etc.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalState;
    ///
    /// let mut state = PhysicalState::empty();
    /// state.set_metadata("total_mass".to_string(), 125.5);
    /// state.set_metadata("inlet_flow_rate".to_string(), 1.0);
    /// ```
    metadata: HashMap<String, f64>,
}

impl PhysicalState {
    // ======================================= Constructors =======================================

    /// Create a new state with a single primary quantity
    ///
    /// # Arguments
    /// * `quantity` - Type of physical quantity
    /// * `value` - Data container (Scalar, Vector, Matrix, or Array)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// // Mono-species: Vector [100]
    /// let state = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::uniform_vector(100, 1.0)
    /// );
    ///
    /// // Multi-species: Matrix [100, 3]
    /// let state = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::uniform_matrix(100, 3, 0.5)
    /// );
    /// ```
    pub fn new(quantity: PhysicalQuantity, value: PhysicalData) -> Self {
        let mut quantities = HashMap::new();
        quantities.insert(quantity, value);

        Self {
            quantities,
            metadata: HashMap::new(),
        }
    }

    /// Create an empty state
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let mut state = PhysicalState::empty();
    ///
    /// // Add quantities as needed
    /// state.set(PhysicalQuantity::Temperature, PhysicalData::from_scalar(298.15));
    /// state.set(PhysicalQuantity::Pressure, PhysicalData::from_scalar(101325.0));
    /// ```
    pub fn empty() -> Self {
        Self {
            quantities: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    // ======================================== Accessors =========================================

    /// Get a quantity by type (immutable reference)
    ///
    /// # Returns
    /// - `Some(&PhysicalData)` if quantity exists
    /// - `None` if quantity not found
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let state = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::uniform_vector(100, 1.0)
    /// );
    ///
    /// if let Some(conc) = state.get(PhysicalQuantity::Concentration) {
    ///     println!("Concentration shape: {:?}", conc.shape());
    /// }
    /// ```
    pub fn get(&self, quantity: PhysicalQuantity) -> Option<&PhysicalData> {
        self.quantities.get(&quantity)
    }

    /// Get mutable reference to a quantity
    ///
    /// Useful for in-place modifications.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let mut state = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::uniform_vector(100, 1.0)
    /// );
    ///
    /// // Modify in-place
    /// if let Some(conc) = state.get_mut(PhysicalQuantity::Concentration) {
    ///     conc.apply(|x| x * 2.0);  // Double all concentrations
    /// }
    /// ```
    pub fn get_mut(&mut self, quantity: PhysicalQuantity) -> Option<&mut PhysicalData> {
        self.quantities.get_mut(&quantity)
    }

    /// Set a quantity
    ///
    /// If quantity already exists, it will be overwritten.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let mut state = PhysicalState::empty();
    ///
    /// state.set(
    ///     PhysicalQuantity::Temperature,
    ///     PhysicalData::from_scalar(298.15)
    /// );
    /// ```
    pub fn set(&mut self, quantity: PhysicalQuantity, value: PhysicalData) {
        self.quantities.insert(quantity, value);
    }

    /// Remove a quantity from the state
    ///
    /// # Returns
    /// The removed value, or `None` if it didn't exist
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let mut state = PhysicalState::new(
    ///     PhysicalQuantity::Temperature,
    ///     PhysicalData::from_scalar(300.0)
    /// );
    ///
    /// let removed = state.remove(PhysicalQuantity::Temperature);
    /// assert!(removed.is_some());
    /// assert!(state.is_empty());
    /// ```
    pub fn remove(&mut self, quantity: PhysicalQuantity) -> Option<PhysicalData> {
        self.quantities.remove(&quantity)
    }

    /// List all available quantities
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let mut state = PhysicalState::empty();
    /// state.set(PhysicalQuantity::Temperature, PhysicalData::from_scalar(298.15));
    /// state.set(PhysicalQuantity::Pressure, PhysicalData::from_scalar(101325.0));
    ///
    /// let quantities = state.available_quantities();
    /// assert_eq!(quantities.len(), 2);
    /// ```
    pub fn available_quantities(&self) -> Vec<PhysicalQuantity> {
        self.quantities.keys().cloned().collect()
    }

    // ======================================= Metadata ===========================================

    /// Get metadata by key
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalState;
    ///
    /// let mut state = PhysicalState::empty();
    /// state.set_metadata("total_mass".to_string(), 125.5);
    ///
    /// assert_eq!(state.get_metadata("total_mass"), Some(125.5));
    /// assert_eq!(state.get_metadata("unknown"), None);
    /// ```
    pub fn get_metadata(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).copied()
    }

    /// Set metadata
    ///
    /// Use for scalar system-level properties.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalState;
    ///
    /// let mut state = PhysicalState::empty();
    /// state.set_metadata("inlet_flow_rate".to_string(), 1.0);  // mL/min
    /// state.set_metadata("column_temperature".to_string(), 298.15);  // K
    /// ```
    pub fn set_metadata(&mut self, key: String, value: f64) {
        self.metadata.insert(key, value);
    }


    /// Estimate total memory usage in bytes
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let mut state = PhysicalState::empty();
    /// state.set(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::uniform_matrix(100, 3, 1.0)
    /// );
    ///
    /// println!("Memory usage: {} bytes", state.memory_bytes());
    /// // Output: Memory usage: 2400 bytes (100 × 3 × 8)
    /// ```
    pub fn memory_bytes(&self) -> usize {
        self.quantities.values().map(|data| data.memory_bytes()).sum()
    }

    /// Number of quantities stored
    pub fn len(&self) -> usize {
        self.quantities.len()
    }

    /// Check if state is empty (no quantities)
    pub fn is_empty(&self) -> bool {
        self.quantities.is_empty()
    }
}

// =================================================================================================
// Arithmetic Operators
// =================================================================================================

impl std::ops::Add for PhysicalState {
    type Output = Self;

    /// Add two physical states
    ///
    /// Adds quantities element-wise. If a quantity exists in only one state,
    /// it's included in the result.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let state1 = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::uniform_vector(100, 1.0)
    /// );
    ///
    /// let state2 = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::uniform_vector(100, 0.5)
    /// );
    ///
    /// let result = state1 + state2;
    /// // Concentration is now 1.5 everywhere
    /// ```
    fn add(mut self, rhs: Self) -> Self::Output {
        for (quantity, rhs_value) in rhs.quantities {
            if let Some(lhs_value) = self.quantities.remove(&quantity) {
                // Both states have this quantity → add them
                let sum = lhs_value + rhs_value;
                self.quantities.insert(quantity, sum);
            } else {
                // Only rhs has this quantity → include it
                self.quantities.insert(quantity, rhs_value);
            }
        }
        self
    }
}

impl std::ops::Mul<f64> for PhysicalState {
    type Output = Self;

    /// Multiply all quantities by a scalar
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// let state = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     PhysicalData::uniform_vector(100, 1.0)
    /// );
    ///
    /// let scaled = state * 2.0;
    /// // Concentration is now 2.0 everywhere
    /// ```
    fn mul(mut self, scalar: f64) -> Self::Output {
        for data in self.quantities.values_mut() {
            *data *= scalar;
        }
        self
    }
}

// =================================================================================================
// Physical Model Trait
// =================================================================================================

/// Trait for physical models
///
/// # Responsibility
///
/// Computes the physics equations of a system at a given state.
/// Does NOT solve them (that's the Solver's job).
///
/// The model provides the "physics" (equations), the Solver provides
/// the "numerics" (method to solve them).
///
/// # API Stability
///
/// This trait is **STABLE** since v0.1.0 and will **NEVER** be modified
/// in backward-incompatible ways. Extensions will use separate optional traits.
///
/// # Mandatory Implementation
///
/// All physical models MUST implement this trait.
///
/// # Example Implementation
///
/// ```rust
/// use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
///
/// struct SimpleTransport {
///     points: usize,
///     velocity: f64,
/// }
///
/// impl PhysicalModel for SimpleTransport {
///     fn points(&self) -> usize {
///         self.points
///     }
///
///     fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
///         // Compute transport: dc/dt = -v * dc/dx
///         // (simplified - actual implementation would use finite differences)
///         let mut result = PhysicalState::empty();
///         // ... compute derivatives ...
///         result
///     }
///
///     fn setup_initial_state(&self) -> PhysicalState {
///         PhysicalState::new(
///             PhysicalQuantity::Concentration,
///             PhysicalData::uniform_vector(self.points, 0.0)
///         )
///     }
///
///     fn name(&self) -> &str {
///         "Simple Transport"
///     }
/// }
/// ```
pub trait PhysicalModel: Send + Sync {
    /// Number of spatial points
    ///
    /// Used by the solver to allocate vectors and check dimensions.
    ///
    /// # Example
    /// ```rust
    /// # use chrom_rs::physics::PhysicalModel;
    /// # struct MyModel { points: usize }
    /// # impl PhysicalModel for MyModel {
    /// #   fn points(&self) -> usize { self.points }
    /// #   fn compute_physics(&self, state: &chrom_rs::physics::PhysicalState) -> chrom_rs::physics::PhysicalState { chrom_rs::physics::PhysicalState::empty() }
    /// #   fn setup_initial_state(&self) -> chrom_rs::physics::PhysicalState { chrom_rs::physics::PhysicalState::empty() }
    /// #   fn name(&self) -> &str { "MyModel" }
    /// # }
    /// let model = MyModel { points: 100 };
    /// assert_eq!(model.points(), 100);
    /// ```
    fn points(&self) -> usize;

    /// Compute the physics at a given state
    ///
    /// # Arguments
    /// * `state` - Current physical state of the system
    ///
    /// # Returns
    /// Result of evaluating the physics (interpretation depends on model type)
    ///
    /// # Physical Interpretation
    ///
    /// **For time-dependent models (ODE)**:
    /// - Returns right-hand side f(y) of dy/dt = f(y)
    /// - Solver integrates this over time using Euler, RK4, etc.
    /// - Example: chromatography transport-dispersion-adsorption
    ///
    /// **For steady-state models (algebraic)**:
    /// - Returns residual F(x) to minimize
    /// - Solver finds x such that F(x) = 0 using Newton-Raphson, etc.
    /// - Example: equilibrium adsorption
    ///
    /// **For PDE models**:
    /// - Returns spatial discretization of differential operators
    /// - Solver integrates or solves the resulting system
    ///
    /// # Encapsulation
    ///
    /// This method encapsulates ALL the physics:
    /// - Isotherms, kinetics, reactions
    /// - Spatial derivatives (finite differences)
    /// - Boundary conditions
    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState;

    /// Create the initial state for this physical model
    ///
    /// Defines what variables the model tracks (concentration, temperature, etc.)
    /// and their initial spatial distribution.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
    ///
    /// struct MyModel;
    /// impl PhysicalModel for MyModel {
    ///     fn points(&self) -> usize { 100 }
    ///
    ///     fn setup_initial_state(&self) -> PhysicalState {
    ///         let mut state = PhysicalState::empty();
    ///
    ///         // Concentration: Gaussian pulse
    ///         let conc = PhysicalData::from_vec(
    ///             (0..100)
    ///                 .map(|i| {
    ///                     let x = (i as f64 - 50.0) / 10.0;
    ///                     (-x * x / 2.0).exp()
    ///                 })
    ///                 .collect()
    ///         );
    ///         state.set(PhysicalQuantity::Concentration, conc);
    ///
    ///         // Temperature: uniform
    ///         state.set(
    ///             PhysicalQuantity::Temperature,
    ///             PhysicalData::from_scalar(298.15)
    ///         );
    ///
    ///         state
    ///     }
    ///
    ///     fn compute_physics(&self, _state: &PhysicalState) -> PhysicalState {
    ///         PhysicalState::empty()
    ///     }
    ///
    ///     fn name(&self) -> &str { "MyModel" }
    /// }
    /// ```
    fn setup_initial_state(&self) -> PhysicalState;

    /// Name of the model (used for display and logging)
    ///
    /// # Example
    /// ```rust
    /// # use chrom_rs::physics::PhysicalModel;
    /// # struct Transport;
    /// # impl PhysicalModel for Transport {
    /// #   fn points(&self) -> usize { 100 }
    /// #   fn compute_physics(&self, _: &chrom_rs::physics::PhysicalState) -> chrom_rs::physics::PhysicalState { chrom_rs::physics::PhysicalState::empty() }
    /// #   fn setup_initial_state(&self) -> chrom_rs::physics::PhysicalState { chrom_rs::physics::PhysicalState::empty() }
    /// fn name(&self) -> &str {
    ///     "Transport-Dispersion-Adsorption Model"
    /// }
    /// # }
    /// ```
    fn name(&self) -> &str;

    /// Description of the model (optional)
    ///
    /// Provides additional information about the model for documentation
    /// or user interfaces.
    ///
    /// # Example
    /// ```rust
    /// # use chrom_rs::physics::PhysicalModel;
    /// # struct MyModel;
    /// # impl PhysicalModel for MyModel {
    /// #   fn points(&self) -> usize { 100 }
    /// #   fn compute_physics(&self, _: &chrom_rs::physics::PhysicalState) -> chrom_rs::physics::PhysicalState { chrom_rs::physics::PhysicalState::empty() }
    /// #   fn setup_initial_state(&self) -> chrom_rs::physics::PhysicalState { chrom_rs::physics::PhysicalState::empty() }
    /// #   fn name(&self) -> &str { "MyModel" }
    /// fn description(&self) -> Option<&str> {
    ///     Some("Transport with linear isotherm (Henry's law)")
    /// }
    /// # }
    /// ```
    fn description(&self) -> Option<&str> {
        None
    }
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {

    use super::*;
    use nalgebra::DMatrix;

    // ===================================== PhysicalQuantity =====================================

    #[test]
    fn test_physical_quantity_creation() {
        let c = PhysicalQuantity::Concentration;
        let t = PhysicalQuantity::Temperature;
        let p = PhysicalQuantity::Pressure;
        let v = PhysicalQuantity::Velocity;

        assert_eq!(format!("{}", c), "Concentration");
        assert_eq!(format!("{}", t), "Temperature");
        assert_eq!(format!("{}", p), "Pressure");
        assert_eq!(format!("{}", v), "Velocity");
    }

    #[test]
    fn test_custom_physical_quantity_create() {
        let viscosity = PhysicalQuantity::Custom("Viscosity");
        let k_langmuir = PhysicalQuantity::Custom("K langmuir");

        assert_eq!(format!("{}", viscosity), "Viscosity");
        assert_eq!(format!("{}", k_langmuir), "K langmuir");
    }

    #[test]
    fn test_physical_quantity_equality() {
        let c1 = PhysicalQuantity::Concentration;
        let c2 = PhysicalQuantity::Concentration;
        let p1 = PhysicalQuantity::Pressure;

        assert_eq!(c1, c2);
        assert_ne!(c1, p1);
    }

    #[test]
    fn test_custom_physical_quantity_equality2() {
        let v1 = PhysicalQuantity::Custom("Viscosity");
        let v2 = PhysicalQuantity::Custom("Viscosity");
        let k = PhysicalQuantity::Custom("K langmuir");

        assert_ne!(v1, k);
        assert_ne!(v2, k);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_physical_quantity_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(PhysicalQuantity::Custom("Viscosity"), 0.15);
        map.insert(PhysicalQuantity::Temperature, 350.0);

        assert_eq!(map.get(&PhysicalQuantity::Custom("Viscosity")), Some(&0.15));
        assert_eq!(map.get(&PhysicalQuantity::Temperature), Some(&350.0));
    }

    // ====================================== PhysicalState ======================================

    #[test]
    fn test_empty_physical_state() {
        let empty = PhysicalState::empty();

        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.available_quantities().len(), 0);
        assert_eq!(empty.memory_bytes(), 0);
    }

    #[test]
    fn test_physical_state_from_scalar() {
        let state = PhysicalState::new(
            PhysicalQuantity::Custom("Viscosity"),
            PhysicalData::Scalar(0.15)
        );

        assert!(!state.is_empty());
        assert_eq!(state.len(), 1);
        assert_eq!(state.available_quantities().len(), 1);
    }

    #[test]
    fn test_retrieve_physical_data () {
        let state = PhysicalState::new(
            PhysicalQuantity::Temperature,
            PhysicalData::Scalar(348.15)
        );
        assert!(state.available_quantities().contains(&PhysicalQuantity::Temperature));
        assert_eq!(state.get(PhysicalQuantity::Temperature).unwrap().as_scalar(), 348.15); // is 75 °C
    }

    #[test]
    fn test_physical_state_from_vector() {
        let state = PhysicalState::new(
            PhysicalQuantity::Velocity,
            PhysicalData::from_vec(vec![25.0, 10.0, 33.0])
        );

        assert_eq!(state.len(), 1);
        assert!(state.get(PhysicalQuantity::Velocity).unwrap().is_vector());
        assert_eq!(state.get(PhysicalQuantity::Velocity).unwrap().as_vector().len(), 3);
    }

    #[test]
    fn test_physical_state_from_matrix() {
        let state = PhysicalState::new(
            PhysicalQuantity::Custom("Viscosity"),
            PhysicalData::Matrix(
                DMatrix::from_row_slice(3, 3,
                                        &[
                                            0.12, 0.5, 0.01,
                                            0.2, 0.23, 0.6,
                                            0.0, 0.0, 1.0,]))
        );

        assert_eq!(state.len(), 1);
        assert!(state.get(PhysicalQuantity::Custom("Viscosity"))
            .unwrap()
            .is_matrix());
        assert_eq!(state.get(PhysicalQuantity::Custom("Viscosity"))
                       .unwrap()
                       .as_matrix()
                       .shape(), (3,3));
    }

    #[test]
    fn test_physical_set_n_get() {
        let mut state = PhysicalState::empty();

        state.set(PhysicalQuantity::Temperature, PhysicalData::Scalar(348.15));

        assert_eq!(state.len(), 1);

        assert!(state.get(PhysicalQuantity::Temperature).unwrap().is_scalar());
        assert_eq!(state.get(PhysicalQuantity::Temperature).unwrap().as_scalar(), 348.15);
    }

    #[test]
    fn test_get_mut_physical_state() {
        let mut state = PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(100, 0.1)
        );

        if let Some(concentration) = state.get_mut(PhysicalQuantity::Concentration) {
            concentration.apply(|c| c * 10.0)
        }

        assert_eq!(state.len(), 1);
        assert_eq!(state.get(PhysicalQuantity::Concentration).unwrap().as_vector()[10], 1.0);
    }

    #[test]
    fn test_remove_quantity() {
        let mut state = PhysicalState::new(
            PhysicalQuantity::Temperature,
            PhysicalData::from_scalar(300.0)
        );

        assert!(!state.is_empty());

        let removed = state.remove(PhysicalQuantity::Temperature);
        assert!(removed.is_some());
        assert!(state.is_empty());

        // Try to remove again
        let removed_again = state.remove(PhysicalQuantity::Temperature);
        assert!(removed_again.is_none());
    }

    #[test]
    fn test_available_quantities() {
        let mut state = PhysicalState::empty();

        state.set(PhysicalQuantity::Concentration, PhysicalData::from_scalar(1.0));
        state.set(PhysicalQuantity::Temperature, PhysicalData::from_scalar(298.15));
        state.set(PhysicalQuantity::Pressure, PhysicalData::from_scalar(101325.0));

        let quantities = state.available_quantities();
        assert_eq!(quantities.len(), 3);
        assert!(quantities.contains(&PhysicalQuantity::Concentration));
        assert!(quantities.contains(&PhysicalQuantity::Temperature));
        assert!(quantities.contains(&PhysicalQuantity::Pressure));
    }

    // ======================================== Metadata ========================================

    #[test]
    fn test_metadata_set_get() {
        let mut state = PhysicalState::empty();

        state.set_metadata("total_mass".to_string(), 125.5);
        state.set_metadata("flow_rate".to_string(), 1.0);

        assert_eq!(state.get_metadata("total_mass"), Some(125.5));
        assert_eq!(state.get_metadata("flow_rate"), Some(1.0));
        assert_eq!(state.get_metadata("unknown"), None);
    }

    #[test]
    fn test_metadata_overwrite() {
        let mut state = PhysicalState::empty();

        state.set_metadata("value".to_string(), 1.0);
        assert_eq!(state.get_metadata("value"), Some(1.0));

        state.set_metadata("value".to_string(), 2.0);
        assert_eq!(state.get_metadata("value"), Some(2.0));
    }

    // ===================================== Memory Usage =====================================

    #[test]
    fn test_memory_bytes() {
        let mut state = PhysicalState::empty();

        // Scalar: 8 bytes
        state.set(PhysicalQuantity::Temperature, PhysicalData::from_scalar(298.15));
        assert_eq!(state.memory_bytes(), 8);

        // Vector[100]: 800 bytes
        state.set(PhysicalQuantity::Concentration, PhysicalData::uniform_vector(100, 1.0));
        assert_eq!(state.memory_bytes(), 808);  // 8 + 800

        // Matrix[100, 3]: 2400 bytes
        state.set(PhysicalQuantity::Pressure, PhysicalData::uniform_matrix(100, 3, 1.0));
        assert_eq!(state.memory_bytes(), 3208);  // 8 + 800 + 2400
    }

    // ================================== Arithmetic Operators ==================================

    #[test]
    fn test_add_states_same_quantities() {
        let state1 = PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(100, 1.0)
        );

        let state2 = PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(100, 0.5)
        );

        let result = state1 + state2;

        let conc = result.get(PhysicalQuantity::Concentration).unwrap();
        assert_eq!(conc.as_vector()[0], 1.5);
        assert_eq!(conc.as_vector()[99], 1.5);
    }

    #[test]
    fn test_add_states_different_quantities() {
        let mut state1 = PhysicalState::empty();
        state1.set(PhysicalQuantity::Concentration, PhysicalData::from_scalar(1.0));

        let mut state2 = PhysicalState::empty();
        state2.set(PhysicalQuantity::Temperature, PhysicalData::from_scalar(298.15));

        let result = state1 + state2;

        assert_eq!(result.len(), 2);
        assert!(result.get(PhysicalQuantity::Concentration).is_some());
        assert!(result.get(PhysicalQuantity::Temperature).is_some());
    }

    #[test]
    fn test_add_states_overlapping() {
        let mut state1 = PhysicalState::empty();
        state1.set(PhysicalQuantity::Concentration, PhysicalData::from_scalar(1.0));
        state1.set(PhysicalQuantity::Temperature, PhysicalData::from_scalar(300.0));

        let mut state2 = PhysicalState::empty();
        state2.set(PhysicalQuantity::Concentration, PhysicalData::from_scalar(0.5));
        state2.set(PhysicalQuantity::Pressure, PhysicalData::from_scalar(101325.0));

        let result = state1 + state2;

        assert_eq!(result.len(), 3);
        assert_eq!(result.get(PhysicalQuantity::Concentration).unwrap().as_scalar(), 1.5);
        assert_eq!(result.get(PhysicalQuantity::Temperature).unwrap().as_scalar(), 300.0);
        assert_eq!(result.get(PhysicalQuantity::Pressure).unwrap().as_scalar(), 101325.0);
    }

    #[test]
    fn test_mul_scalar() {
        let mut state = PhysicalState::empty();
        state.set(PhysicalQuantity::Concentration, PhysicalData::uniform_vector(100, 2.0));
        state.set(PhysicalQuantity::Temperature, PhysicalData::from_scalar(300.0));

        let scaled = state * 0.5;

        let conc = scaled.get(PhysicalQuantity::Concentration).unwrap();
        assert_eq!(conc.as_vector()[0], 1.0);

        let temp = scaled.get(PhysicalQuantity::Temperature).unwrap();
        assert_eq!(temp.as_scalar(), 150.0);
    }
}