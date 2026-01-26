//! Physical models traits and types
//!
//! This module defines the core API for physical models:
//! - `PhysicalModel`: trait for all physical models
//! - `PhysicalState`: flexible state container
//! - `PhysicalQuantity`: type-safe quantity identifiers

use nalgebra::DMatrix;
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
/// # Design (v0.2.0 Update)
///
/// All physical quantities are now stored as `DMatrix<f64>`:
/// - **Rows**: Spatial points (x-axis, typically 100-1000 points)
/// - **Columns**: Chemical species (1 for mono-species, N for multi-species)
///
/// ## Mono-species Systems
/// Matrix shape: (n_spatial_points, 1) - single column
///
/// ## Multi-species Systems
/// Matrix shape: (n_spatial_points, n_species)
///
/// # Backward Compatibility
///
/// Helper methods maintain API compatibility:
/// - `new_mono_species()` - Creates single-column matrix
/// - `get_mono()` - Extracts column as vector
///
/// # Type Safety
///
/// Uses enum `PhysicalQuantity` for type-safe quantity identifiers.
///
/// # Examples
///
/// ```rust
/// use chrom_rs::physics::{PhysicalState, PhysicalQuantity};
/// use nalgebra::DMatrix;
///
/// // Mono-species (backward compatible)
/// let mono = PhysicalState::new_mono_species(
///     PhysicalQuantity::Concentration,
///     vec![1.0, 0.8, 0.6, 0.4, 0.2]
/// );
///
/// // Multi-species (new capability)
/// let multi = PhysicalState::new_multi_species(
///     PhysicalQuantity::Concentration,
///     100,  // spatial points
///     3,    // species
///     0.5   // initial value
/// );
///
/// // Direct matrix creation
/// let state = PhysicalState::new(
///     PhysicalQuantity::Concentration,
///     DMatrix::from_element(100, 2, 1.0)
/// );
/// ```
#[derive(Debug, Clone)]
pub struct PhysicalState {
    /// Physical quantities stored in a dictionary
    quantities: HashMap<PhysicalQuantity, DMatrix<f64>>,

    /// Scalar metadata (optional, e.g. energy, mass, etc.)
    metadata: HashMap<String, f64>,
}

impl PhysicalState {
    /// Create a new state with primary quantity (direct matrix)
    ///
    /// # Arguments
    /// * `quantity` - Type of physical quantity
    /// * `value` - Matrix (n_spatial_points, n_species)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity};
    /// use nalgebra::DMatrix;
    ///
    /// let state = PhysicalState::new(
    ///     PhysicalQuantity::Concentration,
    ///     DMatrix::from_element(100, 1, 1.0)
    /// );
    /// ```
    pub fn new(quantity: PhysicalQuantity, value: DMatrix<f64>) -> Self {
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

    /// Create a mono-species state (convenience - backward compatible)
    ///
    /// Creates a matrix with shape (n_spatial_points, 1)
    ///
    /// # Arguments
    /// * `quantity` - Type of physical quantity
    /// * `values` - Vector of values at each spatial point
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity};
    ///
    /// let state = PhysicalState::mono_specie(
    ///     PhysicalQuantity::Concentration,
    ///     vec![1.0, 0.8, 0.6, 0.4, 0.2]
    /// );
    /// ```
    pub fn mono_specie(quantity: PhysicalQuantity, values: Vec<f64>) -> Self {
        let n_points = values.len();
        let matrix = DMatrix::from_vec(n_points, n_points, values);
        Self::new(quantity, matrix)
    }


    /// Add a new species to a physical quantity
    ///
    /// Expands the matrix by adding a new column at the specified index.
    /// If the quantity doesn't exist, creates it as a single-species matrix.
    ///
    /// # Arguments
    /// * `quantity` - Physical quantity to expand
    /// * `species_data` - Vector of values for the new species (must match n_spatial_points)
    /// * `species_idx` - Column index where to insert the new species
    ///
    /// # Panics
    /// - If `species_data` length doesn't match existing spatial points
    /// - If `species_idx` is out of bounds (> current n_species)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity};
    ///
    /// // Start with mono-species
    /// let mut state = PhysicalState::mono_specie(
    ///     PhysicalQuantity::Concentration,
    ///     vec![1.0, 2.0, 3.0]
    /// );
    ///
    /// // Add second species at index 1 (after first)
    /// state.add_specie(
    ///     PhysicalQuantity::Concentration,
    ///     vec![0.5, 0.6, 0.7],
    ///     1
    /// );
    ///
    /// // Now have 2 species: [1.0, 0.5], [2.0, 0.6], [3.0, 0.7]
    /// assert_eq!(state.n_species(PhysicalQuantity::Concentration).unwrap(), 2);
    /// ```
    pub fn add_specie(&mut self,
                      quantity: PhysicalQuantity,
                      data: Vec<f64>,
                      position: usize) {
        if let Some(matrix) = self.quantities.get_mut(&quantity) {
            // Data for this physical quantity exists
            let rows = matrix.nrows();
            let cols = matrix.ncols();

            // Guaranty data length
            assert_eq!(data.len(),
                       rows,
                       "Specie data length {} mismatch with model variables {}",
                       data.len(),
                       rows);

            // Guaranty index position

            assert!(position <= cols,
                    "New specie index {} is out of bounds (max {})",
                    position,
                    cols);

            // Building the new DMatrix

            let mut extended_matrix = DMatrix::zeros(rows, cols + 1);

            // Copying old matrix in new matrix to the insert position

            for col in 0..position {
                extended_matrix.set_column(col, &matrix.column(col));
            }

            // Insert the new specie

            extended_matrix.set_column(position, &nalgebra::DVector::from_vec(data));

            // Copy remaining value of old matrix

            for col in position.. cols {
                extended_matrix.set_column(col + 1, &matrix.column(col));
            }

            *matrix = extended_matrix;
        } else {
            assert_eq!(position,
                       0,
                       "Cannot insert at index {} when quantity doesn't exist (use index 0)",
                       position);

            let rows = data.len();
            let matrix = DMatrix::from_vec(rows, 1, data);
            self.quantities.insert(quantity, matrix);
        }
    }

    /// Add a new species to the end (convenience method)
    ///
    /// Appends a new species column at the end of the matrix.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity};
    ///
    /// let mut state = PhysicalState::mono_specie(
    ///     PhysicalQuantity::Concentration,
    ///     vec![1.0, 2.0, 3.0]
    /// );
    ///
    /// // Append second species
    /// state.append_specie(
    ///     PhysicalQuantity::Concentration,
    ///     vec![0.5, 0.6, 0.7]
    /// );
    ///
    /// assert_eq!(state.n_species(PhysicalQuantity::Concentration).unwrap(), 2);
    /// ```
    pub fn append_specie(&mut self, quantity: PhysicalQuantity, data: Vec<f64>) {
        let position = self.get(quantity).
            map_or(0, |matrix| matrix.ncols());
        self.add_specie(quantity, data, position);
    }

    /// Remove a species from a physical quantity
    ///
    /// Removes the specified column from the matrix.
    ///
    /// # Arguments
    /// * `quantity` - Physical quantity to modify
    /// * `species_idx` - Index of species to remove
    ///
    /// # Panics
    /// - If quantity doesn't exist
    /// - If species_idx is out of bounds
    /// - If trying to remove the last species (would leave empty matrix)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::{PhysicalState, PhysicalQuantity};
    ///
    /// let mut state = PhysicalState::mono_specie(
    ///     PhysicalQuantity::Concentration,
    ///     vec![1.0, 1.0, 1.0]
    /// );
    /// state.append_specie(PhysicalQuantity::Concentration, vec![2.0, 2.0, 2.0]);
    /// state.append_specie(PhysicalQuantity::Concentration, vec![3.0, 3.0, 3.0]);
    ///
    /// // Remove middle species (index 1)
    /// state.remove_specie(PhysicalQuantity::Concentration, 1);
    ///
    /// assert_eq!(state.n_species(PhysicalQuantity::Concentration).unwrap(), 2);
    /// ```
    pub fn remove_specie(&mut self, quantity: PhysicalQuantity, index: usize) {
        if let Some(matrix) = self.quantities.get_mut(&quantity) {
            let rows = matrix.nrows();
            let cols = matrix.ncols();

            // Verifying index in segment [0, cols[
            assert!(index < cols, "Index {} of specie is out of bounds (max {})", index, cols);

            // Verifying there are at least 2 species
            assert!(cols > 1, "Cannot remove the last specie at index. Use remove_quantity() instead.");

            let mut reduced_matrix = DMatrix::zeros(rows, cols - 1);

            for col in 0..index {
                reduced_matrix.set_column(col, &matrix.column(col));
            }

            for col in index+1 .. cols {
                reduced_matrix.set_column(col - 1, &matrix.column(col));
            }

            *matrix = reduced_matrix;
        } else {
            panic!("Quantity {:?} is not found in this state", quantity);
        }
    }

    


    /// Get a quantity by type
    pub fn get(&self, quantity: PhysicalQuantity) -> Option<&DMatrix<f64>> {
        self.quantities.get(&quantity)
    }

    /// Get mutable reference to a quantity
    pub fn get_mut(&mut self, quantity: PhysicalQuantity) -> Option<&mut DMatrix<f64>> {
        self.quantities.get_mut(&quantity)
    }

    /// Set a quantity
    pub fn set (&mut self, quantity: PhysicalQuantity, value: DMatrix<f64>) {
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
            DMatrix::from_element(2, 2, 1.0),
        );

        assert_eq!(physics.quantities.len(), 1);
        assert_eq!(physics.metadata.len(), 0);
        assert!(physics.available_quantities().contains(&quantity));

        let values = physics.get(quantity).unwrap();

        assert_eq!(values.nrows(), 2);
        assert_eq!(values.ncols(), 2);
    }


}