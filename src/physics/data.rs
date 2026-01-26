//! Physical data types
//!
//! This module provides a flexible container for physical quantities
//! that can be scalars, vectors, or matrices depending on the problem's
//! dimensionality.

use nalgebra::{DVector, DMatrix};
use ndarray::{Array, ArrayD, IxDyn};
use std::fmt;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Physical data container supporting scalar to n-dimensional arrays
///
/// # Storage Types
///
/// - **Scalar**: Single uniform value (0D)
/// - **Vector**: 1D array (spatial variation, single species)
/// - **Matrix**: 2D array (1D spatial + species OR 2D spatial mono-species)
/// - **Array**: 3D+ multidimensional array (general n-D grids)
///
/// # NOT Tensors
///
/// Despite common abuse of terminology (NumPy, TensorFlow), these are
/// **multidimensional arrays**, not mathematical tensors with covariant/
/// contravariant transformation properties.
///
/// # Memory Layout
///
/// - **Scalar**: 8 bytes
/// - **Vector[n]**: 8n bytes
/// - **Matrix[n×m]**: 8nm bytes
/// - **Array[n₁×n₂×...×nₖ]**: 8∏nᵢ bytes
///
/// # Examples
///
/// ```rust
/// use ndarray::Array;
/// use chrom_rs::physics::PhysicalData;
///
/// // 3D array: 2D spatial grid (50×50) + 3 species
/// let data_3d = PhysicalData::from_array(
///     Array::from_elem((50, 50, 3), 0.5).into_dyn()
/// );
///
/// // 4D array: 3D spatial grid (30×30×30) + 5 species
/// let data_4d = PhysicalData::from_array(
///     Array::from_elem((30, 30, 30, 5), 0.1).into_dyn()
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalData {
    /// Scalar value (0D) - 8 bytes
    ///
    /// Use for: Uniform quantities (temperature, pressure)
    Scalar(f64),

    /// Vector (1D) - 8n bytes
    ///
    /// Use for: 1D spatial profiles, single species
    /// Example: Concentration along column axis
    Vector(DVector<f64>),

    /// Matrix (2D) - 8nm bytes
    ///
    /// Use for:
    /// - 1D spatial + multi-species: A[point, species]
    /// - 2D spatial, single species: A[x, y]
    Matrix(DMatrix<f64>),

    /// Multidimensional array (3D+) - 8∏nᵢ bytes
    ///
    /// Use for:
    /// - 2D spatial + species: A[x, y, species]
    /// - 3D spatial + species: A[x, y, z, species]
    /// - Time series + spatial: A[time, x, y]
    ///
    /// # Index Convention
    ///
    /// For spatial + species systems:
    /// - Geometric dimensions: spatial coordinates
    /// - Physical parameters
    /// - ...
    ///
    /// Example: A[x, y, z, s, p, t]
    Array(ArrayD<f64>),
}

impl PhysicalData {

    // ======================================= constructors =======================================

    /// Create from scalar

    pub fn from_scalar(value: f64) -> Self {
        Self::Scalar(value)
    }

    /// Create from vector

    pub fn from_vec(vector: Vec<f64>) -> Self {
        Self::Vector(DVector::from_vec(vector))
    }

    /// Create from DVector

    pub fn from_vector(vector: DVector<f64>) -> Self {
        Self::Vector(vector)
    }

    /// Create from DMatrix

    pub fn from_matrix(matrix: DMatrix<f64>) -> Self {
        Self::Matrix(matrix)
    }

    /// Create from array
    pub fn from_array(array: ArrayD<f64>) -> Self {
        Self::Array(array)
    }

    /// Create uniform vector

    pub fn uniform_vector(size: usize, value: f64) -> Self {
        Self::Vector(DVector::from_element(size, value))
    }

    /// Create uniform matrix
    pub fn uniform_matrix(rows: usize, columns: usize, value: f64) -> Self {
        Self::Matrix(DMatrix::from_element(rows, columns, value))
    }

    /// Create uniform n-D array from shape
    pub fn uniform_array(shape: &[usize], value: f64) -> Self {
        Self::Array(Array::from_elem(IxDyn(shape), value))
    }

    // ========================================== Queries ==========================================

    /// Check data is scalar

    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }

    /// Check data is a vector

    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_))
    }

    /// Check data is a matrix
    pub fn is_matrix(&self) -> bool {
        matches!(self, Self::Matrix(_))
    }

    /// Check data is an array
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    /// Get data dimension
    ///
    /// Returns: 0 (scalar), 1 (vector), 2 (matrix), 3+ (array)
    ///
    pub fn ndim(&self) -> usize {
        match self {
            PhysicalData::Scalar(_) => 0,
            PhysicalData::Vector(_) => 1,
            PhysicalData::Matrix(_) => 2,
            PhysicalData::Array(a) => a.ndim(),
        }
    }

    /// Get shape as a vector
    pub fn shape(&self) -> Vec<usize> {
        match self {
            PhysicalData::Scalar(_) => vec![],
            PhysicalData::Vector(v) => vec![v.len()],
            PhysicalData::Matrix(m) => vec![m.nrows(), m.ncols()],
            PhysicalData::Array(a) => a.shape().to_vec(),
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        match self {
            PhysicalData::Scalar(_) => 1,
            PhysicalData::Vector(v) => v.len(),
            PhysicalData::Matrix(m) => m.len(),
            PhysicalData::Array(a) => a.len(),
        }
    }

    /// Check emptiness
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimate memory usage in bytes
    pub fn memory(&self) -> usize {
        8 * self.len()
    }

    // ======================================== Extractions ========================================

    /// Extract as a scalar (panic if not)

    pub fn as_scalar(&self) -> f64 {
        match self {
            PhysicalData::Scalar(value) => *value,
            _ => panic!("Not a scalar value"),
        }
    }

    /// Try to extract as a scalar

    pub fn try_as_scalar(&self) -> Option<f64> {
        match self {
            PhysicalData::Scalar(value) => Some(*value),
            _ => None,
        }
    }

    /// Extract as a DVector (panic if not)
    pub fn as_vector(&self) -> &DVector<f64> {
        match self {
            PhysicalData::Vector(value) => value,
            _ => panic!("Not a vector value"),
        }
    }

    /// Try to extract as a DVector

    pub fn try_as_vector(&self) -> Option<&DVector<f64>> {
        match self {
            PhysicalData::Vector(value) => Some(value),
            _ => None,
        }
    }

    /// Extract as a DMatrix (Panic if not)
    pub fn as_matrix(&self) -> &DMatrix<f64> {
        match self {
            PhysicalData::Matrix(value) => value,
            _ => panic!("Not a matrix value"),
        }
    }

    /// Try to extract as a DMatrix
    pub fn try_as_matrix(&self) -> Option<&DMatrix<f64>> {
        match self {
            PhysicalData::Matrix(value) => Some(value),
            _ => None,
        }
    }

    /// Extract as an array (panic if not)
    pub fn as_array(&self) -> &ArrayD<f64> {
        match self {
            PhysicalData::Array(value) => value,
            _ => panic!("Not a array value"),
        }
    }

    /// Try to extract as an array
    pub fn try_as_array(&self) -> Option<&ArrayD<f64>> {
        match self {
            PhysicalData::Array(value) => Some(value),
            _ => None,
        }
    }

    // ====================================== Apply functions ======================================

    /// Apply a function f to data
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        match self {
            PhysicalData::Scalar(value) => *value = f(*value),

            PhysicalData::Vector(value) => {
                if value.len() > 999 {
                    #[cfg(feature = "parallel")]
                    value.as_mut_slice().par_iter_mut().for_each(|x| *x = f(*x));
                    #[cfg(not(feature = "parallel"))]
                    value.iter_mut().for_each(|x| *x = f(*x));
                } else {
                    value.iter_mut().for_each(|x| *x = f(*x));
                }
            }

            PhysicalData::Matrix(value) => {
                if value.len() > 999 {
                    #[cfg(feature = "parallel")]
                    value.as_mut_slice().par_iter_mut().for_each(|x| *x = f(*x));
                    #[cfg(not(feature = "parallel"))]
                    value.iter_mut().for_each(|x| *x = f(*x));
                } else {
                    value.iter_mut().for_each(|x| *x = f(*x));
                }
            }

            PhysicalData::Array(value) => {
                value.mapv_inplace(f);
            }
        }
    }
}

// ================================== Simple arithmetic functions ==================================

impl std::ops::Add for PhysicalData {
    type Output = PhysicalData;
    fn add(self, rhs: Self) -> Self::Output {
        use PhysicalData::*;
        match (self, rhs) {

            // Addition with scalar

            (Scalar(x), Scalar(y)) => Scalar(x + y),
            (Scalar(x), Vector(y)) |
            (Vector(y), Scalar(x)) => Vector(y.map(|e| e + x)),
            (Scalar(x), Matrix(y)) |
            (Matrix(y), Scalar(x)) => Matrix(y.map(|e| e + x)),
            (Scalar(x), Array(y)) |
            (Array(y), Scalar(x)) => Array(&y + x),

            // Addition with vectors

            (Vector(x), Vector(y)) => {
                assert_eq!(x.len(), y.len(), "Vector length must match");
                Vector(x + y)
            }

            // Addition with matrices

            (Matrix(x), Matrix(y)) => {
                assert_eq!(x.shape(), y.shape(), "Matrices dimensions must match");
                Matrix(x + y)
            }

            // Addition with arrays

            (Array(x), Array(y)) => {
                assert_eq!(x.shape(), y.shape(), "Arrays dimensions must macth");
                Array(&x + &y)
            }

            _ => panic!("Cannot add different PhysicalData types other than with scalar"),
        }
    }
}

impl std::ops::Mul<f64> for PhysicalData {
    type Output = PhysicalData;
    fn mul(self, scalar: f64) -> Self::Output {
        match self {
            PhysicalData::Scalar(x) => PhysicalData::Scalar(x * scalar),
            PhysicalData::Vector(x) => PhysicalData::Vector(x * scalar),
            PhysicalData::Matrix(x) => PhysicalData::Matrix(x * scalar),
            PhysicalData::Array(x) => PhysicalData::Array(&x * scalar),
        }
    }
}

impl std::ops::Mul<PhysicalData> for f64 {
    type Output = PhysicalData;
    fn mul(self, rhs: PhysicalData) -> Self::Output {
        rhs * self
    }
}

// ======================== Display ============================

impl fmt::Display for PhysicalData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicalData::Scalar(value) => write!(f, "Scalar ({})", value),
            PhysicalData::Vector(value) => write!(f, "Vector [{}]", value.len()),
            PhysicalData::Matrix(value) => write!(f, "Matrix [{} * {}]", value.nrows(), value.ncols()),
            PhysicalData::Array(value) => {
                let str_shape = value.shape()
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(" * ");
                write!(f, "Array [{}]", str_shape)
            }
        }
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar() {
        let data = PhysicalData::Scalar(42.0);
        assert!(data.is_scalar());
        assert_eq!(data.ndim(), 0);
        assert_eq!(data.memory(), 8);
    }

    #[test]
    fn test_vector() {
        let data = PhysicalData::uniform_vector(100, 1.0);
        assert!(data.is_vector());
        assert_eq!(data.ndim(), 1);
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_matrix() {
        let data = PhysicalData::uniform_matrix(100, 3, 0.5);
        assert!(data.is_matrix());
        assert_eq!(data.ndim(), 2);
        assert_eq!(data.shape(), vec![100, 3]);
    }


    #[test]
    fn test_addition() {
        let a = PhysicalData::Scalar(1.0);
        let b = PhysicalData::Scalar(2.0);
        let c = a + b;
        assert_eq!(c.as_scalar(), 3.0);
    }

    #[test]
    fn test_multiplication() {
        let data = PhysicalData::uniform_vector(10, 2.0);
        let result = data * 3.0;
        assert_eq!(result.as_vector()[0], 6.0);
    }

    #[test]
    fn test_memory_efficiency() {
        let scalar = PhysicalData::Scalar(1.0);
        let vector = PhysicalData::uniform_vector(100, 1.0);
        let matrix = PhysicalData::uniform_matrix(100, 3, 1.0);
        let array = PhysicalData::uniform_array(&[10, 10, 10], 50.0);

        assert_eq!(scalar.memory(), 8);
        assert_eq!(vector.memory(), 800);
        assert_eq!(matrix.memory(), 2400);
        assert_eq!(array.memory(), 8000);
    }
}