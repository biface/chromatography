//! Physical data types
//!
//! This module provides a flexible container for physical quantities
//! that can be scalars, vectors, matrices, or n-dimensional arrays
//! depending on the problem's dimensionality.
//!
//! # Design Philosophy
//!
//! `PhysicalData` is an enum that optimizes memory usage by selecting
//! the appropriate representation based on dimensionality:
//!
//! - **Scalar**: Single value (temperature, pressure)
//! - **Vector**: 1D spatial profiles (concentration along column)
//! - **Matrix**: 2D data (spatial + species, or 2D spatial)
//! - **Array**: 3D+ data (spatial + species + time, etc.)
//!
//! # Memory Efficiency
//!
//! | Type | Memory | Example Use Case |
//! |------|--------|------------------|
//! | Scalar | 8 bytes | Uniform temperature |
//! | Vector[n] | 8n bytes | 1D concentration profile |
//! | Matrix[n×m] | 8nm bytes | 100 points × 3 species |
//! | Array[n₁×...×nₖ] | 8∏nᵢ bytes | 3D spatial + species |

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
/// use chrom_rs::physics::PhysicalData;
/// use nalgebra::DVector;
///
/// // Scalar: uniform value
/// let temp = PhysicalData::from_scalar(298.15);
///
/// // Vector: 1D concentration profile
/// let conc = PhysicalData::uniform_vector(100, 1.0);
///
/// // Matrix: 100 spatial points × 3 species
/// let multi_species = PhysicalData::uniform_matrix(100, 3, 0.5);
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
    /// - 1D spatial + multi-species: M[point, species]
    /// - 2D spatial, single species: M[x, y]
    Matrix(DMatrix<f64>),

    /// Multidimensional array (3D+) - 8∏nᵢ bytes
    ///
    /// Use for:
    /// - 2D spatial + species: A[x, y, species]
    /// - 3D spatial + species: A[x, y, z, species]
    /// - Time series + spatial: A[time, x, y]
    Array(ArrayD<f64>),
}

impl PhysicalData {
    // ======================================= Constructors =======================================

    /// Create from scalar value
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let temp = PhysicalData::from_scalar(298.15);
    /// assert_eq!(temp.as_scalar(), 298.15);
    /// ```
    pub fn from_scalar(value: f64) -> Self {
        Self::Scalar(value)
    }

    /// Create from Vec<f64>
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(data.as_vector().len(), 3);
    /// ```
    pub fn from_vec(vector: Vec<f64>) -> Self {
        Self::Vector(DVector::from_vec(vector))
    }

    /// Create from DVector (zero-copy)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    /// use nalgebra::DVector;
    ///
    /// let vec = DVector::from_element(100, 1.0);
    /// let data = PhysicalData::from_vector(vec);
    /// ```
    pub fn from_vector(vector: DVector<f64>) -> Self {
        Self::Vector(vector)
    }

    /// Create from DMatrix (zero-copy)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    /// use nalgebra::DMatrix;
    ///
    /// let matrix = DMatrix::from_element(100, 3, 0.5);
    /// let data = PhysicalData::from_matrix(matrix);
    /// ```
    pub fn from_matrix(matrix: DMatrix<f64>) -> Self {
        Self::Matrix(matrix)
    }

    /// Create from ndarray (zero-copy)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    /// use ndarray::Array;
    ///
    /// let array = Array::from_elem((10, 10, 5), 1.0).into_dyn();
    /// let data = PhysicalData::from_array(array);
    /// ```
    pub fn from_array(array: ArrayD<f64>) -> Self {
        Self::Array(array)
    }

    /// Create uniform vector
    ///
    /// All elements initialized to the same value.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let data = PhysicalData::uniform_vector(100, 1.0);
    /// assert_eq!(data.as_vector()[50], 1.0);
    /// ```
    pub fn uniform_vector(size: usize, value: f64) -> Self {
        Self::Vector(DVector::from_element(size, value))
    }

    /// Create uniform matrix
    ///
    /// All elements initialized to the same value.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let data = PhysicalData::uniform_matrix(100, 3, 0.5);
    /// assert_eq!(data.as_matrix()[(0, 0)], 0.5);
    /// ```
    pub fn uniform_matrix(rows: usize, cols: usize, value: f64) -> Self {
        Self::Matrix(DMatrix::from_element(rows, cols, value))
    }

    /// Create uniform n-D array from shape
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let data = PhysicalData::uniform_array(&[10, 10, 5], 1.0);
    /// assert_eq!(data.shape(), vec![10, 10, 5]);
    /// ```
    pub fn uniform_array(shape: &[usize], value: f64) -> Self {
        Self::Array(Array::from_elem(IxDyn(shape), value))
    }

    // ==================================== Vector Operations ====================================

    /// Insert a value at a specific index in a vector
    ///
    /// # Arguments
    /// * `index` - Position to insert (0-based, can be at the end)
    /// * `value` - Value to insert
    ///
    /// # Panics
    /// If index > vector length or if called on non-Vector type
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    /// use nalgebra::DVector;
    ///
    /// let vec = DVector::from_vec(vec![1.0, 2.0, 4.0, 5.0]);
    /// let mut data = PhysicalData::Vector(vec);
    ///
    /// data.insert_at_index(2, 3.0);
    /// // Result: [1.0, 2.0, 3.0, 4.0, 5.0]
    /// assert_eq!(data.as_vector().len(), 5);
    /// assert_eq!(data.as_vector()[2], 3.0);
    /// ```
    pub fn insert_at_index(&mut self, index: usize, value: f64) {
        let new_data = match self {
            PhysicalData::Vector(v) => {
                let n = v.len();
                assert!(
                    index <= n,
                    "Index {} out of bounds for vector of length {}",
                    index,
                    n
                );

                // Create new vector with size n+1 (single allocation)
                let new_vec = DVector::from_fn(n + 1, |i, _| {
                    if i < index {
                        v[i]
                    } else if i == index {
                        value
                    } else {
                        v[i - 1]
                    }
                });

                PhysicalData::Vector(new_vec)
            }

            _ => panic!("insert_at_index only works with Vector"),
        };

        *self = new_data;
    }

    /// Remove value at a specific index from a vector
    ///
    /// # Arguments
    /// * `index` - Position to remove (0-based)
    ///
    /// # Panics
    /// If index >= vector length or if called on non-Vector type
    ///
    /// # Behavior
    /// - Vector[2] → Scalar (when removing from 2-element vector)
    /// - Vector[n] → Vector[n-1] (otherwise)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    /// use nalgebra::DVector;
    ///
    /// let vec = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let mut data = PhysicalData::Vector(vec);
    ///
    /// data.remove_at_index(2);
    /// // Result: [1.0, 2.0, 4.0, 5.0]
    /// assert_eq!(data.as_vector().len(), 4);
    /// ```
    pub fn remove_at_index(&mut self, index: usize) {
        let new_data = match self {
            PhysicalData::Vector(v) => {
                let n = v.len();
                assert!(
                    index < n,
                    "Index {} out of bounds for vector of length {}",
                    index,
                    n
                );

                if n == 2 {
                    // Vector[2] → Scalar
                    let remain_index = if index == 0 { 1 } else { 0 };
                    PhysicalData::Scalar(v[remain_index])
                } else {
                    // Vector[n] → Vector[n-1]
                    let new_vec = DVector::from_fn(n - 1, |i, _| {
                        if i < index {
                            v[i]
                        } else {
                            v[i + 1]
                        }
                    });

                    PhysicalData::Vector(new_vec)
                }
            }

            _ => panic!("remove_at_index only works with Vector"),
        };

        *self = new_data;
    }

    // ==================================== Matrix Column Operations ====================================

    /// Add a column from DVector to a Vector (creates Matrix) or Matrix
    ///
    /// # Interpretation
    ///
    /// `DVector` is treated as a **column vector** (vertical orientation).
    ///
    /// # Behavior
    /// - Vector[n] + Column[n] → Matrix[n, 2]
    /// - Matrix[n, m] + Column[n] → Matrix[n, m+1]
    ///
    /// # Arguments
    /// * `column` - DVector to add as a new column
    ///
    /// # Panics
    /// If column length doesn't match the number of rows
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    /// use nalgebra::DVector;
    ///
    /// // Vector is treated as a column: [[1], [2], [3]]
    /// let col1 = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let mut data = PhysicalData::Vector(col1);
    ///
    /// // Add second column
    /// let col2 = DVector::from_vec(vec![4.0, 5.0, 6.0]);
    /// data.add_column(&col2);
    ///
    /// // Result: Matrix [3, 2]
    /// // [[1.0, 4.0],
    /// //  [2.0, 5.0],
    /// //  [3.0, 6.0]]
    /// assert_eq!(data.as_matrix().shape(), (3, 2));
    /// assert_eq!(data.as_matrix()[(0, 1)], 4.0);
    /// ```
    pub fn add_column(&mut self, column: &DVector<f64>) {
        let new_data = match self {
            PhysicalData::Vector(v) => {
                let n_rows = v.len();
                assert_eq!(
                    column.len(),
                    n_rows,
                    "Column length ({}) must match vector length ({})",
                    column.len(),
                    n_rows
                );

                // Vector → Matrix [n_rows, 2]
                let matrix = DMatrix::from_fn(n_rows, 2, |i, j| {
                    if j == 0 {
                        v[i]
                    } else {
                        column[i]
                    }
                });

                PhysicalData::Matrix(matrix)
            }

            PhysicalData::Matrix(m) => {
                let (n_rows, n_cols) = m.shape();
                assert_eq!(
                    column.len(),
                    n_rows,
                    "Column length ({}) must match matrix rows ({})",
                    column.len(),
                    n_rows
                );

                // Matrix [n, m] → Matrix [n, m+1]
                let new_matrix = DMatrix::from_fn(n_rows, n_cols + 1, |i, j| {
                    if j < n_cols {
                        m[(i, j)]
                    } else {
                        column[i]
                    }
                });

                PhysicalData::Matrix(new_matrix)
            }

            _ => panic!("add_column only works with Vector or Matrix"),
        };

        *self = new_data;
    }

    /// Remove a column from a Matrix
    ///
    /// # Behavior
    /// - Matrix[n, 2] → Vector[n] (when removing from 2-column matrix)
    /// - Matrix[n, m] → Matrix[n, m-1] (otherwise)
    ///
    /// # Arguments
    /// * `col_index` - Index of column to remove (0-based)
    ///
    /// # Panics
    /// If col_index >= number of columns or if called on non-Matrix type
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let mut data = PhysicalData::uniform_matrix(100, 3, 1.0);
    /// data.remove_column(1);
    /// // Matrix [100, 2]
    /// assert_eq!(data.as_matrix().ncols(), 2);
    ///
    /// data.remove_column(0);
    /// // Vector [100] (only 1 column left)
    /// assert!(data.is_vector());
    /// ```
    pub fn remove_column(&mut self, col_index: usize) {
        let new_data = match self {
            PhysicalData::Matrix(m) => {
                let (n_rows, n_cols) = m.shape();
                assert!(
                    col_index < n_cols,
                    "Column index {} out of bounds (matrix has {} columns)",
                    col_index,
                    n_cols
                );

                if n_cols == 2 {
                    // Matrix [n, 2] → Vector [n]
                    let remain_col = if col_index == 0 { 1 } else { 0 };
                    PhysicalData::Vector(m.column(remain_col).clone_owned())
                } else {
                    // Matrix [n, m] → Matrix [n, m-1]
                    let new_matrix = DMatrix::from_fn(n_rows, n_cols - 1, |i, j| {
                        if j < col_index {
                            m[(i, j)]
                        } else {
                            m[(i, j + 1)]
                        }
                    });

                    PhysicalData::Matrix(new_matrix)
                }
            }

            _ => panic!("remove_column only works with Matrix"),
        };

        *self = new_data;
    }

    // ==================================== Matrix Row Operations ====================================

    /// Add a row from DVector to a Vector (creates Matrix) or Matrix
    ///
    /// # Interpretation
    ///
    /// `DVector` is treated as a **row vector** (horizontal orientation).
    ///
    /// # Behavior
    /// - Vector[n] + Row[n] → Matrix[2, n]
    /// - Matrix[n, m] + Row[m] → Matrix[n+1, m]
    ///
    /// # Arguments
    /// * `row` - DVector representing the new row
    ///
    /// # Panics
    /// If row length doesn't match the vector length (for Vector) or number of columns (for Matrix)
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    /// use nalgebra::DVector;
    ///
    /// // Vector [0, 1, 2] is treated as a ROW: [[0, 1, 2]]
    /// let vec = DVector::from_vec(vec![0.0, 1.0, 2.0]);
    /// let mut data = PhysicalData::Vector(vec);
    ///
    /// // Add second row [3, 4, 5]
    /// let new_row = DVector::from_vec(vec![3.0, 4.0, 5.0]);
    /// data.add_row(&new_row);
    ///
    /// // Result: Matrix [2, 3]
    /// // [[0.0, 1.0, 2.0],
    /// //  [3.0, 4.0, 5.0]]
    /// assert_eq!(data.as_matrix().shape(), (2, 3));
    /// assert_eq!(data.as_matrix()[(1, 2)], 5.0);
    /// ```
    pub fn add_row(&mut self, row: &DVector<f64>) {
        let new_data = match self {
            PhysicalData::Vector(v) => {
                let n_cols = v.len();
                assert_eq!(
                    row.len(),
                    n_cols,
                    "Row length ({}) must match vector length ({})",
                    row.len(),
                    n_cols
                );

                // Vector → Matrix [2, n_cols]
                let matrix = DMatrix::from_fn(2, n_cols, |i, j| {
                    if i == 0 {
                        v[j]
                    } else {
                        row[j]
                    }
                });

                PhysicalData::Matrix(matrix)
            }

            PhysicalData::Matrix(m) => {
                let (n_rows, n_cols) = m.shape();
                assert_eq!(
                    row.len(),
                    n_cols,
                    "Row length ({}) must match matrix columns ({})",
                    row.len(),
                    n_cols
                );

                // Matrix [n, m] → Matrix [n+1, m]
                let new_matrix = DMatrix::from_fn(n_rows + 1, n_cols, |i, j| {
                    if i < n_rows {
                        m[(i, j)]
                    } else {
                        row[j]
                    }
                });

                PhysicalData::Matrix(new_matrix)
            }

            _ => panic!("add_row only works with Vector or Matrix"),
        };

        *self = new_data;
    }

    /// Remove a row from a Matrix or Vector
    ///
    /// # Behavior
    /// - Vector[n] → Vector[n-1]
    /// - Matrix[2, n] → Vector[n] (when removing from 2-row matrix)
    /// - Matrix[n, m] → Matrix[n-1, m] (otherwise)
    ///
    /// # Arguments
    /// * `row_index` - Index of row to remove (0-based)
    ///
    /// # Panics
    /// If row_index >= number of rows
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let mut data = PhysicalData::uniform_matrix(100, 3, 1.0);
    /// data.remove_row(50);
    /// // Matrix [99, 3]
    /// assert_eq!(data.as_matrix().nrows(), 99);
    /// ```
    pub fn remove_row(&mut self, row_index: usize) {
        let new_data = match self {
            PhysicalData::Vector(v) => {
                let n = v.len();
                assert!(
                    row_index < n,
                    "Row index {} out of bounds for vector of length {}",
                    row_index,
                    n
                );

                // Vector[n] → Vector[n-1]
                let new_vec = DVector::from_fn(n - 1, |i, _| {
                    if i < row_index {
                        v[i]
                    } else {
                        v[i + 1]
                    }
                });

                PhysicalData::Vector(new_vec)
            }

            PhysicalData::Matrix(m) => {
                let (n_rows, n_cols) = m.shape();
                assert!(
                    row_index < n_rows,
                    "Row index {} out of bounds (matrix has {} rows)",
                    row_index,
                    n_rows
                );

                if n_rows == 2 {
                    // Matrix [2, n] → Vector [n]
                    let remain_row = if row_index == 0 { 1 } else { 0 };
                    PhysicalData::Vector(m.row(remain_row).clone_owned().transpose())
                } else {
                    // Matrix [n, m] → Matrix [n-1, m]
                    let new_matrix = DMatrix::from_fn(n_rows - 1, n_cols, |i, j| {
                        if i < row_index {
                            m[(i, j)]
                        } else {
                            m[(i + 1, j)]
                        }
                    });

                    PhysicalData::Matrix(new_matrix)
                }
            }

            _ => panic!("remove_row only works with Vector or Matrix"),
        };

        *self = new_data;
    }

    // ==================================== Square Matrix Operations ====================================

    /// Extend a square matrix by adding one dimension
    ///
    /// This is used for interaction matrices where each dimension
    /// interacts with all others (e.g., Langmuir multi-species interactions).
    ///
    /// # Transitions
    /// - Scalar (1×1) → Matrix 2×2
    /// - Matrix n×n → Matrix (n+1)×(n+1)
    ///
    /// # Arguments
    /// * `diagonal_value` - Value on the diagonal for the new dimension
    /// * `off_diagonal_value` - Values off-diagonal (interactions with existing dimensions)
    ///
    /// # Panics
    /// If called on non-square matrix
    ///
    /// # Example: Langmuir Interaction Matrix
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// // Start: 1 species (K₁)
    /// let mut k_matrix = PhysicalData::Scalar(1.5);
    ///
    /// // Add species 2 → 2×2 matrix
    /// k_matrix = k_matrix.extend_square_matrix(2.0, 0.5);
    /// // [[1.5, 0.5],
    /// //  [0.5, 2.0]]
    ///
    /// // Add species 3 → 3×3 matrix
    /// k_matrix = k_matrix.extend_square_matrix(3.0, 0.3);
    /// // [[1.5, 0.5, 0.3],
    /// //  [0.5, 2.0, 0.3],
    /// //  [0.3, 0.3, 3.0]]
    /// ```
    pub fn extend_square_matrix(self, diagonal: f64, off_diagonal: f64) -> Self {
        match self {
            // Scalar → Matrix 2×2
            Self::Scalar(value) => {
                let matrix = DMatrix::from_row_slice(
                    2,
                    2,
                    &[value, off_diagonal, off_diagonal, diagonal],
                );
                PhysicalData::Matrix(matrix)
            }

            // Matrix n×n → Matrix (n+1)×(n+1)
            Self::Matrix(m) => {
                let n = m.nrows();
                assert_eq!(
                    n,
                    m.ncols(),
                    "extend_square_matrix only works with square matrices (got {}×{})",
                    n,
                    m.ncols()
                );

                let new_n = n + 1;

                // Use from_fn for efficient single allocation
                let extended = DMatrix::from_fn(new_n, new_n, |i, j| {
                    if i < n && j < n {
                        m[(i, j)] // Existing data
                    } else if i == n && j == n {
                        diagonal // New diagonal element
                    } else {
                        off_diagonal // New off-diagonal elements
                    }
                });

                PhysicalData::Matrix(extended)
            }

            _ => panic!("extend_square_matrix only works with Scalar or square Matrix"),
        }
    }

    /// Reduce a square matrix by removing one dimension
    ///
    /// # Transitions
    /// - Matrix 2×2 → Scalar (1×1)
    /// - Matrix (n+1)×(n+1) → Matrix n×n
    ///
    /// # Arguments
    /// * `dimension_index` - Index of the dimension to remove (0-based)
    ///
    /// # Panics
    /// If called on non-square matrix or if dimension_index out of bounds
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    /// use nalgebra::DMatrix;
    ///
    /// let matrix = DMatrix::from_row_slice(3, 3, &[
    ///     1.0, 0.5, 0.3,
    ///     0.5, 2.0, 0.4,
    ///     0.3, 0.4, 3.0,
    /// ]);
    /// let mut data = PhysicalData::Matrix(matrix);
    ///
    /// // Remove dimension 1 (middle species)
    /// data = data.reduce_square_matrix(1);
    /// // [[1.0, 0.3],
    /// //  [0.3, 3.0]]
    /// assert_eq!(data.as_matrix().shape(), (2, 2));
    /// ```
    pub fn reduce_square_matrix(self, dimension_index: usize) -> Self {
        match self {
            PhysicalData::Matrix(m) => {
                let n = m.nrows();
                assert_eq!(
                    n,
                    m.ncols(),
                    "reduce_square_matrix only works with square matrices (got {}×{})",
                    n,
                    m.ncols()
                );
                assert!(
                    dimension_index < n,
                    "Dimension index {} out of bounds (matrix is {}×{})",
                    dimension_index,
                    n,
                    n
                );

                if n == 2 {
                    // Matrix 2×2 → Scalar
                    let remain_idx = if dimension_index == 0 { 1 } else { 0 };
                    PhysicalData::Scalar(m[(remain_idx, remain_idx)])
                } else {
                    // Matrix n×n → Matrix (n-1)×(n-1)
                    let new_n = n - 1;

                    // Use from_fn for efficient single allocation
                    let reduced = DMatrix::from_fn(new_n, new_n, |i, j| {
                        let orig_i = if i < dimension_index { i } else { i + 1 };
                        let orig_j = if j < dimension_index { j } else { j + 1 };
                        m[(orig_i, orig_j)]
                    });

                    PhysicalData::Matrix(reduced)
                }
            }

            _ => panic!("reduce_square_matrix only works with square Matrix"),
        }
    }

    // ======================================== Accessors ========================================

    /// Get a column as DVector (clones the data)
    ///
    /// # Returns
    /// - `Some(DVector)` if index is valid
    /// - `None` if index out of bounds or wrong type
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let data = PhysicalData::uniform_matrix(10, 3, 1.0);
    /// let col = data.get_column(1).unwrap();
    /// assert_eq!(col.len(), 10);
    /// ```
    pub fn get_column(&self, col_index: usize) -> Option<DVector<f64>> {
        match self {
            PhysicalData::Vector(v) => {
                if col_index == 0 {
                    Some(v.clone())
                } else {
                    None
                }
            }

            PhysicalData::Matrix(m) => {
                if col_index < m.ncols() {
                    Some(m.column(col_index).clone_owned())
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Get a row as DVector (clones the data)
    ///
    /// # Returns
    /// - `Some(DVector)` if index is valid
    /// - `None` if index out of bounds or wrong type
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let data = PhysicalData::uniform_matrix(10, 3, 1.0);
    /// let row = data.get_row(5).unwrap();
    /// assert_eq!(row.len(), 3);
    /// ```
    pub fn get_row(&self, row_index: usize) -> Option<DVector<f64>> {
        match self {
            PhysicalData::Vector(v) => {
                if row_index < v.len() {
                    Some(DVector::from_element(1, v[row_index]))
                } else {
                    None
                }
            }

            PhysicalData::Matrix(m) => {
                if row_index < m.nrows() {
                    Some(m.row(row_index).clone_owned().transpose())
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    // ========================================== Queries ==========================================

    /// Check if data is scalar
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }

    /// Check if data is a vector
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_))
    }

    /// Check if data is a matrix
    pub fn is_matrix(&self) -> bool {
        matches!(self, Self::Matrix(_))
    }

    /// Check if data is an array
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    /// Get number of dimensions
    ///
    /// Returns: 0 (scalar), 1 (vector), 2 (matrix), 3+ (array)
    pub fn ndim(&self) -> usize {
        match self {
            PhysicalData::Scalar(_) => 0,
            PhysicalData::Vector(_) => 1,
            PhysicalData::Matrix(_) => 2,
            PhysicalData::Array(a) => a.ndim(),
        }
    }

    /// Get shape as a vector
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let data = PhysicalData::uniform_matrix(100, 3, 1.0);
    /// assert_eq!(data.shape(), vec![100, 3]);
    /// ```
    pub fn shape(&self) -> Vec<usize> {
        match self {
            PhysicalData::Scalar(_) => vec![],
            PhysicalData::Vector(v) => vec![v.len()],
            PhysicalData::Matrix(m) => vec![m.nrows(), m.ncols()],
            PhysicalData::Array(a) => a.shape().to_vec(),
        }
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        match self {
            PhysicalData::Scalar(_) => 1,
            PhysicalData::Vector(v) => v.len(),
            PhysicalData::Matrix(m) => m.len(),
            PhysicalData::Array(a) => a.len(),
        }
    }

    /// Check if empty (always false for current implementation)
    #[allow(clippy::len_without_is_empty)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimate memory usage in bytes
    ///
    /// Each f64 element uses 8 bytes.
    pub fn memory_bytes(&self) -> usize {
        8 * self.len()
    }

    // ======================================== Extractions ========================================

    /// Extract as a scalar (panics if not scalar)
    ///
    /// # Panics
    /// If called on non-Scalar variant
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let data = PhysicalData::Scalar(42.0);
    /// assert_eq!(data.as_scalar(), 42.0);
    /// ```
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

    /// Extract as a DVector (panics if not vector)
    ///
    /// # Panics
    /// If called on non-Vector variant
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

    /// Extract as a DMatrix (panics if not matrix)
    ///
    /// # Panics
    /// If called on non-Matrix variant
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

    /// Extract as an array (panics if not array)
    ///
    /// # Panics
    /// If called on non-Array variant
    pub fn as_array(&self) -> &ArrayD<f64> {
        match self {
            PhysicalData::Array(value) => value,
            _ => panic!("Not an array value"),
        }
    }

    /// Try to extract as an array
    pub fn try_as_array(&self) -> Option<&ArrayD<f64>> {
        match self {
            PhysicalData::Array(value) => Some(value),
            _ => None,
        }
    }

    // ====================================== Apply Functions ======================================

    /// Apply a function to all elements
    ///
    /// Uses parallel iteration for large data (>999 elements) when compiled
    /// with `parallel` feature.
    ///
    /// # Example
    /// ```rust
    /// use chrom_rs::physics::PhysicalData;
    ///
    /// let mut data = PhysicalData::uniform_vector(100, 1.0);
    /// data.apply(|x| x * 2.0);
    /// assert_eq!(data.as_vector()[0], 2.0);
    /// ```
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        match self {
            PhysicalData::Scalar(value) => *value = f(*value),

            PhysicalData::Vector(v) => {
                if v.len() > 999 {
                    #[cfg(feature = "parallel")]
                    v.as_mut_slice().par_iter_mut().for_each(|x| *x = f(*x));
                    #[cfg(not(feature = "parallel"))]
                    v.iter_mut().for_each(|x| *x = f(*x));
                } else {
                    v.iter_mut().for_each(|x| *x = f(*x));
                }
            }

            PhysicalData::Matrix(m) => {
                if m.len() > 999 {
                    #[cfg(feature = "parallel")]
                    m.as_mut_slice().par_iter_mut().for_each(|x| *x = f(*x));
                    #[cfg(not(feature = "parallel"))]
                    m.iter_mut().for_each(|x| *x = f(*x));
                } else {
                    m.iter_mut().for_each(|x| *x = f(*x));
                }
            }

            PhysicalData::Array(a) => {
                a.mapv_inplace(f);
            }
        }
    }
}

// ================================== Arithmetic Operators ==================================

impl std::ops::Add for PhysicalData {
    type Output = PhysicalData;

    fn add(self, rhs: Self) -> Self::Output {
        use PhysicalData::*;
        match (self, rhs) {
            // Scalar operations
            (Scalar(x), Scalar(y)) => Scalar(x + y),
            (Scalar(x), Vector(y)) | (Vector(y), Scalar(x)) => Vector(y.map(|e| e + x)),
            (Scalar(x), Matrix(y)) | (Matrix(y), Scalar(x)) => Matrix(y.map(|e| e + x)),
            (Scalar(x), Array(y)) | (Array(y), Scalar(x)) => Array(&y + x),

            // Vector operations
            (Vector(x), Vector(y)) => {
                assert_eq!(x.len(), y.len(), "Vector lengths must match");
                Vector(x + y)
            }

            // Matrix operations
            (Matrix(x), Matrix(y)) => {
                assert_eq!(x.shape(), y.shape(), "Matrix dimensions must match");
                Matrix(x + y)
            }

            // Array operations
            (Array(x), Array(y)) => {
                assert_eq!(x.shape(), y.shape(), "Array dimensions must match");
                Array(&x + &y)
            }

            _ => panic!("Cannot add different PhysicalData types (except with Scalar)"),
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

impl std::ops::MulAssign<f64> for PhysicalData {
    fn mul_assign(&mut self, scalar: f64) {
        match self {
            PhysicalData::Scalar(x) => *x *= scalar,
            PhysicalData::Vector(x) => *x *= scalar,
            PhysicalData::Matrix(x) => *x *= scalar,
            PhysicalData::Array(x) => *x *= scalar,
        }
    }
}

impl std::ops::Mul<PhysicalData> for f64 {
    type Output = PhysicalData;

    fn mul(self, rhs: PhysicalData) -> Self::Output {
        rhs * self
    }
}

// ======================================== Display ========================================

impl fmt::Display for PhysicalData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicalData::Scalar(value) => write!(f, "Scalar({})", value),
            PhysicalData::Vector(v) => write!(f, "Vector[{}]", v.len()),
            PhysicalData::Matrix(m) => write!(f, "Matrix[{} × {}]", m.nrows(), m.ncols()),
            PhysicalData::Array(a) => {
                let shape_str = a
                    .shape()
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(" × ");
                write!(f, "Array[{}]", shape_str)
            }
        }
    }
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DVector, DMatrix};

    // ======================================= Constructors =======================================

    #[test]
    fn test_from_scalar() {
        let data = PhysicalData::Scalar(1.0);
        assert!(data.is_scalar());
        assert_eq!(data.as_scalar(), 1.0);
        assert_eq!(data.ndim(), 0);
        assert_eq!(data.memory_bytes(), 8);
    }

    #[test]
    fn test_from_vec() {
        let data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(data.is_vector());
        assert_eq!(data.as_vector().len(), 3);
        assert_eq!(data.ndim(), 1);
        assert_eq!(data.memory_bytes(), 24);
    }

    #[test]
    fn test_from_vector() {
        let data = PhysicalData::from_vector(DVector::from_element(3, 0.5));
        assert!(data.is_vector());
        assert_eq!(data.as_vector().len(), 3);
        assert_eq!(data.ndim(), 1);
        assert_eq!(data.memory_bytes(), 24);
    }

    #[test]
    fn test_from_matrix() {
        let data = PhysicalData::from_matrix(DMatrix::from_element(4, 2, 1.0));
        assert!(data.is_matrix());
        assert_eq!(data.as_matrix().shape(), (4, 2) );

    }

    #[test]
    fn test_uniform_vector() {
        let data = PhysicalData::uniform_vector(4, 0.5);
        assert!(data.is_vector());
        assert_eq!(data.as_vector().len(), 4);
        assert_eq!(data.ndim(), 1);
        assert_eq!(data.memory_bytes(), 32);
        assert_eq!(data.as_vector()[0], 0.5);
        assert_eq!(data.as_vector()[3], 0.5);
    }

    #[test]
    fn test_uniform_matrix() {
        let data = PhysicalData::from_matrix(DMatrix::from_element(4, 2, 1.0));
        assert!(data.is_matrix());
        assert_eq!(data.ndim(), 2 );
        assert_eq!(data.as_matrix().shape(), (4, 2) );
        assert_eq!(data.shape(), &[4, 2] );
        assert_eq!(data.as_matrix()[(0,0)], 1.0 );
        assert_eq!(data.as_matrix()[(3,1)], 1.0);
    }

    #[test]
    fn test_uniform_array() {
        let data = PhysicalData::uniform_array(&[10, 5, 12], 13.0);
        assert!(data.is_array());
        assert_eq!(data.shape(), &[10, 5, 12]);
        assert_eq!(data.ndim(), 3);
        assert_eq!(data.as_array()[IxDyn(&[0, 1, 3])], 13.0);
    }

    // =================================== operation on vectors ===================================

    #[test]
    fn test_insert_at_beginning() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
        data.insert_at_index(0, -1.0);

        let result = data.as_vector();
        assert_eq!(result.len(), 4);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 3.0);
    }

    #[test]
    fn test_insert_at_middle() {
        let mut data = PhysicalData::from_vec(vec![0.0, 1.0, 3.0, 4.0]);
        data.insert_at_index(2, 2.0);
        let result = data.as_vector();
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 3.0);
        assert_eq!(result[4], 4.0);
    }

    #[test]
    fn test_insert_at_end() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        data.insert_at_index(4, 5.0);
        let result = data.as_vector();
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 4.0);
        assert_eq!(result[4], 5.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_insert_at_out_of_bounds() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        data.insert_at_index(5, 6.0);
    }

    #[test]
    fn test_remove_at_beginning() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        data.remove_at_index(0);
        let result = data.as_vector();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);

    }

    #[test]
    fn test_remove_at_middle() {
        let mut data = PhysicalData::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        data.remove_at_index(2);
        let result = data.as_vector();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 4.0);
    }

    #[test]
    fn test_remove_at_end() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        data.remove_at_index(3);
        let result = data.as_vector();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_remove_at_out_of_bounds() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        data.remove_at_index(4);
    }

    #[test]
    fn test_remove_at_index_swap_to_scalar() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0]);
        data.remove_at_index(0);

        assert!(data.is_scalar());
        assert_eq!(data.as_scalar(), 2.0);
    }

    // ================================= column and row operations =================================

    #[test]
    fn test_add_column_to_vector() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
        data.add_column(&DVector::from_column_slice(&[4.0, 5.0, 6.0]));

        assert!(data.is_matrix());

        let result = data.as_matrix();

        assert_eq!(result.shape(), (3,2));
        assert_eq!(result[(0,0)], 1.0);
        assert_eq!(result[(0,1)], 4.0);
        assert_eq!(result[(2,0)], 3.0);
    }

    #[test]
    fn test_add_column_to_matrix() {
        let mut data = PhysicalData::uniform_matrix(100, 2, 1.0);
        let vector = DVector::from_element(100, 0.5);
        data.add_column(&vector);
        let result = data.as_matrix();
        assert_eq!(result.shape(), (100,3));
        assert_eq!(result[(0,2)], 0.5);
    }

    #[test]
    fn test_add_multiple_columns() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
        data.add_column(&DVector::from_column_slice(&[4.0, 5.0, 6.0]));
        data.add_column(&DVector::from_column_slice(&[7.0, 8.0, 9.0]));

        let result = data.as_matrix();
        assert_eq!(result.shape(), (3,3));
        assert_eq!(result[(0,2)], 7.0);
    }

    #[test]
    #[should_panic(expected = "Column length")]
    fn test_add_column_wrong_length() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
        data.add_column(&DVector::from_column_slice(&[4.0, 5.0, 6.0, 7.0]));
    }

    #[test]
    fn test_remove_column_swap_to_vector() {
        let mut data = PhysicalData::uniform_matrix(100, 2, 1.0);
        data.remove_column(0);

        assert!(data.is_vector());
        let result = data.as_vector();
        assert_eq!(result.len(), 100);
        assert_eq!(result[99], 1.0);
    }

    #[test]
    fn test_remove_column_matrix() {
        let mut data = PhysicalData::uniform_matrix(10, 5, 1.0);
        assert_eq!(data.shape(), &[10, 5]);

        data.remove_column(2);
        assert_eq!(data.shape(), &[10, 4]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_remove_column_matrix_wrong_shape() {
        let mut data = PhysicalData::uniform_matrix(100,5, 1.0);
        data.remove_column(6);
    }

    #[test]
    fn test_add_row_to_vector() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
        data.add_row(&DVector::from_column_slice(&[4.0, 5.0, 6.0]));

        assert!(data.is_matrix());
        assert_eq!(data.shape(), &[2, 3]);
        assert_eq!(data.as_matrix()[(0, 0)], 1.0);
        assert_eq!(data.as_matrix()[(0, 2)], 3.0);
        assert_eq!(data.as_matrix()[(1, 0)], 4.0);
    }

    #[test]
    fn test_add_row_to_matrix() {
        let mut data = PhysicalData::uniform_matrix(10, 2, 1.0);
        data.add_row(&DVector::from_column_slice(&[2.0, 3.0]));

        let result = data.as_matrix();
        assert_eq!(result.shape(), (11, 2));
        assert_eq!(result[(0,0)], 1.0);
        assert_eq!(result[(9,0)], 1.0);
        assert_eq!(result[(9,1)], 1.0);
        assert_eq!(result[(10,0)], 2.0);
        assert_eq!(result[(10,1)], 3.0);
    }

    #[test]
    #[should_panic(expected = "Row length")]
    fn test_add_row_wrong_length() {
        let mut data = PhysicalData::uniform_matrix(10, 3, 1.0);
        data.add_row(&DVector::from_column_slice(&[2.0, 3.0]));
    }

    #[test]
    fn test_remove_row_matrix_swap_to_vector() {
        let matrix = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,]);

        let mut data = PhysicalData::from_matrix(matrix);

        data.remove_row(0);

        assert!(data.is_vector());
        let result = data.as_vector();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 4.0);
    }

    #[test]
    fn test_remove_row_matrix() {
        let mut data = PhysicalData::uniform_matrix(10, 3, 1.0);

        data.remove_row(5);
        assert!(data.is_matrix());
        assert_eq!(data.shape(), &[9,3]);
    }

    #[test]
    fn test_remove_row_vector() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
        data.remove_row(0);
        assert!(data.is_vector());
        assert_eq!(data.len(), 2);
        assert_eq!(data.as_vector()[0], 2.0);
    }

    // square matrices

    #[test]
    fn test_extend_scalar_to_matrix() {
        let data = PhysicalData::from_scalar(1.5);

        let result = data.extend_square_matrix(2.0, 0.5);

        assert!(result.is_matrix());
        assert_eq!(result.as_matrix()[(0, 0)], 1.5);
        assert_eq!(result.as_matrix()[(0, 1)], 0.5);
        assert_eq!(result.as_matrix()[(1, 0)], 0.5);
        assert_eq!(result.as_matrix()[(1, 1)], 2.0);
    }

    #[test]
    fn test_extend_square_matrix() {
        let data = PhysicalData::uniform_matrix(2, 2, 1.0);
        let result = data.extend_square_matrix(2.0, 0.5);
        assert!(result.is_matrix());
        assert_eq!(result.shape(), &[3,3]);
        assert_eq!(result.as_matrix()[(0, 0)], 1.0);
        assert_eq!(result.as_matrix()[(0, 1)], 1.0);
        assert_eq!(result.as_matrix()[(0, 2)], 0.5);
        assert_eq!(result.as_matrix()[(1, 0)], 1.0);
        assert_eq!(result.as_matrix()[(1, 1)], 1.0);
        assert_eq!(result.as_matrix()[(1, 2)], 0.5);
        assert_eq!(result.as_matrix()[(2, 0)], 0.5);
        assert_eq!(result.as_matrix()[(2, 1)], 0.5);
        assert_eq!(result.as_matrix()[(2, 2)], 2.0);
    }

    #[test]
    #[should_panic(expected = "square matrices")]
    fn test_extend_matrix_failed() {
        let data = PhysicalData::from_matrix(
            DMatrix::from_element(2, 3, 1.0)
        );

        data.extend_square_matrix(2.0, 0.5);
    }

    #[test]
    fn test_reduce_square_matrix_swap_to_vector() {
        let data = PhysicalData::from_matrix(
            DMatrix::from_element(2, 2, 1.0));

        let scalar = data.reduce_square_matrix(0);
        assert!(scalar.is_scalar());
        assert_eq!(scalar.as_scalar(), 1.0);
    }

    #[test]
    fn test_reduce_square_matrix() {
        let matrix = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.5, 0.3,
            0.5, 2.0, 0.4,
            0.3, 0.4, 3.0,
        ]);
        let data = PhysicalData::Matrix(matrix);

        let result = data.reduce_square_matrix(1);

        assert!(result.is_matrix());
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_matrix()[(0, 0)], 1.0);
        assert_eq!(result.as_matrix()[(0, 1)], 0.3);
        assert_eq!(result.as_matrix()[(1, 1)], 3.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_reduce_square_matrix_out_of_bounds() {
        let data = PhysicalData::uniform_matrix(2, 2, 1.0);

        data.reduce_square_matrix(5);
    }


    // ======================================= Accessors =======================================

    #[test]
    fn test_get_column_from_vector() {
        let vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let data = PhysicalData::Vector(vec);

        let col = data.get_column(0).unwrap();
        assert_eq!(col.len(), 3);
        assert_eq!(col[1], 2.0);

        assert!(data.get_column(1).is_none());
    }

    #[test]
    fn test_get_column_from_matrix() {
        let matrix = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let data = PhysicalData::Matrix(matrix);

        let col = data.get_column(1).unwrap();
        assert_eq!(col[0], 2.0);
        assert_eq!(col[1], 5.0);
        assert_eq!(col[2], 8.0);

        assert!(data.get_column(5).is_none());
    }

    #[test]
    fn test_get_row_from_vector() {
        let vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let data = PhysicalData::Vector(vec);

        let row = data.get_row(1).unwrap();
        assert_eq!(row.len(), 1);
        assert_eq!(row[0], 2.0);

        assert!(data.get_row(5).is_none());
    }

    #[test]
    fn test_get_row_from_matrix() {
        let matrix = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let data = PhysicalData::Matrix(matrix);

        let row = data.get_row(1).unwrap();
        assert_eq!(row[0], 4.0);
        assert_eq!(row[1], 5.0);
        assert_eq!(row[2], 6.0);

        assert!(data.get_row(5).is_none());
    }


    // ======================================== Queries ========================================

    #[test]
    fn test_is_scalar() {
        assert!(PhysicalData::Scalar(1.0).is_scalar());
        assert!(!PhysicalData::uniform_vector(10, 1.0).is_scalar());
    }

    #[test]
    fn test_is_vector() {
        assert!(PhysicalData::uniform_vector(10, 1.0).is_vector());
        assert!(!PhysicalData::Scalar(1.0).is_vector());
    }

    #[test]
    fn test_is_matrix() {
        assert!(PhysicalData::uniform_matrix(10, 3, 1.0).is_matrix());
        assert!(!PhysicalData::Scalar(1.0).is_matrix());
    }

    #[test]
    fn test_is_array() {
        assert!(PhysicalData::uniform_array(&[10, 10, 5], 1.0).is_array());
        assert!(!PhysicalData::Scalar(1.0).is_array());
    }

    #[test]
    fn test_ndim() {
        assert_eq!(PhysicalData::Scalar(1.0).ndim(), 0);
        assert_eq!(PhysicalData::uniform_vector(10, 1.0).ndim(), 1);
        assert_eq!(PhysicalData::uniform_matrix(10, 3, 1.0).ndim(), 2);
        assert_eq!(PhysicalData::uniform_array(&[10, 10, 5], 1.0).ndim(), 3);
    }

    #[test]
    fn test_shape() {
        assert_eq!(PhysicalData::Scalar(1.0).shape(), vec![]);
        assert_eq!(PhysicalData::uniform_vector(10, 1.0).shape(), vec![10]);
        assert_eq!(PhysicalData::uniform_matrix(10, 3, 1.0).shape(), vec![10, 3]);
        assert_eq!(PhysicalData::uniform_array(&[10, 10, 5], 1.0).shape(), vec![10, 10, 5]);
    }

    #[test]
    fn test_len() {
        assert_eq!(PhysicalData::Scalar(1.0).len(), 1);
        assert_eq!(PhysicalData::uniform_vector(10, 1.0).len(), 10);
        assert_eq!(PhysicalData::uniform_matrix(10, 3, 1.0).len(), 30);
    }

    #[test]
    fn test_memory_bytes() {
        assert_eq!(PhysicalData::Scalar(1.0).memory_bytes(), 8);
        assert_eq!(PhysicalData::uniform_vector(100, 1.0).memory_bytes(), 800);
        assert_eq!(PhysicalData::uniform_matrix(100, 3, 1.0).memory_bytes(), 2400);
        assert_eq!(PhysicalData::uniform_array(&[10, 10, 10], 1.0).memory_bytes(), 8000);
    }

    // ==================================== Extractions ====================================

    #[test]
    fn test_as_scalar() {
        let data = PhysicalData::Scalar(42.0);
        assert_eq!(data.as_scalar(), 42.0);
    }

    #[test]
    #[should_panic(expected = "Not a scalar")]
    fn test_as_scalar_panic() {
        let data = PhysicalData::uniform_vector(10, 1.0);
        data.as_scalar();
    }

    #[test]
    fn test_try_as_scalar() {
        let scalar = PhysicalData::Scalar(42.0);
        assert_eq!(scalar.try_as_scalar(), Some(42.0));

        let vector = PhysicalData::uniform_vector(10, 1.0);
        assert_eq!(vector.try_as_scalar(), None);
    }

    #[test]
    fn test_as_vector() {
        let data = PhysicalData::uniform_vector(10, 1.0);
        assert_eq!(data.as_vector().len(), 10);
    }

    #[test]
    fn test_as_matrix() {
        let data = PhysicalData::uniform_matrix(10, 3, 1.0);
        assert_eq!(data.as_matrix().shape(), (10, 3));
    }

    // ========================================== Apply ==========================================

    #[test]
    fn test_apply_function_to_scalar() {
        let mut data = PhysicalData::Scalar(42.0);
        data.apply(|v| v * v + 2.0 * v + 1.0);

        assert_eq!(data.as_scalar(), 1849.0);

    }

    #[test]
    fn test_apply_function_to_vector() {
        let mut data = PhysicalData::from_vec(vec![1.0, 2.0, 3.0]);
        let result = vec![4.0, 9.0, 16.0];

        data.apply(|v| v * v + 2.0 * v + 1.0);

        assert!(data.is_vector());
        for i in 0..result.len() - 1 {
            assert_eq!(data.as_vector()[i], result[i]);
        }
    }

    #[test]
    fn test_apply_function_to_matrix() {
        let mut data = PhysicalData::from_matrix(
            DMatrix::from_row_slice(3, 2, &[
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ]));

        let result =
            DMatrix::from_row_slice(3,
                                    2,
                                    &[4.0, 9.0, 16.0, 25.0, 36.0, 49.0]);

        data.apply(|v| v * v + 2.0 * v + 1.0);
        let (rows, cols) = data.as_matrix().shape();

        for i in 0..rows - 1 {
            for j in 0..cols - 1 {
                assert_eq!(data.as_matrix()[(i,j)], result[(i,j)]);
            }
        }
    }

    #[test]
    fn test_apply_function_to_large_matrix() {
        let mut data = PhysicalData::uniform_matrix(2000, 2000, 1.0);
        data.apply(|v| v * v + 2.0 * v + 1.0);

        assert_eq!(data.as_matrix()[(500,500)], 4.0);
    }

    // =================================== Arithmetic Operators ===================================

    #[test]
    fn test_add_scalars() {
        let a = PhysicalData::Scalar(1.0);
        let b = PhysicalData::Scalar(2.0);
        let c = a + b;
        assert_eq!(c.as_scalar(), 3.0);
    }

    #[test]
    fn test_add_scalar_to_vector() {
        let scalar = PhysicalData::Scalar(1.0);
        let vector = PhysicalData::uniform_vector(10, 2.0);
        let result = scalar + vector;
        assert_eq!(result.as_vector()[0], 3.0);
    }

    #[test]
    fn test_add_vectors() {
        let a = PhysicalData::uniform_vector(10, 1.0);
        let b = PhysicalData::uniform_vector(10, 2.0);
        let c = a + b;
        assert_eq!(c.as_vector()[0], 3.0);
    }

    #[test]
    #[should_panic(expected = "must match")]
    fn test_add_vectors_different_size() {
        let a = PhysicalData::uniform_vector(10, 1.0);
        let b = PhysicalData::uniform_vector(20, 2.0);
        let _ = a + b;
    }

    #[test]
    fn test_mul_scalar() {
        let data = PhysicalData::uniform_vector(10, 2.0);
        let result = data * 3.0;
        assert_eq!(result.as_vector()[0], 6.0);
    }

    #[test]
    fn test_mul_assign() {
        let mut data = PhysicalData::uniform_vector(10, 2.0);
        data *= 3.0;
        assert_eq!(data.as_vector()[0], 6.0);
    }

    #[test]
    fn test_scalar_mul_data() {
        let data = PhysicalData::uniform_vector(10, 2.0);
        let result = 3.0 * data;
        assert_eq!(result.as_vector()[0], 6.0);
    }

    // ========================================== Display ==========================================

    #[test]
    fn test_display_scalar() {
        let data = PhysicalData::Scalar(42.0);
        assert_eq!(format!("{}", data), "Scalar(42)");
    }

    #[test]
    fn test_display_vector() {
        let data = PhysicalData::uniform_vector(100, 1.0);
        assert_eq!(format!("{}", data), "Vector[100]");
    }

    #[test]
    fn test_display_matrix() {
        let data = PhysicalData::uniform_matrix(100, 3, 1.0);
        assert_eq!(format!("{}", data), "Matrix[100 × 3]");
    }

    #[test]
    fn test_display_array() {
        let data = PhysicalData::uniform_array(&[10, 10, 5], 1.0);
        assert_eq!(format!("{}", data), "Array[10 × 10 × 5]");
    }

}