//! n-dimensional domain boundaries with time convention
//!
//! # Design Philosophy
//!
//! Instead of enumerating 1D, 2D, 3D separately, we use a generic
//! structure that works for n dimensions:
//!
//! - `Vec<DimensionBoundary>`: boundaries for each dimension
//! - `TimeAxisConvention`: which dimension (if any) is temporal
//!
//! This allows arbitrary dimensional problems without artificial limits.

use crate::physics::PhysicalState;
use std::fmt;

// =================================================================================================
// Domain Boundaries
// =================================================================================================

/// n-dimensional domain boundaries
///
/// # Design
///
/// Stores boundary states directly as vectors of PhysicalState,
/// without typing them as Dirichlet/Neumann/etc.
///
/// - Spatial dimension: [left_state, right_state]
/// - Temporal dimension: [initial_state] or [initial, final]
///
/// The solver interprets how to use these states.
///
/// # Examples
///
/// ```rust
/// // ODE: temporal only
/// let boundaries = DomainBoundaries::temporal(initial_state);
///
/// // 1D spatial + time
/// let boundaries = DomainBoundaries::space_time_1d(
///     x_left, x_right, initial
/// );
///
/// // 3D spatial + time
/// let boundaries = DomainBoundaries::new(vec![
///     DimensionBoundary::spatial("x", x_left, x_right),
///     DimensionBoundary::spatial("y", y_bottom, y_top),
///     DimensionBoundary::spatial("z", z_front, z_back),
///     DimensionBoundary::temporal("t", initial),
/// ]);
/// ```
#[derive(Debug, Clone)]
pub struct DomainBoundaries {
    /// Boundaries for each dimension
    pub dimensions: Vec<DimensionBoundary>,

    /// Convention for identifying time dimensions
    pub  convention: TimeAxisConvention,
}

impl DomainBoundaries {
    /// Create without axis convention
    ///
    /// default axis convention is set to default ```rust TimeAxisConvention::Last```
    pub fn new(dimensions: Vec<DimensionBoundary>) -> Self {
        Self {
            dimensions,
            convention: TimeAxisConvention::Last,
        }
    }

    /// Create with defined axis convention
    ///
    ///
    pub fn create(dimensions: Vec<DimensionBoundary>,
    convention: Option<TimeAxisConvention>) -> Self {
        Self {
            dimensions,
            convention: convention.unwrap_or(TimeAxisConvention::Last),
        }
    }

    // ====================================== Factory methods ======================================

    /// Create temporal-only domain (ODE)
    ///
    /// Creates a 1D domain with time as the only dimension.
    /// Convention is automatically set to First.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let initial = PhysicalState::new()
    ///     .with_concentration("A", 1.0);
    ///
    /// let boundaries = DomainBoundaries::temporal(initial);
    ///
    /// assert_eq!(boundaries.ndim(), 1);
    /// assert_eq!(boundaries.spatial_ndim(), 0);
    /// assert!(boundaries.is_time_dependent());
    /// ```
    pub fn temporal(initial: PhysicalState) -> Self {
        Self::new(
            vec![DimensionBoundary::new(
                "t",
                vec![initial],
            )]
        )
    }

    /// Create n-dimensional spatial domain (steady-state)
    ///
    /// Creates a spatial domain with n dimensions, where each dimension
    /// has left and right boundary states. No time dimension is included.
    ///
    /// # Arguments
    ///
    /// * `names` - Names for each spatial dimension (e.g., `&["x", "y", "z"]`)
    /// * `begins` - Boundary states at lower bounds (left/bottom/front)
    /// * `ends` - Boundary states at upper bounds (right/top/back)
    ///
    /// # Panics
    ///
    /// Panics if `names`, `begins`, and `ends` have different lengths.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // 2D spatial domain
    /// let boundaries = DomainBoundaries::spatial(
    ///     &["x", "y"],
    ///     vec![x_left, y_bottom],
    ///     vec![x_right, y_top]
    /// );
    ///
    /// assert_eq!(boundaries.ndim(), 2);
    /// assert_eq!(boundaries.spatial_ndim(), 2);
    /// assert!(!boundaries.is_time_dependent());
    /// ```
    ///
    /// ```rust
    /// // 3D spatial domain
    /// let boundaries = DomainBoundaries::spatial(
    ///     &["x", "y", "z"],
    ///     vec![x_left, y_bottom, z_front],
    ///     vec![x_right, y_top, z_back]
    /// );
    ///
    /// assert_eq!(boundaries.ndim(), 3);
    /// assert_eq!(boundaries.spatial_ndim(), 3);
    /// ```
    pub fn spatial(names: &[&str],
                   begins: Vec<PhysicalState>,
                   ends: Vec<PhysicalState>) -> Self {

        assert_eq!(names.len(), begins.len());
        assert_eq!(names.len(), ends.len());

        let dimensions: Vec<_>= names.iter()
        .zip(begins.iter().zip(ends.iter()))
        .map(|(name, (begin, end))| {
            DimensionBoundary::new(*name, vec![begin.clone(), end.clone()])
        }).collect();

        Self::create(dimensions, Some(TimeAxisConvention::None))
    }

    /// Create n-dimensional spatial + temporal domain
    ///
    /// Creates a mixed domain with n spatial dimensions and one temporal dimension.
    /// The temporal dimension is placed last (convention: Last).
    ///
    /// # Arguments
    ///
    /// * `names` - Names for spatial dimensions (e.g., `&["x", "y", "z"]`)
    /// * `begins` - Boundary states at spatial lower bounds
    /// * `ends` - Boundary states at spatial upper bounds
    /// * `initial` - Initial condition (temporal boundary at t=0)
    ///
    /// # Panics
    ///
    /// Panics if `names`, `begins`, and `ends` have different lengths.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // 1D space + time
    /// let boundaries = DomainBoundaries::mixed(
    ///     &["x"],
    ///     vec![x_left],
    ///     vec![x_right],
    ///     initial
    /// );
    ///
    /// assert_eq!(boundaries.ndim(), 2);
    /// assert_eq!(boundaries.spatial_ndim(), 1);
    /// assert!(boundaries.is_time_dependent());
    /// assert_eq!(boundaries.time_convention, TimeAxisConvention::Last);
    /// ```
    ///
    /// ```rust
    /// // 3D space + time
    /// let boundaries = DomainBoundaries::mixed(
    ///     &["x", "y", "z"],
    ///     vec![x_left, y_bottom, z_front],
    ///     vec![x_right, y_top, z_back],
    ///     initial
    /// );
    ///
    /// assert_eq!(boundaries.ndim(), 4);
    /// assert_eq!(boundaries.spatial_ndim(), 3);
    /// ```
    pub fn mixed(names: &[&str],
    begins: Vec<PhysicalState>,
    ends: Vec<PhysicalState>,
    initial: PhysicalState) -> Self {
        assert_eq!(names.len(), begins.len());
        assert_eq!(names.len(), ends.len());

        let mut dimensions: Vec<_>= names.iter()
            .zip(begins.iter().zip(ends.iter()))
            .map(|(name, (begin, end))| {
                DimensionBoundary::new(*name, vec![begin.clone(), end.clone()])
            }).collect();

        dimensions.push(DimensionBoundary::new("t", vec![initial]));

        Self::create(dimensions, Some(TimeAxisConvention::Last))
    }

    // ===================================== Query methods =========================================

    /// Total number of dimensions
    pub fn ndim(&self) -> usize {
        self.dimensions.len()
    }

    /// Total number of spatial dimensions
    pub fn sdim(&self) -> usize {
        match self.convention {
            TimeAxisConvention::None => self.ndim(),
            _ => self.ndim() - 1
        }
    }

    /// Check time dependant equation
    pub fn is_time_dependent(&self) -> bool {
        self.convention != TimeAxisConvention::None
    }

    /// Get time dimension index
    pub fn time_index(&self) -> Option<usize> {
        match self.convention {
            TimeAxisConvention::Last => Some(self.ndim() - 1),
            TimeAxisConvention::First => Some(0),
            TimeAxisConvention::None => None,
            TimeAxisConvention::Index(i) => Some(i)
        }
    }

    /// Get temporal boundary
    pub fn time_boundary(&self) -> Option<&DimensionBoundary> {
        self.time_index().and_then(|index| self.dimensions.get(index))
    }

    /// Ges initial condition as the first physical state of temporal boundary
    pub fn initial_condition(&self) -> Option<&PhysicalState> {
        self.time_boundary().and_then(|boundary| boundary.states.first())
    }

    /// Get spatial boundaries
    pub fn spatial_boundaries(&self) -> Vec<&DimensionBoundary> {
        let excl_idx = self.time_index();

        self.dimensions
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != excl_idx)
            .map(|(_, dimension)| dimension)
            .collect()
    }

    /// Get dimension by its name
    pub fn get_boundary(&self, name: &str) -> Option<&DimensionBoundary> {
        self.dimensions.iter().find(|boundary| boundary.name == name)
    }

    /// Validate the object contents
    pub fn validate(&self) -> Result<(), String> {
        // Validate it is not empty
        if self.dimensions.is_empty() {
            return Err("Dimension boundaries cannot be empty.".into());
        }

        // Validate each dimension
        for dimension in &self.dimensions {
            dimension.validate()?;
        }

        // Check unicity of dimension's name

        let names:Vec<&str> = self.dimensions.iter()
            .map(|d| d.name.as_str())
        .collect();

        let unicity: std::collections::HashSet<&str> = names.iter().copied().collect();

        if unicity.len() != names.len() {
            return Err("It is impossible to store two dimensions with the same name.".into());
        }

        Ok(())
    }
}

impl Default for DomainBoundaries {
    fn default() -> Self {
        Self {
            dimensions: Vec::new(),
            convention: TimeAxisConvention::None,
        }
    }
}

// =================================================================================================
// Dimension Boundary
// =================================================================================================

/// Boundary for one dimension (variable)
///
/// # Design
///
/// Just stores boundary states - no typing as Spatial/Temporal.
/// TimeAxisConvention in DomainBoundaries identifies which is time.
///
/// # Convention
///
/// - 1 state: temporal dimension (initial condition)
/// - 2 states: spatial dimension (left, right boundaries)
#[derive(Debug, Clone)]
pub struct DimensionBoundary {
    /// Dimension name
    pub name: String,

    /// Physical states at boundaries
    pub states: Vec<PhysicalState>,
}

impl DimensionBoundary {
    /// Generic constructor
    pub fn new(name: impl Into<String>, states: Vec<PhysicalState>) -> Self {
        Self { name: name.into(), states }
    }

    /// Get first boundary state
    pub fn first(&self) -> Option<&PhysicalState> {
        Some(&self.states[0])
    }

    /// Get last boundary state
    pub fn last(&self) -> Option<&PhysicalState> {
        Some(&self.states[self.states.len() - 1])
    }

    /// Get size of boundaries
    pub fn size(&self) -> usize {
        self.states.len()
    }

    /// Verify if there are no boundaries
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Validate dimension
    pub fn validate(&self) -> Result<(), String> {
        if self.is_empty() {
            return Err(format!(
                "Dimensions '{}' must have at least one boundary state",
                self.name)
            );
        }

        Ok(())
    }
}

// =================================================================================================
// Time Axis Convention
// =================================================================================================

/// Convention for identifying registration of time variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeAxisConvention {
    /// No time dimension (steady-state)
    None,

    /// First dimension is time (t, x, y, z, ...)
    First,

    /// Last dimension is time (x, y, z,..., t)
    Last,

    /// Explicit index: dimension 'u' is time dimension
    Index(usize),
}

impl fmt::Display for TimeAxisConvention {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TimeAxisConvention::None => write!(f, "None"),
            TimeAxisConvention::First => write!(f, "First"),
            TimeAxisConvention::Last => write!(f, "Last"),
            TimeAxisConvention::Index(u) => write!(f, "Index ({})", u),
        }
    }
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use nalgebra::DVector;
    use num::integer::div_mod_floor;
    use crate::physics::PhysicalQuantity;
    use super::*;

    // =================================== Time Axis Convention ===================================

    #[test]
    fn test_axis_convention_first() {
        let first = TimeAxisConvention::First;

        assert_eq!(format!("{}", first), "First");
    }

    #[test]
    fn test_axis_convention_last() {
        let last = TimeAxisConvention::Last;
        assert_eq!(format!("{}", last), "Last");
    }

    #[test]
    fn test_axis_convention_index() {
        let first = TimeAxisConvention::Index(10);
        assert_eq!(format!("{}", first), "Index (10)");
    }

    #[test]
    fn test_axis_convention_none() {
        let first = TimeAxisConvention::None;
        assert_eq!(format!("{}", first), "None");
    }

    // ==================================== Dimension Boundary ====================================

    #[test]
    fn test_dimension_boundary_new() {
        let dimension = DimensionBoundary::new(
            "volume",
            vec![PhysicalState::new(
                PhysicalQuantity::Concentration,
                DVector::from_row_slice(&[1., 1., 1., 1.]),
            )]
        );

        assert_eq!(dimension.name, "volume");
        assert_eq!(dimension.states.len(), 1);
        assert_eq!(dimension.size(), 1)
    }

    #[test]
    fn test_dimension_boundary_content() {
        let dimension = DimensionBoundary::new(
            "volume",
            vec![PhysicalState::new(
                PhysicalQuantity::Concentration,
                DVector::from_row_slice(&[1., 2., 3., 4.]),
            )]
        );

        assert!(dimension
            .first()
            .unwrap()
            .available_quantities().
            contains(&PhysicalQuantity::Concentration));

        let data = dimension.
            first()
            .unwrap()
            .get(PhysicalQuantity::Concentration)
            .unwrap();

        assert_eq!(data[0], 1.0) ;
        assert_eq!(data[2], 3.0) ;
    }



    // ===================================== Domain Boundary =====================================
    #[test]
    fn test_temporal_only() {
        let initial = PhysicalState::empty();
        let boundary = DomainBoundaries::temporal(initial);

        assert_eq!(boundary.ndim(), 1);
        assert_eq!(boundary.sdim(), 0);
        assert!(boundary.is_time_dependent());
        assert_eq!(boundary.convention, TimeAxisConvention::Last);

    }

    #[test]
    fn test_spatial_only() {
        let domain = DomainBoundaries::spatial(
            &["x", "y", "z"],
            vec![PhysicalState::empty(), PhysicalState::empty(), PhysicalState::empty()],
            vec![PhysicalState::empty(), PhysicalState::empty(), PhysicalState::empty()],
        ) ;

        assert_eq!(domain.ndim(), 3);
        assert_eq!(domain.sdim(), 3);
        assert!(!domain.is_time_dependent());
        assert_eq!(domain.convention, TimeAxisConvention::None);
    }

    #[test]
    fn test_mixed() {
        let initial = PhysicalState::new(
            PhysicalQuantity::Concentration,
            DVector::from_vec(vec![2.0])
        );

        let boundary = DomainBoundaries::mixed(
            &["x", "y", "z"],
            vec![PhysicalState::empty(), PhysicalState::empty(), PhysicalState::empty()],
            vec![PhysicalState::empty(), PhysicalState::empty(), PhysicalState::empty()],
            initial
        ) ;

        assert_eq!(boundary.ndim(), 4);
        assert_eq!(boundary.sdim(), 3);
        assert!(boundary.is_time_dependent());
        assert_eq!(boundary.convention, TimeAxisConvention::Last);
        assert_eq!(boundary.time_index(), Some(3));

        assert!(boundary.dimensions[0].name.contains("x"));
        assert!(boundary.dimensions[1].name.contains("y"));
        assert!(boundary.dimensions[2].name.contains("z"));
        assert!(boundary.dimensions[3].name.contains("t"));
    }

    #[test]
    fn test_mixed_spatial() {
        let initial = PhysicalState::new(
            PhysicalQuantity::Concentration,
            DVector::from_vec(vec![2.0])
        );

        let boundary = DomainBoundaries::mixed(
            &["x", "y", "z"],
            vec![PhysicalState::empty(), PhysicalState::empty(), PhysicalState::empty()],
            vec![PhysicalState::empty(), PhysicalState::empty(), PhysicalState::empty()],
            initial
        ) ;

        let spatials = boundary.spatial_boundaries();

        assert_eq!(spatials.len(), 3);
        assert_eq!(spatials[0].name, "x");
        assert_eq!(spatials[1].name, "y");
        assert_eq!(spatials[2].name, "z");

    }

    #[test]
    fn test_mixed_temporal() {
        let initial = PhysicalState::new(
            PhysicalQuantity::Concentration,
            DVector::from_vec(vec![2.0])
        );

        let boundary = DomainBoundaries::mixed(
            &["x", "y", "z"],
            vec![PhysicalState::empty(), PhysicalState::empty(), PhysicalState::empty()],
            vec![PhysicalState::empty(), PhysicalState::empty(), PhysicalState::empty()],
            initial
        ) ;

        let temporal = boundary.time_boundary();

        assert_eq!(temporal.is_some(), true);
        assert_eq!(temporal.unwrap().name, "t");

    }

    // Validation tests
    #[test]
    fn test_empty_boundary() {
        let false_boundary = DomainBoundaries::new(vec![]);
        let result = false_boundary.validate();

        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Dimension boundaries cannot be empty.");
    }

    #[test]
    fn test_duplicate_dimensions() {
        let false_boundary = DomainBoundaries {
            dimensions: vec![
                DimensionBoundary::new(
                    "x",
                    vec![PhysicalState::empty(), PhysicalState::empty()]
                ),
                DimensionBoundary::new(
                    "x",
                    vec![PhysicalState::empty(), PhysicalState::empty()]
                )
            ],
            convention: TimeAxisConvention::None,
        } ;

        let result = false_boundary.validate();
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "It is impossible to store two dimensions with the same name.");
    }

    #[test]
    fn test_domain_without_state() {
        let false_boundary = DomainBoundaries {
            dimensions: vec![
                DimensionBoundary::new("x", vec![])
            ],
            convention: TimeAxisConvention::None,
        };

        let result = false_boundary.validate();
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Dimensions 'x' must have at least one boundary state");
    }
}