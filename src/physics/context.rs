//! Typed compute context for physical model evaluation.
//!
//! This module defines [`ComputeContext`], the type passed by solvers to
//! physical models at each time step. It replaces the previous mechanism
//! that injected the current time via `state.set_metadata("time", t)`.
//!
//! # Design (DD-008)
//!
//! `ComputeContext` is structurally aligned with the oxiflow `ComputeContext`,
//! without depending on it. Convergence is deferred to post-v1.0.0 (DD-014).
//!
//! The two infallible fields `time` and `time_step` are always available.
//! Optional derived variables are stored in a typed
//! `HashMap<ContextVariable, ContextValue>`.
//!
//! # Usage in solvers
//!
//! ```rust,no_run
//! use chrom_rs::physics::ComputeContext;
//!
//! // Inside the solver integration loop:
//! // let ctx = ComputeContext::new(t, dt);
//! // let physics = model.compute_physics(&state, &ctx);
//! ```
//!
//! # Usage in models
//!
//! ```rust,no_run
//! use chrom_rs::physics::{ComputeContext, PhysicalState};
//!
//! // Inside PhysicalModel::compute_physics:
//! // let t = ctx.time();
//! // let c_inlet = self.injection.evaluate(t);
//! ```

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;

// =============================================================================
// ContextVariable
// =============================================================================

/// Typed key for [`ComputeContext`] variables.
///
/// Each variant identifies a category of derived physical quantity.
/// `ContextVariable` implements `Hash + Eq` so it can be used as a
/// `HashMap` key.
///
/// # Variants
///
/// | Variant | Usage |
/// |---------|-------|
/// | `Time` | Current time $t$ (infallible duplicate of `ctx.time()`) |
/// | `TimeStep` | Time step $\Delta t$ (infallible duplicate of `ctx.time_step()`) |
/// | `SpatialGradient` | Spatial gradient $\partial / \partial x_d$ of a component |
/// | `External` | User-defined named variable |
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextVariable {
    /// Current simulation time $t$ \[s\].
    Time,

    /// Current time step $\Delta t$ \[s\].
    TimeStep,

    /// Spatial gradient along axis `dimension` for species / component `component`.
    SpatialGradient {
        /// Index of the spatial dimension (0 = axial for 1-D models).
        dimension: usize,
        /// Component (species) index — `None` for a scalar field.
        component: Option<usize>,
    },

    /// User-defined named variable.
    External {
        /// Variable name.
        name: Cow<'static, str>,
    },
}

// =============================================================================
// ContextValue
// =============================================================================

/// Typed value stored in [`ComputeContext`].
///
/// Aligned with oxiflow variants to facilitate future convergence (DD-014).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextValue {
    /// Scalar value.
    Scalar(f64),

    /// Boolean flag.
    Boolean(bool),

    /// Nodal scalar field — one value per spatial node.
    ///
    /// Un élément par nœud spatial $z_i$, $i = 0, \ldots, N_z - 1$.
    ScalarField(DVector<f64>),

    /// Nodal vector field — $N_z \times d$ matrix.
    ///
    /// Matrice $N_z \times d$ : une ligne par nœud, une colonne par dimension.
    VectorField(DMatrix<f64>),
}

// =============================================================================
// ComputeContext
// =============================================================================

/// Typed compute context passed to physical models at each time step.
///
/// Replaces implicit time injection via `state.set_metadata("time", t)` with
/// an explicit, infallible contract. `time()` and `time_step()` are always
/// available — no `unwrap` required in model implementations.
///
/// # Example
///
/// ```rust
/// use chrom_rs::physics::ComputeContext;
///
/// let ctx = ComputeContext::new(10.0, 0.01);
/// assert!((ctx.time() - 10.0).abs() < 1e-12);
/// assert!((ctx.time_step() - 0.01).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct ComputeContext {
    /// Current simulation time $t$ \[s\].
    time: f64,

    /// Current time step $\Delta t$ \[s\].
    time_step: f64,

    /// Optional derived variables.
    ///
    /// Variables dérivées optionnelles, indexées par [`ContextVariable`].
    variables: HashMap<ContextVariable, ContextValue>,
}

impl ComputeContext {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Creates a minimal context carrying `time` and `time_step`.
    ///
    /// The optional variable map is empty. Use [`insert`](Self::insert) to
    /// add derived quantities when needed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::physics::ComputeContext;
    ///
    /// let ctx = ComputeContext::new(10.0, 0.01);
    /// assert!((ctx.time() - 10.0).abs() < 1e-12);
    /// assert!((ctx.time_step() - 0.01).abs() < 1e-12);
    /// assert!(ctx.is_empty());
    /// ```
    pub fn new(time: f64, time_step: f64) -> Self {
        Self {
            time,
            time_step,
            variables: HashMap::new(),
        }
    }

    // =========================================================================
    // Infallible accessors
    // =========================================================================

    /// Current simulation time $t$ \[s\].
    ///
    /// Infallible — always available without `unwrap`.
    #[inline]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Current time step $\Delta t$ \[s\].
    ///
    /// Infallible — always available without `unwrap`.
    #[inline]
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    // =========================================================================
    // Variable accessors
    // =========================================================================

    /// Inserts or replaces a variable in the context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::physics::{ComputeContext, ContextVariable, ContextValue};
    ///
    /// let mut ctx = ComputeContext::new(0.0, 0.01);
    /// ctx.insert(
    ///     ContextVariable::External { name: "inlet_flow".into() },
    ///     ContextValue::Scalar(1e-6),
    /// );
    /// ```
    pub fn insert(&mut self, key: ContextVariable, value: ContextValue) {
        self.variables.insert(key, value);
    }

    /// Returns a reference to the value for a key, if present.
    pub fn get(&self, key: &ContextVariable) -> Option<&ContextValue> {
        self.variables.get(key)
    }

    /// Returns the scalar value of an `External` key, if present and scalar.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::physics::{ComputeContext, ContextVariable, ContextValue};
    ///
    /// let mut ctx = ComputeContext::new(0.0, 0.01);
    /// ctx.insert(
    ///     ContextVariable::External { name: "pressure".into() },
    ///     ContextValue::Scalar(101325.0),
    /// );
    /// assert!((ctx.external_scalar("pressure").unwrap() - 101325.0).abs() < 1e-6);
    /// ```
    pub fn external_scalar(&self, name: &str) -> Option<f64> {
        let key = ContextVariable::External {
            name: Cow::Owned(name.to_string()),
        };
        match self.variables.get(&key) {
            Some(ContextValue::Scalar(v)) => Some(*v),
            _ => None,
        }
    }

    /// Returns the scalar field for a spatial gradient variable, if present.
    pub fn spatial_gradient(
        &self,
        dimension: usize,
        component: Option<usize>,
    ) -> Option<&DVector<f64>> {
        let key = ContextVariable::SpatialGradient {
            dimension,
            component,
        };
        match self.variables.get(&key) {
            Some(ContextValue::ScalarField(v)) => Some(v),
            _ => None,
        }
    }

    /// Returns `true` if no optional variables are stored.
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_context_new() {
        let ctx = ComputeContext::new(10.0, 0.01);
        assert!((ctx.time() - 10.0).abs() < 1e-12);
        assert!((ctx.time_step() - 0.01).abs() < 1e-12);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_context_insert_and_get_scalar() {
        let mut ctx = ComputeContext::new(0.0, 0.01);
        ctx.insert(
            ContextVariable::External {
                name: "pressure".into(),
            },
            ContextValue::Scalar(101325.0),
        );
        assert!(!ctx.is_empty());
        match ctx.get(&ContextVariable::External {
            name: "pressure".into(),
        }) {
            Some(ContextValue::Scalar(v)) => assert!((v - 101325.0).abs() < 1e-6),
            _ => panic!("Expected Scalar"),
        }
    }

    #[test]
    fn test_external_scalar_accessor() {
        let mut ctx = ComputeContext::new(0.0, 0.01);
        ctx.insert(
            ContextVariable::External {
                name: "flow".into(),
            },
            ContextValue::Scalar(1e-6),
        );
        assert!((ctx.external_scalar("flow").unwrap() - 1e-6).abs() < 1e-18);
        assert!(ctx.external_scalar("unknown").is_none());
    }

    #[test]
    fn test_spatial_gradient_accessor() {
        let mut ctx = ComputeContext::new(0.0, 0.01);
        let field = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        ctx.insert(
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            ContextValue::ScalarField(field.clone()),
        );
        let retrieved = ctx.spatial_gradient(0, None).unwrap();
        assert_eq!(retrieved, &field);
        assert!(ctx.spatial_gradient(1, None).is_none());
    }

    #[test]
    fn test_context_variable_hash_eq() {
        let a = ContextVariable::External {
            name: "pressure".into(),
        };
        let b = ContextVariable::External {
            name: "pressure".into(),
        };
        let c = ContextVariable::External {
            name: "flow".into(),
        };
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Usable as HashMap key
        let mut map = HashMap::new();
        map.insert(a.clone(), 1u32);
        assert_eq!(map.get(&b), Some(&1u32));
        assert_eq!(map.get(&c), None);
    }

    #[test]
    fn test_context_variable_spatial_gradient_eq() {
        let a = ContextVariable::SpatialGradient {
            dimension: 0,
            component: Some(1),
        };
        let b = ContextVariable::SpatialGradient {
            dimension: 0,
            component: Some(1),
        };
        let c = ContextVariable::SpatialGradient {
            dimension: 0,
            component: None,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_context_value_boolean() {
        let mut ctx = ComputeContext::new(0.0, 0.01);
        ctx.insert(ContextVariable::Time, ContextValue::Boolean(true));
        match ctx.get(&ContextVariable::Time) {
            Some(ContextValue::Boolean(v)) => assert!(*v),
            _ => panic!("Expected Boolean"),
        }
    }
}
