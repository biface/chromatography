//! Multi-species Langmuir model with competitive adsorption
//!
//! # Physical background
//!
//! In liquid chromatography, several chemical species (solutes) coexist in the
//! column and **compete for the same adsorption sites** on the stationary phase.
//! This competition effect is the core of the multi-species Langmuir model:
//! the presence of one species reduces the adsorption capacity available to all others.
//!
//! # Model equations
//!
//! ## Competitive Langmuir isotherm
//!
//! For species $i$ in a mixture of $n$ species:
//!
//! $$\bar{C}_i = \lambda_i \cdot C_i + \frac{\bar{N}_i \cdot \tilde{K}_i \cdot C_i}{1 + \sum_j \tilde{K}_j \cdot C_j}$$
//!
//! where $\bar{N}_i = (1 - \varepsilon) \cdot N_i$ is the effective site count.
//!
//! ## Jacobian matrix
//!
//! The partial derivatives $M_{ij} = \partial \bar{C}_i / \partial C_j$ form an $n \times n$ matrix:
//!
//! **Diagonal terms** ($i = j$):
//! $$M_{ii} = \lambda_i + \frac{\bar{N}_i \cdot \tilde{K}_i \cdot (1 + \sum_k{\tilde{K}_k \cdot C_k}  - \tilde{K}_i C_i)}{(1 + \sum_k{\tilde{K}_k \cdot C_k})^2}$$
//!
//! **Off-diagonal terms** ($i \neq j$):
//! $$M_{ij} = -\frac{\bar{N}_i \cdot \tilde{K}_i \cdot \tilde{K}_j \cdot C_i}{(1 + \sum_k{\tilde{K}_k \cdot C_k})^2}$$
//!
//! ## Transport equation
//!
//! $$\frac{\partial C}{\partial t} = -(I + F_e \cdot M)^{-1} \cdot u_e \cdot \frac{\partial C}{\partial z}$$
//!
//! with $F_e = (1-\varepsilon)/\varepsilon$ and $u_e = u/\varepsilon$.
//!
//! # Spatial discretisation
//!
//! Concentrations are stored in a `[n_points × n_species]` matrix:
//! - **Rows**: spatial points $z_0, z_1, \ldots, z_{N-1}$
//! - **Columns**: species $0, 1, \ldots, n-1$
//!
//! The spatial gradient uses an **upwind scheme** (backward difference),
//! unconditionally stable for left-to-right convection ($u_e > 0$).
//!
//! # Injection strategy
//!
//! Injection is modelled as a **temporal boundary condition** at the column inlet ($z = 0$).
//! The column starts empty at $t = 0$ and concentration enters at each time step via
//! `injection.evaluate(t)`, where `t` is read from the `PhysicalState` metadata.
//!
//! Each species carries its own [`TemporalInjection`] profile, evaluated independently
//! at every call to `compute_physics`. This mirrors the approach used in
//! [`crate::models::LangmuirSingle`] and works with any solver that writes `"time"` into
//! the state metadata before each step.
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::models::{LangmuirMulti, SpeciesParams};
//! use chrom_rs::models::injection::TemporalInjection;
//! use chrom_rs::physics::PhysicalModel;
//! use chrom_rs::solver::{Scenario, SolverConfiguration, DomainBoundaries};
//!
//! // Malic acid (low affinity — elutes first)
//! let malic = SpeciesParams::new("Malic", 1.0, 0.5, 1,
//!     TemporalInjection::dirac(0.0, 1e-3));
//!
//! let mut model = LangmuirMulti::new(vec![malic], 150, 0.4, 0.001, 0.25).unwrap();
//!
//! // Citric acid (higher affinity — elutes later)
//! let citric = SpeciesParams::new("Citric", 1.0, 2.0, 1,
//!     TemporalInjection::dirac(0.0, 1e-3));
//! model.add_species(citric).unwrap();
//!
//! let initial = model.setup_initial_state();
//! let boundaries = DomainBoundaries::temporal(initial);
//! let scenario = Scenario::new(Box::new(model), boundaries);
//! let config = SolverConfiguration::time_evolution(600.0, 6000);
//! ```

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;

use crate::models::injection::TemporalInjection;
use crate::physics::{
    ExportError, Exportable, PhysicalData, PhysicalModel, PhysicalQuantity, PhysicalState,
    outlet_data, sample_indices,
};

// =================================================================================================
// SpeciesParams — Physical parameters of a chemical species
// =================================================================================================

// =================================================================================================
// SpeciesParams — Physical parameters of a chemical species
// =================================================================================================

/// Physical parameters of a chemical species for the Langmuir model
///
/// # Properties
///
/// Each species is characterised by three parameters that govern its
/// chromatographic behaviour:
///
/// | Parameter    | Symbol       | Unit              | Effect                              |
/// |--------------|--------------|-------------------|-------------------------------------|
/// | `lambda`     | $\lambda$    | dimensionless     | Residual linear retention           |
/// | `langmuir_k` | $\tilde{K}$  | $\text{L/mol}$    | Affinity for the stationary phase   |
/// | `port_number`| $N$          | dimensionless     | Adsorption sites occupied           |
///
/// # Effective site count $\bar{N}$
///
/// The effective site count accounts for the stationary phase fraction:
/// $$\bar{N}_i = (1 - \varepsilon) \cdot N_i$$
///
/// This calculation is performed inside [`LangmuirMulti`] using `jacobian` private method.
/// using `stationary_fraction = 1 - ε` stored on the column struct.
///
/// # Example
///
/// ```
/// use chrom_rs::models::{SpeciesParams, injection::TemporalInjection};
///
/// let malic = SpeciesParams::new(
///     "Malic Acid",
///     1.0,   // λ [dimensionless]
///     0.5,   // K̃ [L/mol] — low affinity → fast elution
///     1,   // N [dimensionless]
///     TemporalInjection::dirac(0.0, 1e-3),
/// );
/// assert!(malic.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesParams {
    /// Species name (used for plot legends and CSV headers)
    pub name: String,

    /// Isotherm intercept $\lambda$ **\[dimensionless\]**, must be $\geq 0$
    ///
    /// Represents residual linear retention. Typical value: 1.0 (pure Henry).
    pub lambda: f64,

    /// Langmuir equilibrium constant $\tilde{K}$ **\[L/mol\]**, must be $> 0$
    ///
    /// Controls affinity for the stationary phase. Higher $\tilde{K}$ → stronger
    /// retention → later elution.
    pub langmuir_k: f64,

    /// Number of adsorption sites occupied $N$ **\[dimensionless\]**, must be $> 0$
    ///
    /// Adsorption stoichiometry on the stationary phase. The effective count
    /// $\bar{N} = (1-\varepsilon) \cdot N$ is computed inside the model.
    pub port_number: u32,

    /// Temporal injection profile at column inlet ($z = 0$)
    pub injection: TemporalInjection,
}

impl SpeciesParams {
    /// Creates a new species parameter set
    ///
    /// # Arguments
    ///
    /// * `name`        — Species identifier (e.g. `"Malic Acid"`)
    /// * `lambda`      — $\lambda \geq 0$ **\[dimensionless\]**
    /// * `langmuir_k`  — $\tilde{K} > 0$ **\[L/mol\]**
    /// * `port_number` — $N > 0$ **\[dimensionless\]**
    /// * `injection`   — Injection profile at $z = 0$
    ///
    /// This constructor does not validate. Call [`validate`](Self::validate) to
    /// check physical constraints before passing to [`LangmuirMulti`].
    pub fn new(
        name: impl Into<String>,
        lambda: f64,
        langmuir_k: f64,
        port_number: u32,
        injection: TemporalInjection,
    ) -> Self {
        Self {
            name: name.into(),
            lambda,
            langmuir_k,
            port_number,
            injection,
        }
    }

    /// Validates the physical constraints of the parameters
    ///
    /// # Rules
    ///
    /// - $\lambda \geq 0$: linear retention cannot be negative
    /// - $\tilde{K} > 0$: thermodynamic equilibrium constant is strictly positive
    /// - $N > 0$: at least one adsorption site must be occupied
    ///
    /// # Returns
    ///
    /// - `Ok(())` if all parameters are physically valid
    /// - `Err(String)` with a descriptive message otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use chrom_rs::models::{SpeciesParams, injection::TemporalInjection};
    ///
    /// // Invalid: K̃ = 0 has no physical meaning
    /// let bad = SpeciesParams::new("Bad", 1.0, 0.0, 1,
    ///     TemporalInjection::dirac(0.0, 0.1));
    /// assert!(bad.validate().is_err());
    ///
    /// // Valid: λ = 0 means pure Langmuir isotherm (no linear term)
    /// let ok = SpeciesParams::new("Ok", 0.0, 0.5, 1,
    ///     TemporalInjection::dirac(0.0, 0.1));
    /// assert!(ok.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        // λ < 0 is physically impossible (negative linear retention)
        if self.lambda < 0.0 {
            return Err(format!(
                "Species '{}' : lambda must be >= 0, got {}",
                self.name, self.lambda
            ));
        }
        // K̃ = 0 would produce division by zero in the Jacobian denominator
        if self.langmuir_k <= 0.0 {
            return Err(format!(
                "Species '{}' : Langmuir K̃ must be > 0, got {}",
                self.name, self.langmuir_k
            ));
        }
        // N = 0 would nullify all competitive adsorption terms
        if self.port_number == 0 {
            return Err(format!(
                "Species '{}' : Port number must be > 0 (competitiveness), got {}",
                self.name, self.port_number
            ));
        }
        Ok(())
    }
}

// =================================================================================================
// LangmuirMulti — Complete physical model
// =================================================================================================

/// Multi-species Langmuir model with competitive adsorption
///
/// # Architecture
///
/// The model separates **column** properties (fixed at construction) from
/// **species** properties (extensible via [`add_species`](Self::add_species)).
/// Derived quantities ($F_e$, $u_e$, $\Delta z$) are precomputed at construction
/// to avoid repeated divisions inside the time-integration loop.
///
/// # State layout
///
/// The physical state is a `[n_points × n_species]` matrix:
///
/// ```text
///            species 0   species 1   species 2
/// z_0     [  C(0,0)      C(0,1)      C(0,2)  ]
/// z_1     [  C(1,0)      C(1,1)      C(1,2)  ]
///  ⋮
/// z_N-1   [  C(N-1,0)    C(N-1,1)    C(N-1,2)]
/// ```
///
/// The **column outlet** (chromatogram signal) is the last row: `C[n_points-1, :]`.
///
/// # Progressive species addition
///
/// ```rust
/// use chrom_rs::models::{TemporalInjection, SpeciesParams, LangmuirMulti};
///
/// let first_species = SpeciesParams::new("SP1", 1.0, 0.5, 1, TemporalInjection::dirac(0.0, 1e-3));
/// let second_species = SpeciesParams::new("SP2", 1.0, 0.6, 1, TemporalInjection::dirac(0.0, 1e-3));
/// let third_species = SpeciesParams::new("SP3", 1.0, 0.4, 1, TemporalInjection::dirac(0.0, 1e-3));
/// let fourth_species = SpeciesParams::new("SP4", 1.0, 0.3, 1, TemporalInjection::dirac(0.0, 1e-3));
///
/// let mut model = LangmuirMulti::new(vec![first_species, second_species], 150, 0.4, 0.001, 0.25).unwrap();
/// model.add_species(third_species).unwrap();   // Jacobian: 2×2 → 3×3
/// model.add_species(fourth_species).unwrap();    // Jacobian: 3×3 → 4×4
/// ```
#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct LangmuirMulti {
    /// Chemical species (at least 1, extensible via \[`add_species`\](Self::add_species))
    species: Vec<SpeciesParams>,

    /// Number of spatial discretisation points $N_z$
    n_points: usize,

    /// Extra-granular porosity $\varepsilon$ **\[dimensionless\]** ∈ (0, 1)
    porosity: f64,

    /// Superficial velocity $u$ **\[m/s\]**
    velocity: f64,

    /// Column length $L$ **\[m\]**
    column_length: f64,

    /// Spatial step $\Delta z = L / (N_z - 1)$ **\[m\]** — precomputed
    dz: f64,

    /// Phase ratio $F_e = (1-\varepsilon)/\varepsilon$ **\[dimensionless\]** — precomputed
    fe: f64,

    /// Interstitial velocity $u_e = u/\varepsilon$ **\[m/s\]** — precomputed
    ue: f64,

    /// Stationary phase fraction $(1-\varepsilon)$ **\[dimensionless\]** — precomputed
    ///
    /// Used for the effective site count: $\bar{N}_i = (1-\varepsilon) \cdot N_i$.
    stationary_fraction: f64,
}

impl LangmuirMulti {
    /// Creates a multi-species Langmuir model
    ///
    /// # Arguments
    ///
    /// * `species`        — List of chemical species (at least one)
    /// * `n_points`       — Number of spatial points $N_z \geq 2$
    /// * `porosity`       — $\varepsilon \in (0, 1)$ **\[dimensionless\]**
    /// * `velocity`       — Superficial velocity $u > 0$ **\[m/s\]**
    /// * `column_length`  — Column length $L > 0$ **\[m\]**
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if any parameter fails validation.
    ///
    /// # Example
    ///
    /// ```
    /// use chrom_rs::models::{LangmuirMulti, SpeciesParams, injection::TemporalInjection};
    ///
    /// let tfa = SpeciesParams::new("TFA", 1.2, 0.4, 2,
    ///     TemporalInjection::dirac(0.0, 1e-3));
    /// assert!(LangmuirMulti::new(vec![tfa], 100, 0.4, 0.001, 0.25).is_ok());
    /// ```
    pub fn new(
        species: Vec<SpeciesParams>,
        n_points: usize,
        porosity: f64,
        velocity: f64,
        column_length: f64,
    ) -> Result<Self, String> {
        // At least, one specie...
        if species.is_empty() {
            return Err("Langmuir multi species must have at least one species.".to_string());
        }

        // validate each specie...

        for sp in &species {
            sp.validate()?;
        }

        // no double in species...

        let mut seen = std::collections::HashSet::new();
        for sp in &species {
            if !seen.insert(&sp.name) {
                return Err(format!(
                    "Duplicate speciees name '{}' in intial species list",
                    &sp.name
                ));
            }
        }

        if n_points < 2 {
            return Err(format!(
                "number of points must have at least two points, got {n_points}."
            ));
        }

        if porosity <= 0.0 || porosity >= 1.0 {
            return Err(format!("porosity must be in ]0, 1[, got {porosity}."));
        }

        if velocity <= 0.0 {
            return Err(format!(
                "velocity must be strictly positive, got {velocity}."
            ));
        }

        if column_length <= 0.0 {
            return Err(format!(
                "column length must be strictly positive, got {column_length}."
            ));
        }

        let dz = column_length / n_points as f64;
        let fe = (1.0 - porosity) / porosity;
        let ue = velocity / porosity;
        let stationary_fraction = 1.0 - porosity;

        Ok(Self {
            species,
            n_points,
            porosity,
            velocity,
            column_length,
            dz,
            fe,
            ue,
            stationary_fraction,
        })
    }

    /// Adds a competing chemical species to the model
    ///
    /// # Errors
    ///
    /// - Species parameters are invalid
    /// - A species with the same name already exists
    ///
    /// # Example
    ///
    /// ```
    /// use chrom_rs::models::{LangmuirMulti, SpeciesParams, injection::TemporalInjection};
    ///
    /// let inj = TemporalInjection::dirac(0.0, 1e-3);
    /// let sp1 = SpeciesParams::new("A", 1.0, 0.5, 1, inj.clone());
    /// let sp2 = SpeciesParams::new("B", 1.0, 2.0, 1, inj);
    ///
    /// let mut model = LangmuirMulti::new(vec![sp1], 50, 0.4, 0.001, 0.25).unwrap();
    /// model.add_species(sp2).unwrap();
    /// assert_eq!(model.n_species(), 2);
    /// ```
    pub fn add_species(&mut self, species: SpeciesParams) -> Result<(), String> {
        // validate the specie
        species.validate()?;

        // Build the set of existing names on the fly

        let names: std::collections::HashSet<&str> =
            self.species.iter().map(|s| s.name.as_str()).collect();

        if names.contains(species.name.as_str()) {
            return Err(format!(
                "Species '{}' already exists in this model",
                species.name
            ));
        }

        self.species.push(species);
        Ok(())
    }

    /// Applies the same injection profile to **all** species.
    ///
    /// Used by the config loader when `scenario.yml` defines only a
    /// `default_injection` with no per-species override.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::models::{LangmuirMulti, SpeciesParams, TemporalInjection};
    ///
    /// let sp = SpeciesParams::new("A", 1.0, 0.5, 1, TemporalInjection::none());
    /// let mut model = LangmuirMulti::new(vec![sp], 100, 0.4, 0.001, 0.25).unwrap();
    ///
    /// model.set_injection_all(TemporalInjection::dirac(5.0, 0.1));
    /// ```
    pub fn set_injection_all(&mut self, injection: TemporalInjection) {
        for sp in &mut self.species {
            sp.injection = injection.clone();
        }
    }

    /// Replaces the injection profile for a single species identified by name.
    ///
    /// Used by the config loader when `scenario.yml` lists per-species overrides
    /// in the `injections` array. Species not listed keep the profile
    /// set by [`set_injection_all`](Self::set_injection_all).
    ///
    /// # Errors
    ///
    /// Returns `Err` if no species with the given name exists in the model.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::models::{LangmuirMulti, SpeciesParams, TemporalInjection};
    ///
    /// let sp_a = SpeciesParams::new("A", 1.0, 0.5, 1, TemporalInjection::none());
    /// let sp_b = SpeciesParams::new("B", 1.0, 2.0, 1, TemporalInjection::none());
    /// let mut model = LangmuirMulti::new(vec![sp_a, sp_b], 100, 0.4, 0.001, 0.25).unwrap();
    ///
    /// model.set_injection_all(TemporalInjection::dirac(5.0, 0.1));
    /// model.set_injection_for("B", TemporalInjection::gaussian(10.0, 3.0, 0.05)).unwrap();
    /// ```
    pub fn set_injection_for(
        &mut self,
        species: &str,
        injection: TemporalInjection,
    ) -> Result<(), String> {
        match self.species.iter_mut().find(|sp| sp.name == species) {
            Some(sp) => {
                sp.injection = injection;
                Ok(())
            }
            None => Err(format!("Species '{species}' not found in model")),
        }
    }

    /// Returns the current number of species in the model
    /// Returns the current number of chemical species in the model.
    ///
    /// Grows by one on each successful call to [`add_species`](Self::add_species).
    pub fn n_species(&self) -> usize {
        self.species.len()
    }

    /// Returns the extra-granular porosity $\varepsilon$ **\[dimensionless\]**.
    ///
    /// Defined at construction; constant over the simulation lifetime.
    /// Used to derive $F_e = (1-\varepsilon)/\varepsilon$ and $u_e = u/\varepsilon$.
    pub fn porosity(&self) -> f64 {
        self.porosity
    }

    /// Returns the superficial velocity $u$ **\[m/s\]**.
    ///
    /// The interstitial velocity $u_e = u / \varepsilon$ is precomputed and stored
    /// separately; this accessor returns the raw constructor argument.
    pub fn velocity(&self) -> f64 {
        self.velocity
    }

    /// Returns the column length $L$ **\[m\]**.
    ///
    /// The spatial step $\Delta z = L / N_z$ is precomputed at construction.
    pub fn column_length(&self) -> f64 {
        self.column_length
    }

    /// Returns the number of spatial discretisation points $N_z$.
    ///
    /// Matches the number of rows in the concentration state matrix
    /// `[n_points × n_species]`. Must be $\geq 2$ (enforced by the constructor).
    pub fn spatial_points(&self) -> usize {
        self.n_points
    }

    /// Returns all species parameter sets in insertion order.
    ///
    /// The slice index matches the column index of the state matrix
    /// `[n_points × n_species]` — same order as [`species_names`](Self::species_names).
    /// Use this accessor when full parameter detail is needed; prefer
    /// [`species_names`](Self::species_names) for display purposes only.
    pub fn species_params(&self) -> &[SpeciesParams] {
        &self.species
    }

    /// Returns species names in insertion order
    ///
    /// The order matches column indices of the state matrix `[n_points × n_species]`.
    pub fn species_names(&self) -> Vec<&str> {
        self.species.iter().map(|s| s.name.as_str()).collect()
    }

    /// Computes the $n \times n$ Jacobian matrix $M$
    ///
    /// $M_{ij} = \partial \bar{C}_i / \partial C_j$
    ///
    /// Let $D = 1 + \sum_k \tilde{K}_k C_k$.
    ///
    /// **Diagonal** ($i = j$):
    /// $$M_{ii} = \lambda_i + \frac{\bar{N}_i \tilde{K}_i (D - \tilde{K}_i C_i)}{D^2}$$
    ///
    /// **Off-diagonal** ($i \neq j$, always $\leq 0$):
    /// $$M_{ij} = -\frac{\bar{N}_i \tilde{K}_i \tilde{K}_j C_i}{D^2}$$
    ///
    /// $\Sigma_{KC}$ and $D^2$ are computed once before the double loop: $O(n^2)$.
    ///
    /// # Panics
    ///
    /// Debug-panics if `c.len() != self.n_species()`.
    pub(crate) fn jacobian(&self, c: &[f64]) -> DMatrix<f64> {
        let n = self.n_species();
        debug_assert_eq!(
            n,
            self.species.len(),
            "Jacobian: c.len()={} != n_species={}",
            c.len(),
            n
        );

        // Σ_KC — computed once, shared by all matrix elements
        let sum_kc: f64 = self
            .species
            .iter()
            .zip(c.iter())
            .map(|(sp, &ck)| sp.langmuir_k * ck)
            .sum();

        let denom = 1.0 + sum_kc;
        let denom_exp2 = denom * denom;

        let mut jacobian = DMatrix::zeros(n, n);

        for i in 0..n {
            let sp_i = &self.species[i];
            let n_bar_i = self.stationary_fraction * sp_i.port_number as f64;

            for j in 0..n {
                jacobian[(i, j)] = if i == j {
                    // Diagonal: ∂C̄_i/∂C_i
                    let num = n_bar_i * sp_i.langmuir_k * (denom - sp_i.langmuir_k * c[i]);
                    sp_i.lambda + num / denom_exp2
                } else {
                    // Off-diagonal: ∂C̄_i/∂C_j  (always ≤ 0 — competitive inhibition)
                    let num = -n_bar_i * sp_i.langmuir_k * self.species[j].langmuir_k * c[i];
                    num / denom_exp2
                }
            }
        }

        jacobian
    }

    /// Computes $(I + F_e \cdot M)^{-1}$
    ///
    /// Returns `Err` if the matrix is singular (physically pathological).
    /// The caller falls back to the identity matrix in that case.
    pub(crate) fn inverse_propagation(&self, c: &[f64]) -> Result<DMatrix<f64>, String> {
        let n = self.n_species();
        let jacobian = self.jacobian(c);
        let mat = DMatrix::identity(n, n) + self.fe * jacobian;

        mat.try_inverse().ok_or_else(|| {
            format!(
                "Matrix (I + Fe * J) is singular at concentration {:?}. \
            Please check langmuir K and port number values",
                c
            )
        })
    }
}

// =================================================================================================
// PhysicalModel implementation
// =================================================================================================

#[typetag::serde]
impl PhysicalModel for LangmuirMulti {
    /// Returns the number of spatial discretization points (row dimension of the state matrix)
    /// Returns the total number of state variables — `n_points × n_species`.
    ///
    /// This is the flattened size of the concentration matrix, used by the
    /// solver to determine step sizes and parallelism thresholds.
    fn points(&self) -> usize {
        self.n_points
    }

    /// Computes $\partial C / \partial t$ for all spatial points and all species
    ///
    /// For each spatial point $i$:
    /// 1. Extract concentration vector $\mathbf{c}_i$
    /// 2. Compute $(I + F_e M)^{-1}$
    /// 3. Compute upwind gradient $\partial C_k / \partial z$
    /// 4. Apply $\dot{\mathbf{C}}_i = -u_e (I + F_e M)^{-1} \nabla_z \mathbf{C}_i$
    ///
    /// **Left boundary** ($i = 0$): closed inlet after the initial impulse — upstream
    /// concentration is zero. v0.2.0 will replace this with `injection.evaluate(t)`.
    ///
    /// **Singular matrix fallback**: identity matrix + warning log. The solver will
    /// catch any NaN/Inf in the resulting state.
    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
        let n_species = self.n_species();

        // ── Time ──────────────────────────────────────────────────────────────────
        //
        // The solver writes the current simulation time into the state metadata
        // before each call. If absent (e.g. first call, or solver does not support
        // metadata), we default to t=0.0 — injection profiles will return their
        // value at t=0.
        let t = state.get_metadata("time").unwrap_or(0.0);

        // ── State extraction ──────────────────────────────────────────────────────
        //
        // The concentration matrix has shape [n_points × n_species]:
        //   rows    → spatial points z_0, z_1, ..., z_{N-1}
        //   columns → species 0, 1, ..., n-1
        //
        // clone_owned() is required because compute_row borrows c_matrix immutably
        // while dc_dt is being built — both coexist during the computation.
        let c_matrix = state
            .get(PhysicalQuantity::Concentration)
            .expect("LangmuirMulti::compute_physics: state must contain Concentration")
            .as_matrix()
            .clone_owned();

        debug_assert_eq!(
            (c_matrix.nrows(), c_matrix.ncols()),
            (self.n_points, n_species),
            "State matrix shape [{}, {}] expected, got [{}, {}]",
            self.n_points,
            n_species,
            c_matrix.nrows(),
            c_matrix.ncols()
        );

        // ── Row kernel ────────────────────────────────────────────────────────────
        //
        // compute_row(i) encapsulates the full physics computation for a single
        // spatial point i. It returns dC/dt for all species at that point as a
        // Vec<f64> of length n_species.
        //
        // Separating the kernel from the loop allows the same physics code to be
        // called from both the sequential and parallel paths without duplication
        // (WHAT to compute vs HOW to iterate — same principle as Scenario/Solver).
        //
        // Thread safety: compute_row reads c_matrix[(i,k)] and c_matrix[(i-1,k)]
        // but never writes to c_matrix. All writes go to dc_dt at row i only.
        // Since no two iterations share the same output row, rayon can run them
        // concurrently without data races.

        let compute_row = |i: usize| -> Vec<f64> {
            // Step 1 — Extract concentration vector at point i
            //
            // We collect into Vec<f64> to pass to inverse_propagation, which
            // expects a slice. This is a small allocation (n_species elements,
            // typically 2-5) — negligible cost.

            let c_at_i: Vec<f64> = (0..n_species).map(|k| c_matrix[(i, k)]).collect();

            // Step 2 — Compute the inverse propagation matrix (I + Fe·M)^{-1}
            //
            // M is the Jacobian ∂C̄/∂C evaluated at c_at_i (see jacobian()).
            // The matrix (I + Fe·M) captures how adsorption slows down the
            // propagation of each species relative to pure convection.
            //
            // If the matrix is singular (physically pathological concentrations),
            // we fall back to the identity matrix — equivalent to ignoring
            // adsorption at this point for this time step. The solver will detect
            // any downstream instability via NaN/Inf checks.

            let inv = match self.inverse_propagation(&c_at_i) {
                Ok(m_inv) => m_inv,
                Err(e) => {
                    log::warn!("LangmuirMulti at point {i}: {e}. Using identity fallback.");
                    DMatrix::identity(n_species, n_species)
                }
            };

            // Step 3 — Compute the upwind spatial gradient ∂C_k/∂z
            //
            // Upwind (backward difference) scheme: uses the value at i-1 as the
            // upstream neighbor. This is unconditionally stable for left-to-right
            // convection (u_e > 0) and avoids spurious oscillations near sharp
            // concentration fronts.
            //
            // Finite volume convention: Δz = L / n_points (cell width).
            // Each cell i spans [i·Δz, (i+1)·Δz].
            //
            // At the left boundary (i=0) there is no physical neighbor to the
            // left — the upstream value is provided by the temporal injection
            // profile. Each species k has its own profile evaluated at time t.
            // This is the sole entry point of material into the column.

            let mut grad = DVector::zeros(n_species);

            for k in 0..n_species {
                let c_cur = c_matrix[(i, k)];
                grad[k] = if i > 0 {
                    // Interior point and right boundary: real upstream neighbor
                    (c_cur - c_matrix[(i - 1, k)]) / self.dz
                } else {
                    // Left boundary: fictitious upstream cell = injection profile
                    let c_upstream = self.species[k].injection.evaluate(t);
                    (c_cur - c_upstream) / self.dz
                };
            }

            // Step 4 — Apply the transport equation
            //
            // dC/dt = -u_e · (I + Fe·M)^{-1} · ∂C/∂z
            //
            // The matrix-vector product (I + Fe·M)^{-1} · grad couples all
            // species together: the adsorption competition means the velocity
            // of species i depends on the concentrations of all other species j.
            //
            // The result is collected as Vec<f64> — the caller assembles rows
            // into the output matrix dc_dt.

            let dv: DVector<f64> = inv * grad;
            (0..n_species).map(|k| -self.ue * dv[k]).collect()
        };

        // ── Assembly ──────────────────────────────────────────────────────────────
        //
        // Call compute_row for every spatial point and write the results into
        // dc_dt[i, k].
        //
        // Two paths share the same kernel:
        //   - Sequential: straightforward loop, always available.
        //   - Parallel:   rayon par_iter, enabled by the `parallel` feature flag
        //                 and only when n_points exceeds parallel_threshold().
        //                 Below the threshold, thread overhead exceeds the gain.

        let mut dc_dt = DMatrix::zeros(self.n_points, n_species);

        if (self.n_points * self.n_species()) > crate::solver::parallel_threshold() {
            // Above threshold — use parallel iteration if feature is enabled.
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                // Each call to compute_row(i) is independent: reads from c_matrix
                // (shared immutable reference) and produces a Vec<f64> for row i.
                // collect() gathers all rows in index order before assembly.
                let rows: Vec<Vec<f64>> =
                    (0..n_species).into_par_iter().map(&compute_row).collect();

                for (i, row) in rows.into_iter().enumerate() {
                    for k in 0..n_species {
                        dc_dt[(i, k)] = row[k];
                    }
                }
            }
            // parallel feature not compiled in — fall through to sequential.
            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..self.n_points {
                    let row = compute_row(i);
                    for k in 0..n_species {
                        dc_dt[(i, k)] = row[k];
                    }
                }
            }
        } else {
            // Below threshold — sequential is faster than spawning threads.
            for i in 0..self.n_points {
                let row = compute_row(i);
                for k in 0..n_species {
                    dc_dt[(i, k)] = row[k];
                }
            }
        }

        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::from_matrix(dc_dt),
        )
    }

    /// Initialises the column state to zero (empty column)
    ///
    /// # Initial state
    ///
    /// The column is empty at $t = 0$. Concentration enters from the left
    /// boundary at each time step via `injection.evaluate(t)` in `compute_physics`.
    ///
    /// This mirrors the `LangmuirSingle` approach: the temporal injection profile
    /// is the sole source of material, read from `PhysicalState` metadata at
    /// each call to `compute_physics`.
    fn setup_initial_state(&self) -> PhysicalState {
        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::from_matrix(DMatrix::zeros(self.n_points, self.n_species())),
        )
    }

    /// Returns the human-readable model identifier.
    ///
    /// Used as the `"model"` key in the JSON export map and in
    /// simulation result metadata.
    fn name(&self) -> &str {
        "Langmuir Multi-Species"
    }

    /// Returns a brief description of the model's physical assumptions.
    ///
    /// Documents the competitive Langmuir isotherm, the upwind spatial scheme,
    /// and the matrix Jacobian inversion at the heart of the multi-species
    /// transport computation.
    fn description(&self) -> Option<&str> {
        Some(
            "Competitive Langmuir adsorption model for n species \
             with upwind spatial discretisation and matrix Jacobian inversion.",
        )
    }

    fn set_injections(
        &mut self,
        injections: &HashMap<Option<String>, TemporalInjection>,
    ) -> Result<(), String> {
        // Step 1 — apply the default injection to all species.
        // This establishes the baseline before per-species overrides.
        let default_key: Option<String> = None;
        if let Some(inj) = injections.get(&default_key) {
            self.set_injection_all(inj.clone());
        }

        // Step 2 — apply per-species overrides.
        // Order within the HashMap is not guaranteed, but each named species
        // is independent, so ordering does not matter.
        for (key, inj) in injections {
            if let Some(name) = key {
                self.set_injection_for(name, inj.clone())?;
            }
        }
        Ok(())
    }
}

impl Exportable for LangmuirMulti {
    /// Builds the export map for a multi-species Langmuir simulation.
    ///
    /// Each species is serialised as a named block under `"profiles"`.
    /// The `"name"` key is colocated with the species data so that identity
    /// and measurements travel together — preventing species permutation on
    /// reload.
    ///
    /// # JSON layout
    ///
    /// ```json
    /// {
    ///   "metadata": {
    ///     "model":     "Langmuir multi species with temporal injection",
    ///     "n_species": 2,
    ///     "nz":        150,
    ///     "porosity":  0.4,
    ///     "velocity":  0.001,
    ///     "length":    0.25,
    ///     "species": [
    ///       { "name": "Malic",  "lambda": 1.0, "langmuir_k": 0.5, "port_number": 1 },
    ///       { "name": "Citric", "lambda": 1.0, "langmuir_k": 2.0, "port_number": 1 }
    ///     ]
    ///   },
    ///   "data": {
    ///     "time_points": [0.0, 0.1, "..."],
    ///     "profiles": {
    ///       "species_0": { "name": "Malic",  "Concentration": ["..."] },
    ///       "species_1": { "name": "Citric", "Concentration": ["..."] },
    ///       "global":    {}
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// The `"global"` block is included but empty by default — it is the
    /// extension point for future scalar or vector quantities (temperature,
    /// pressure, etc.) that are not species-specific.
    ///
    /// # Dispatch rule
    ///
    /// `PhysicalData::Matrix [n_points × n_species]` → `"species_N"` blocks.
    /// `PhysicalData::Scalar` or `Vector [n_points]`  → `"global"` block.
    fn to_map(
        &self,
        time_points: &[f64],
        trajectory: &[crate::physics::PhysicalState],
        metadata: &HashMap<String, String>,
    ) -> Map<String, Value> {
        let mut root = Map::new();

        // ── metadata ──────────────────────────────────────────────────────────
        let species_meta: Vec<Value> = self
            .species_params()
            .iter()
            .map(|s| {
                json!({
                    "name":        s.name,
                    "lambda":      s.lambda,
                    "langmuir_k":  s.langmuir_k,
                    "port_number": s.port_number,
                })
            })
            .collect();

        let mut meta: Map<String, Value> = [
            ("model", Value::String(self.name().into())),
            ("n_species", Value::from(self.n_species() as u64)),
            ("nz", Value::from(self.spatial_points() as u64)),
            ("porosity", Value::from(self.porosity())),
            ("velocity", Value::from(self.velocity())),
            ("length", Value::from(self.column_length())),
            ("species", Value::Array(species_meta)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        // Merge solver metadata — model keys take precedence
        for (k, v) in metadata {
            meta.entry(k.clone())
                .or_insert_with(|| Value::String(v.clone()));
        }

        root.insert("metadata".into(), Value::Object(meta));

        // ── data ──────────────────────────────────────────────────────────────
        let indices = sample_indices(time_points.len(), None);

        let tp_values: Vec<Value> = indices
            .iter()
            .map(|&i| Value::from(time_points[i]))
            .collect();

        let names = self.species_names();
        let n = names.len();

        // Build one concentration Vec per species across all sampled steps
        let mut per_species: Vec<Vec<Value>> = vec![Vec::with_capacity(indices.len()); n];
        for &i in &indices {
            let outlet = outlet_data(PhysicalQuantity::Concentration, trajectory, i);
            for (s, col) in per_species.iter_mut().enumerate() {
                col.push(Value::from(*outlet.get(s).unwrap_or(&0.0)));
            }
        }

        let mut profiles = Map::new();
        for (s, conc_vals) in per_species.into_iter().enumerate() {
            let mut block = Map::new();
            block.insert("name".into(), Value::String(names[s].to_string()));
            block.insert("Concentration".into(), Value::Array(conc_vals));
            profiles.insert(format!("species_{s}"), Value::Object(block));
        }
        // Extension point for future scalar/vector quantities
        profiles.insert("global".into(), Value::Object(Map::new()));

        root.insert(
            "data".into(),
            json!({
                "time_points": tp_values,
                "profiles":    profiles,
            }),
        );

        root
    }

    /// Reconstructs a `LangmuirMulti` configuration from an export map.
    ///
    /// # Required keys in `metadata`
    ///
    /// | Key | Type | Description |
    /// |---|---|---|
    /// | `nz` | `u64` | Number of spatial points |
    /// | `porosity` | `f64` | Extra-granular porosity ε ∈ (0, 1) |
    /// | `velocity` | `f64` | Superficial velocity (m/s) |
    /// | `length` | `f64` | Column length (m) |
    /// | `species` | array | Per-species parameters (see below) |
    ///
    /// Each entry in `metadata.species` must contain:
    /// `name` (string), `lambda` (f64), `langmuir_k` (f64), `port_number` (u64).
    ///
    /// The injection profile is not serialised. It defaults to
    /// [`TemporalInjection::none`] for each species and must be overridden
    /// by the caller after reconstruction if needed.
    ///
    /// # Errors
    ///
    /// Returns [`ExportError::MissingKey`] if a required key is absent,
    /// [`ExportError::InvalidValue`] if a value has the wrong type or model
    /// construction fails, or [`ExportError::SpeciesCountMismatch`] if the
    /// species list is empty.
    fn from_map(map: Map<String, Value>) -> Result<Self, ExportError>
    where
        Self: Sized,
    {
        let meta = map
            .get("metadata")
            .and_then(|v| v.as_object())
            .ok_or_else(|| ExportError::MissingKey("metadata".into()))?;

        macro_rules! get_f64 {
            ($key:expr) => {
                meta.get($key)
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| ExportError::MissingKey($key.into()))?
            };
        }
        macro_rules! get_usize {
            ($key:expr) => {
                meta.get($key)
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .ok_or_else(|| ExportError::MissingKey($key.into()))?
            };
        }

        let nz = get_usize!("nz");
        let porosity = get_f64!("porosity");
        let velocity = get_f64!("velocity");
        let length = get_f64!("length");

        let species_arr = meta
            .get("species")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ExportError::MissingKey("metadata.species".into()))?;

        if species_arr.is_empty() {
            return Err(ExportError::SpeciesCountMismatch {
                expected: 1,
                got: 0,
            });
        }

        let mut params: Vec<SpeciesParams> = Vec::with_capacity(species_arr.len());
        for (i, s) in species_arr.iter().enumerate() {
            let obj = s.as_object().ok_or_else(|| ExportError::InvalidValue {
                key: format!("metadata.species[{i}]"),
                reason: "expected object".into(),
            })?;

            macro_rules! field_f64 {
                ($key:expr) => {
                    obj.get($key).and_then(|v| v.as_f64()).ok_or_else(|| {
                        ExportError::MissingKey(format!("metadata.species[{i}].{}", $key))
                    })?
                };
            }
            macro_rules! field_u32 {
                ($key:expr) => {
                    obj.get($key)
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .ok_or_else(|| {
                            ExportError::MissingKey(format!("metadata.species[{i}].{}", $key))
                        })?
                };
            }

            let name = obj
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ExportError::MissingKey(format!("metadata.species[{i}].name")))?
                .to_string();

            params.push(SpeciesParams::new(
                &name,
                field_f64!("lambda"),
                field_f64!("langmuir_k"),
                field_u32!("port_number"),
                TemporalInjection::none(),
            ));
        }

        let first = params.remove(0);
        let mut model =
            LangmuirMulti::new(vec![first], nz, porosity, velocity, length).map_err(|e| {
                ExportError::InvalidValue {
                    key: "metadata".into(),
                    reason: e,
                }
            })?;

        for sp in params {
            model
                .add_species(sp)
                .map_err(|e| ExportError::InvalidValue {
                    key: "metadata.species".into(),
                    reason: e,
                })?;
        }

        Ok(model)
    }
}

// =================================================================================================
// Tests
// =================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── Shared helpers ────────────────────────────────────────────────────────

    fn inj() -> TemporalInjection {
        TemporalInjection::dirac(0.0, 1e-3) // t=0, amplitude 1 mmol/L
    }

    fn species(name: &str, lambda: f64, k: f64, n: u32) -> SpeciesParams {
        SpeciesParams::new(name, lambda, k, n, inj())
    }

    fn single_model() -> LangmuirMulti {
        LangmuirMulti::new(vec![species("A", 1.0, 0.5, 1)], 100, 0.4, 0.001, 0.25).unwrap()
    }

    fn two_species_model() -> LangmuirMulti {
        let mut m = single_model();
        m.add_species(species("B", 1.0, 2.0, 1)).unwrap();
        m
    }

    // ── SpeciesParams::validate ───────────────────────────────────────────────

    #[test]
    fn test_species_lambda_zero_is_valid() {
        assert!(species("X", 0.0, 1.0, 1).validate().is_ok());
    }

    #[test]
    fn test_species_lambda_negative_is_invalid() {
        let err = species("X", -0.1, 1.0, 1).validate().unwrap_err();
        assert!(err.contains("lambda") && err.contains('X'));
    }

    #[test]
    fn test_species_k_zero_is_invalid() {
        let err = species("X", 1.0, 0.0, 1).validate().unwrap_err();
        assert!(err.contains("langmuir_k") || err.contains("K̃"));
    }

    #[test]
    fn test_species_k_negative_is_invalid() {
        assert!(species("X", 1.0, -1.0, 1).validate().is_err());
    }

    #[test]
    fn test_species_port_number_zero_is_invalid() {
        let err = species("X", 1.0, 1.0, 0).validate().unwrap_err();
        assert!(err.contains("Port number") || err.contains('N'));
    }

    // ── LangmuirMulti::new ────────────────────────────────────────────────────

    #[test]
    fn test_new_valid_model() {
        let m = single_model();
        assert_eq!(m.n_species(), 1);
        assert_eq!(m.n_points, 100);
    }

    #[test]
    fn test_new_rejects_n_points_less_than_2() {
        assert!(
            LangmuirMulti::new(vec![species("A", 1.0, 0.5, 1)], 1, 0.4, 0.001, 0.25)
                .unwrap_err()
                .contains("number of points")
        );
    }

    #[test]
    fn test_new_rejects_porosity_zero() {
        assert!(
            LangmuirMulti::new(vec![species("A", 1.0, 0.5, 1)], 100, 0.0, 0.001, 0.25)
                .unwrap_err()
                .contains("porosity")
        );
    }

    #[test]
    fn test_new_rejects_porosity_one() {
        assert!(
            LangmuirMulti::new(vec![species("A", 1.0, 0.5, 1)], 100, 1.0, 0.001, 0.25)
                .unwrap_err()
                .contains("porosity")
        );
    }

    #[test]
    fn test_new_rejects_zero_velocity() {
        assert!(
            LangmuirMulti::new(vec![species("A", 1.0, 0.5, 1)], 100, 0.4, 0.0, 0.25)
                .unwrap_err()
                .contains("velocity")
        );
    }

    #[test]
    fn test_new_rejects_zero_length() {
        assert!(
            LangmuirMulti::new(vec![species("A", 1.0, 0.5, 1)], 100, 0.4, 0.001, 0.0)
                .unwrap_err()
                .contains("column length")
        );
    }

    #[test]
    fn test_new_precomputes_derived_quantities() {
        // ε = 0.4 → Fe = 0.6/0.4 = 1.5 ; ue = 0.001/0.4 = 0.0025 ; dz = 0.25/99
        let m = LangmuirMulti::new(vec![species("A", 1.0, 0.5, 1)], 100, 0.4, 0.001, 0.25).unwrap();
        assert_relative_eq!(m.fe, 1.5, epsilon = 1e-12);
        assert_relative_eq!(m.ue, 0.0025, epsilon = 1e-12);
        assert_relative_eq!(m.dz, 0.25 / 100.0, epsilon = 1e-12);
        assert_relative_eq!(m.stationary_fraction, 0.6, epsilon = 1e-12);
    }

    // ── add_species ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_species_increases_count() {
        let mut m = single_model();
        m.add_species(species("B", 1.0, 2.0, 1)).unwrap();
        assert_eq!(m.n_species(), 2);
        m.add_species(species("C", 1.0, 5.0, 1)).unwrap();
        assert_eq!(m.n_species(), 3);
    }

    #[test]
    fn test_add_species_rejects_duplicate_name() {
        let mut m = single_model();
        let err = m.add_species(species("A", 1.0, 2.0, 1)).unwrap_err();
        assert!(err.contains("already exists") && err.contains('A'));
    }

    #[test]
    fn test_add_species_rejects_invalid_params_and_leaves_model_unchanged() {
        let mut m = single_model();
        assert!(m.add_species(species("B", -1.0, 2.0, 1)).is_err());
        assert_eq!(m.n_species(), 1); // model must not be modified on error
    }

    #[test]
    fn test_species_names_order() {
        let mut m = single_model();
        m.add_species(species("B", 1.0, 2.0, 1)).unwrap();
        m.add_species(species("C", 1.0, 5.0, 1)).unwrap();
        assert_eq!(m.species_names(), vec!["A", "B", "C"]);
    }

    // ── compute_jacobian ──────────────────────────────────────────────────────

    #[test]
    fn test_jacobian_1x1_at_zero_concentration() {
        // At C=0: M₀₀ = λ + N̄·K̃ = 1.0 + (1-0.4)·1.0·0.5 = 1.3
        let jac = single_model().jacobian(&[0.0]);
        assert_eq!((jac.nrows(), jac.ncols()), (1, 1));
        assert_relative_eq!(jac[(0, 0)], 1.0 + 0.6 * 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_jacobian_1x1_with_n_bar_calculation() {
        // N = 2.0 → N̄ = 0.6·2.0 = 1.2 → M₀₀ = 1.0 + 1.2·0.5 = 1.6
        let sp = SpeciesParams::new("T", 1.0, 0.5, 2, inj());
        let m = LangmuirMulti::new(vec![sp], 50, 0.4, 0.001, 0.25).unwrap();
        assert_relative_eq!(m.jacobian(&[0.0])[(0, 0)], 1.0 + 1.2 * 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_jacobian_2x2_shape() {
        let jac = two_species_model().jacobian(&[1e-8, 1e-8]);
        assert_eq!((jac.nrows(), jac.ncols()), (2, 2));
    }

    #[test]
    fn test_jacobian_diagonal_positive() {
        let jac = two_species_model().jacobian(&[1e-4, 1e-4]);
        assert!(jac[(0, 0)] > 0.0 && jac[(1, 1)] > 0.0);
    }

    #[test]
    fn test_jacobian_off_diagonal_negative() {
        let jac = two_species_model().jacobian(&[1e-4, 1e-4]);
        assert!(jac[(0, 1)] <= 0.0 && jac[(1, 0)] <= 0.0);
    }

    #[test]
    fn test_jacobian_off_diagonal_zero_when_ci_zero() {
        // M_ij ∝ C_i — if C_i = 0 then M_ij = 0
        let jac = two_species_model().jacobian(&[0.0, 1e-3]);
        assert_relative_eq!(jac[(0, 1)], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_jacobian_all_finite() {
        let jac = two_species_model().jacobian(&[1e-6, 2e-6]);
        assert!(jac.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_jacobian_3x3_dimensions() {
        let mut m = two_species_model();
        m.add_species(species("C", 1.0, 5.0, 1)).unwrap();
        let jac = m.jacobian(&[1e-8, 1e-8, 1e-8]);
        assert_eq!((jac.nrows(), jac.ncols()), (3, 3));
    }

    // ── compute_inverse_propagation_matrix ───────────────────────────────────

    #[test]
    fn test_inverse_matrix_invertible() {
        assert!(
            two_species_model()
                .inverse_propagation(&[1e-8, 1e-8])
                .is_ok()
        );
    }

    #[test]
    fn test_inverse_matrix_dimensions() {
        let inv = two_species_model()
            .inverse_propagation(&[1e-8, 1e-8])
            .unwrap();
        assert_eq!((inv.nrows(), inv.ncols()), (2, 2));
    }

    #[test]
    fn test_inverse_matrix_times_original_is_identity() {
        // A · A⁻¹ ≈ I (numerical tolerance 1e-10)
        let model = two_species_model();
        let c = &[1e-6, 2e-6];
        let mat = DMatrix::identity(2, 2) + model.fe * model.jacobian(c);
        let inv = model.inverse_propagation(c).unwrap();
        let prod = &mat * &inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[(i, j)] - expected).abs() < 1e-10,
                    "A·A⁻¹[{i},{j}] should be {}",
                    prod[(i, j)]
                );
            }
        }
    }

    // ── setup_initial_state ───────────────────────────────────────────────────

    #[test]
    fn test_initial_state_is_empty_column() {
        // The column starts empty — injection enters via the boundary condition
        let mat = two_species_model()
            .setup_initial_state()
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_matrix()
            .clone_owned();

        assert_eq!((mat.nrows(), mat.ncols()), (100, 2));
        assert!(
            mat.iter().all(|&v| v == 0.0),
            "Initial state must be zero everywhere — injection handled by compute_physics"
        );
    }

    // ── compute_physics — temporal injection ─────────────────────────────────

    #[test]
    fn test_compute_physics_output_shape() {
        let model = two_species_model();
        let phys = model.compute_physics(&model.setup_initial_state());
        let mat = phys
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_matrix();
        assert_eq!((mat.nrows(), mat.ncols()), (100, 2));
    }

    #[test]
    fn test_compute_physics_reads_time_from_metadata() {
        // Gaussian injection peaking at t=10: inlet effect at t=10 must exceed t=0
        let mut model = LangmuirMulti::new(
            vec![SpeciesParams::new(
                "A",
                1.0,
                0.5,
                1,
                TemporalInjection::gaussian(10.0, 2.0, 0.1),
            )],
            100,
            0.4,
            0.001,
            0.25,
        )
        .unwrap();
        model
            .add_species(SpeciesParams::new(
                "B",
                1.0,
                2.0,
                1,
                TemporalInjection::gaussian(10.0, 2.0, 0.1),
            ))
            .unwrap();

        let mut state = model.setup_initial_state();

        state.set_metadata("time".to_string(), 0.0);
        let dc_t0 = model
            .compute_physics(&state)
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_matrix()
            .clone_owned();

        state.set_metadata("time".to_string(), 10.0);
        let dc_t10 = model
            .compute_physics(&state)
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_matrix()
            .clone_owned();

        assert!(
            dc_t10[(0, 0)].abs() > dc_t0[(0, 0)].abs(),
            "Inlet dC/dt for A must be stronger at injection peak (t=10)"
        );
        assert!(
            dc_t10[(0, 1)].abs() > dc_t0[(0, 1)].abs(),
            "Inlet dC/dt for B must be stronger at injection peak (t=10)"
        );
    }

    #[test]
    fn test_compute_physics_defaults_to_t0_without_metadata() {
        // Without metadata the model must not panic — defaults to t=0.0
        let model = two_species_model();
        let mat = model
            .compute_physics(&model.setup_initial_state())
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_matrix()
            .clone_owned();
        assert!(
            mat.iter().all(|v| v.is_finite()),
            "compute_physics must produce finite values when metadata is absent"
        );
    }

    #[test]
    fn test_compute_physics_no_nan_or_inf() {
        let model = two_species_model();
        let mut st = model.setup_initial_state();
        st.set_metadata("time".to_string(), 5.0);
        let mat = model
            .compute_physics(&st)
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_matrix()
            .clone_owned();
        assert!(
            mat.iter().all(|v| v.is_finite()),
            "All dC/dt values must be finite"
        );
    }

    #[test]
    fn test_compute_physics_interior_zero_on_empty_column() {
        // On an empty column, interior points (i>0) have zero gradient → dC/dt = 0.
        // Row 0 may be non-zero if injection.evaluate(t) > 0 (Dirac at t=0).
        let model = two_species_model();
        let mut st = model.setup_initial_state();
        st.set_metadata("time".to_string(), 0.0);
        let mat = model
            .compute_physics(&st)
            .get(PhysicalQuantity::Concentration)
            .unwrap()
            .as_matrix()
            .clone_owned();

        for i in 1..100 {
            for k in 0..2 {
                assert!(
                    mat[(i, k)].abs() < 1e-15,
                    "dC/dt[{i},{k}] must be 0 for interior of empty column"
                );
            }
        }
    }

    // ── PhysicalModel trait ───────────────────────────────────────────────────

    #[test]
    fn test_points_returns_n_points() {
        assert_eq!(single_model().points(), 100);
    }

    #[test]
    fn test_name() {
        assert_eq!(single_model().name(), "Langmuir Multi-Species");
    }

    #[test]
    fn test_description_is_some() {
        assert!(single_model().description().is_some());
    }

    // ── set_injection_all / set_injection_for ─────────────────────────────────

    #[test]
    fn test_set_injection_all_applies_to_every_species() {
        let mut model = two_species_model();
        model.set_injection_all(TemporalInjection::dirac(5.0, 0.1));

        // Both species must return a non-zero value at t=5
        for sp in model.species_params() {
            assert!(
                sp.injection.evaluate(5.0) > 0.0,
                "Species '{}' must have non-zero injection at peak",
                sp.name
            );
        }
    }

    #[test]
    fn test_set_injection_all_overwrites_previous_profile() {
        let mut model = two_species_model();
        // First: set all to gaussian
        model.set_injection_all(TemporalInjection::gaussian(10.0, 3.0, 0.1));
        // Then overwrite with none
        model.set_injection_all(TemporalInjection::none());

        for sp in model.species_params() {
            assert!(
                sp.injection.evaluate(10.0).abs() < 1e-15,
                "Species '{}' injection must be none after overwrite",
                sp.name
            );
        }
    }

    #[test]
    fn test_set_injection_for_targets_single_species() {
        let mut model = two_species_model();
        model.set_injection_all(TemporalInjection::none());
        model
            .set_injection_for("B", TemporalInjection::dirac(5.0, 0.1))
            .unwrap();

        // A must stay none
        let sp_a = model
            .species_params()
            .iter()
            .find(|s| s.name == "A")
            .unwrap();
        assert!(sp_a.injection.evaluate(5.0).abs() < 1e-15);

        // B must be non-zero
        let sp_b = model
            .species_params()
            .iter()
            .find(|s| s.name == "B")
            .unwrap();
        assert!(sp_b.injection.evaluate(5.0) > 0.0);
    }

    #[test]
    fn test_set_injection_for_unknown_species_returns_err() {
        let mut model = single_model();
        let result = model.set_injection_for("Unknown", TemporalInjection::none());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown"));
    }

    #[test]
    fn test_set_injection_for_does_not_alter_physical_params() {
        let mut model = two_species_model();
        model
            .set_injection_for("A", TemporalInjection::gaussian(10.0, 3.0, 0.1))
            .unwrap();

        assert_eq!(model.n_species(), 2);
        assert_eq!(model.spatial_points(), 100);
        assert!((model.porosity() - 0.4).abs() < 1e-15);
    }

    // ── Exportable ────────────────────────────────────────────────────────────

    /// Builds a minimal trajectory (3 steps, all-zero concentrations) suitable
    /// for exercising `to_map` without running a full simulation.
    fn make_trajectory(model: &LangmuirMulti) -> (Vec<f64>, Vec<PhysicalState>) {
        let state = model.setup_initial_state();
        let time_points = vec![0.0, 0.5, 1.0];
        let trajectory = vec![state.clone(), state.clone(), state.clone()];
        (time_points, trajectory)
    }

    #[test]
    fn test_to_map_species_blocks() {
        let model = two_species_model();
        let (tp, traj) = make_trajectory(&model);
        let map = model.to_map(&tp, &traj, &HashMap::new());

        let profiles = &map["data"]["profiles"];
        assert!(
            profiles.get("species_0").is_some(),
            "species_0 block missing"
        );
        assert!(
            profiles.get("species_1").is_some(),
            "species_1 block missing"
        );

        // name must be colocated with the concentration data
        assert_eq!(profiles["species_0"]["name"].as_str().unwrap(), "A");
        assert_eq!(profiles["species_1"]["name"].as_str().unwrap(), "B");

        let conc0 = profiles["species_0"]["Concentration"].as_array().unwrap();
        assert_eq!(
            conc0.len(),
            tp.len(),
            "species_0 Concentration length mismatch"
        );
    }

    #[test]
    fn test_to_map_global_block_present_and_empty() {
        // The global block is the extension point for future scalar/vector quantities.
        // It must be present even when all data is in species_N blocks (Matrix dispatch).
        let model = single_model();
        let (tp, traj) = make_trajectory(&model);
        let map = model.to_map(&tp, &traj, &HashMap::new());

        let profiles = map["data"]["profiles"].as_object().unwrap();
        assert!(
            profiles.contains_key("global"),
            "global extension block missing"
        );
        assert!(
            profiles["global"].as_object().unwrap().is_empty(),
            "global block must be empty for Matrix data"
        );
    }

    #[test]
    fn test_to_map_metadata_species_array() {
        let model = two_species_model();
        let (tp, traj) = make_trajectory(&model);
        let map = model.to_map(&tp, &traj, &HashMap::new());

        let meta = map["metadata"].as_object().unwrap();
        assert_eq!(meta["n_species"].as_u64().unwrap(), 2);

        let species_arr = meta["species"].as_array().unwrap();
        assert_eq!(species_arr.len(), 2, "species array length mismatch");
        assert_eq!(species_arr[0]["name"].as_str().unwrap(), "A");
        assert_eq!(species_arr[1]["name"].as_str().unwrap(), "B");
        assert!((species_arr[0]["lambda"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_from_map_round_trip() {
        let original = two_species_model();
        let (tp, traj) = make_trajectory(&original);
        let map = original.to_map(&tp, &traj, &HashMap::new());

        let reconstructed = LangmuirMulti::from_map(map).expect("from_map must succeed");

        assert_eq!(reconstructed.n_species(), original.n_species());
        assert_eq!(reconstructed.spatial_points(), original.spatial_points());
        assert!((reconstructed.porosity() - original.porosity()).abs() < 1e-12);
        assert!((reconstructed.velocity() - original.velocity()).abs() < 1e-12);
        assert!((reconstructed.column_length() - original.column_length()).abs() < 1e-12);

        // Species names must survive the round-trip (order preserved)
        assert_eq!(reconstructed.species_names(), original.species_names());
    }

    #[test]
    fn test_from_map_species_count_mismatch() {
        use serde_json::json;
        let map = json!({
            "metadata": {
                "nz": 100, "porosity": 0.4, "velocity": 0.001, "length": 0.25,
                "species": []
            }
        })
        .as_object()
        .cloned()
        .unwrap();

        assert!(
            matches!(
                LangmuirMulti::from_map(map),
                Err(ExportError::SpeciesCountMismatch { .. })
            ),
            "Empty species array must yield SpeciesCountMismatch"
        );
    }

    #[test]
    fn test_from_map_missing_key() {
        use serde_json::json;

        // "porosity" absent — all other required keys present
        let map = json!({
            "metadata": {
                "nz": 100, "velocity": 0.001, "length": 0.25,
                "species": [{"name": "A", "lambda": 1.0, "langmuir_k": 0.5, "port_number": 1}]
            }
        })
        .as_object()
        .cloned()
        .unwrap();

        assert!(
            matches!(
                LangmuirMulti::from_map(map),
                Err(ExportError::MissingKey(_))
            ),
            "Missing 'porosity' must yield MissingKey"
        );
    }
}
