//! chrom-rs: Chromatography Simulation Framework
//!
//! A flexible and extensible framework for simulating liquid chromatographic
//! processes using numerical methods. Built with Rust for performance and safety.
//!
//! # Architecture
//!
//! chrom-rs is built on two core principles:
//!
//! 1. **Separation of physics and numerics** — physical models define the
//!    equations (what to solve); numerical solvers provide the integration
//!    method (how to solve them). The same model can be solved with any
//!    solver, and the same solver can integrate any model.
//!
//! 2. **Extensibility and type safety** — all extension points are traits;
//!    state is managed through typed containers; the API is stable from
//!    v0.1.0 onwards.
//!
//! # Quick Start
//!
//! ```rust
//! use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};
//! use chrom_rs::solver::{EulerSolver, Solver, SolverConfiguration, Scenario, DomainBoundaries};
//! use nalgebra::DVector;
//! use serde::{Deserialize, Serialize};
//!
//! # #[derive(Deserialize, Serialize)]
//! # struct MyModel;
//! # #[typetag::serde]
//! # impl PhysicalModel for MyModel {
//! #     fn points(&self) -> usize { 1 }
//! #     fn compute_physics(&self, state: &PhysicalState) -> PhysicalState { state.clone() }
//! #     fn setup_initial_state(&self) -> PhysicalState {
//! #         PhysicalState::new(PhysicalQuantity::Concentration, PhysicalData::Vector(nalgebra::DVector::from_vec(vec![1.0])))
//! #     }
//! #     fn name(&self) -> &str { "MyModel" }
//! # }
//! # fn main() -> Result<(), String> {
//! // 1. Configure physical model and scenario
//! let model = Box::new(MyModel);
//! let initial_state = model.setup_initial_state();
//! let boundaries = DomainBoundaries::temporal(initial_state);
//! let scenario = Scenario::new(model, boundaries);
//!
//! // 2. Configure solver
//! let config = SolverConfiguration::time_evolution(
//!     600.0,    // 10 minutes total time
//!     1000,     // 1000 time steps
//! );
//!
//! // 3. Run simulation
//! let solver = EulerSolver::new();
//! let result = solver.solve(&scenario, &config)?;
//!
//! // 4. Access results
//! println!("Simulation completed!");
//! println!("Trajectory length: {}", result.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`physics`] | Core traits and data types for physical models |
//! | [`models`] | Concrete chromatography models (Langmuir single and multi-species) |
//! | [`solver`] | Numerical solvers, scenario definition, and simulation result |
//! | [`config`] | YAML/JSON configuration file loaders |
//! | [`output`] | CSV export, JSON export, and chromatogram visualisation |
//! | [`cli`] | Command-line interface built on `dynamic-cli` |
//! | [`prelude`] | Convenience re-exports for the most common types |

/// Physical model traits, state containers, and data types.
///
/// Defines the extension points that all physical models must implement,
/// as well as the typed containers ([`physics::PhysicalState`],
/// [`physics::PhysicalData`]) used to exchange state between models and
/// solvers.
pub mod physics;

/// Concrete chromatography models.
///
/// Contains [`models::LangmuirSingle`] for single-species simulations and
/// [`models::LangmuirMulti`] for competitive multi-species adsorption, along
/// with the [`models::TemporalInjection`] type that defines inlet boundary
/// conditions as a function of time.
pub mod models;

/// Numerical solvers and simulation infrastructure.
///
/// Provides [`solver::Solver`] implementations (Forward Euler, RK4), the
/// [`solver::Scenario`] type that binds a model to its boundary conditions,
/// [`solver::SolverConfiguration`] for numerical parameters, and
/// [`solver::SimulationResult`] which holds the computed trajectory.
pub mod solver;

/// Result visualisation and data export.
///
/// Sub-modules cover CSV export, JSON export, and chromatogram plots via
/// `plotters`. See [`output::export`] and [`output::visualization`].
pub mod output;

/// Configuration file loaders for the three-file layout.
///
/// Each loader reads one YAML or JSON file and returns the corresponding
/// domain object: [`config::model::load_model`] → `Box<dyn PhysicalModel>`,
/// [`config::scenario::load_scenario`] → [`solver::DomainBoundaries`],
/// [`config::solver::load_solver`] → [`solver::SolverConfiguration`].
pub mod config;

/// Command-line interface.
///
/// Entry point is [`cli::build_app`], which assembles the `dynamic-cli`
/// application from the embedded `commands.yml` declaration and wires
/// [`cli::app::RunHandler`] to the simulation pipeline.
pub mod cli;

/// Convenient re-exports for the most commonly used types.
///
/// Import everything with `use chrom_rs::prelude::*` to get the core traits
/// and types without writing long paths.
///
/// ```rust
/// use chrom_rs::prelude::*;
/// ```
pub mod prelude {
    pub use crate::models::TemporalInjection;
    pub use crate::physics::{PhysicalData, PhysicalModel, PhysicalQuantity, PhysicalState};
    pub use crate::solver::{
        EulerSolver, RK4Solver, Scenario, SimulationResult, Solver, SolverConfiguration, SolverType,
    };
}
