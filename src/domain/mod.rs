//! Physical domain model — equipment description layer.
//!
//! This module groups the types describing the **physical equipment** of a
//! chromatographic system, independent of the mathematical model chosen
//! (Langmuir, pore-diffusion, dispersive, etc.).
//!
//! # Design (DD-011)
//!
//! `domain/` is a **validated construction facade**: its types encapsulate and
//! validate physical parameters, then pass them to models via the
//! `from_domain` constructor. Models retain their internal fields — migration
//! of data ownership to `domain/` is deferred to a future milestone.
//!
//! # Modules
//!
//! | Module | Primary type | Role |
//! |--------|-------------|------|
//! | [`column`] | [`Column`] | Column geometry |
//! | [`phases`] | [`MobilePhase`] | Mobile-phase properties |
//! | [`sample`] | [`Sample`] | Inlet injection profiles |
//! | [`detector`] | [`Detector`] | Signal measurement point |
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::domain::{Column, MobilePhase, Sample, Detector};
//! use chrom_rs::models::TemporalInjection;
//!
//! let column = Column::new(0.25, 100, 0.4, None).unwrap();
//! let mobile_phase = MobilePhase::new(1e-4, None).unwrap();
//! let sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
//! let detector = Detector::outlet();
//! ```

pub mod column;
pub mod detector;
pub mod phases;
pub mod sample;

pub use column::{Column, ColumnError};
pub use detector::{Detector, DetectorError, DetectorPosition};
pub use phases::{MobilePhase, MobilePhaseError};
pub use sample::Sample;
