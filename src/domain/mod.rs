//! Modèle du domaine — physical domain model.
//!
//! Ce module regroupe les types décrivant l'**équipement physique** d'un
//! système chromatographique, indépendamment du modèle mathématique choisi
//! (Langmuir, diffusion dans les pores, etc.).
//!
//! # Design (DD-011)
//!
//! `domain/` est une **façade de construction validée** : ses types
//! encapsulent et valident les paramètres physiques, puis les transmettent
//! aux modèles via le constructeur `from_domain`. Les modèles conservent
//! leurs champs internes — la migration de la propriété des données vers
//! `domain/` est différée à un jalon futur.
//!
//! # Modules
//!
//! | Module | Type principal | Rôle |
//! |--------|---------------|------|
//! | [`column`] | [`Column`] | Géométrie de la colonne |
//! | [`phases`] | [`MobilePhase`] | Propriétés de la phase mobile |
//! | [`sample`] | [`Sample`] | Profils d'injection à l'entrée |
//! | [`detector`] | [`Detector`] | Position du point de mesure |
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
