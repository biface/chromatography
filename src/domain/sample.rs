//! Inlet injection profiles — sample description at the column inlet.
//!
//! This module defines [`Sample`], which groups the temporal injection profiles
//! at the column inlet ($z = 0$) for one or more species.
//!
//! # Design
//!
//! `Sample` uses the same key/value scheme as [`PhysicalModel::set_injections`]:
//!
//! | Key | Meaning |
//! |-----|---------|
//! | `None` | Default profile — applied to all species without a per-species override |
//! | `Some(name)` | Per-species override for the named species |
//!
//! This scheme is identical to the `HashMap<Option<String>, TemporalInjection>`
//! already used inside `LangmuirSingle` and `LangmuirMulti`, and integrates
//! directly with [`PhysicalModel::set_injections`].
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::domain::Sample;
//! use chrom_rs::models::TemporalInjection;
//!
//! // Single Gaussian profile applied to all species
//! let sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
//!
//! // Per-species override
//! let mut sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
//! sample.set_species("B", TemporalInjection::rectangle(5.0, 15.0, 0.05));
//! ```

use crate::models::TemporalInjection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Sample
// =============================================================================

/// Inlet injection profiles for a chromatographic run.
///
/// Groups [`TemporalInjection`] profiles for one or more species.
/// The `None` key defines the default profile (all species);
/// `Some(name)` overrides the profile for a named species.
///
/// # Example
///
/// ```rust
/// use chrom_rs::domain::Sample;
/// use chrom_rs::models::TemporalInjection;
///
/// // Single profile for all species
/// let sample = Sample::uniform(TemporalInjection::dirac(5.0, 0.1));
/// assert!(sample.default_injection().is_some());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Injection profiles indexed by species.
    ///
    /// Profils d'injection indexés par espèce :
    /// - `None`    → profil par défaut pour toutes les espèces
    /// - `Some(s)` → override pour l'espèce nommée `s`
    pub injections: HashMap<Option<String>, TemporalInjection>,
}

impl Sample {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Creates an empty sample with no injection profiles.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::Sample;
    ///
    /// let sample = Sample::empty();
    /// assert!(sample.is_empty());
    /// ```
    pub fn empty() -> Self {
        Self {
            injections: HashMap::new(),
        }
    }

    /// Creates a sample with a single default injection profile applied to
    /// all species (key `None`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::Sample;
    /// use chrom_rs::models::TemporalInjection;
    ///
    /// let sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
    /// assert!(sample.default_injection().is_some());
    /// ```
    pub fn uniform(injection: TemporalInjection) -> Self {
        let mut injections = HashMap::new();
        injections.insert(None, injection);
        Self { injections }
    }

    // =========================================================================
    // Accessors and mutators
    // =========================================================================

    /// Sets or replaces the default injection profile (all species).
    pub fn set_default(&mut self, injection: TemporalInjection) {
        self.injections.insert(None, injection);
    }

    /// Sets or replaces the injection profile for a named species.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chrom_rs::domain::Sample;
    /// use chrom_rs::models::TemporalInjection;
    ///
    /// let mut sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
    /// sample.set_species("B", TemporalInjection::rectangle(5.0, 15.0, 0.05));
    /// assert!(sample.species_injection("B").is_some());
    /// ```
    pub fn set_species(&mut self, name: &str, injection: TemporalInjection) {
        self.injections.insert(Some(name.to_string()), injection);
    }

    /// Returns the default injection profile, if any.
    pub fn default_injection(&self) -> Option<&TemporalInjection> {
        self.injections.get(&None)
    }

    /// Returns the injection profile for a named species, if any.
    pub fn species_injection(&self, name: &str) -> Option<&TemporalInjection> {
        self.injections.get(&Some(name.to_string()))
    }

    /// Returns `true` if no injection profile is defined.
    pub fn is_empty(&self) -> bool {
        self.injections.is_empty()
    }

    /// Returns a reference to the inner map, compatible with
    /// [`PhysicalModel::set_injections`].
    pub fn as_map(&self) -> &HashMap<Option<String>, TemporalInjection> {
        &self.injections
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_empty() {
        let sample = Sample::empty();
        assert!(sample.is_empty());
        assert!(sample.default_injection().is_none());
    }

    #[test]
    fn test_sample_uniform() {
        let sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
        assert!(!sample.is_empty());
        assert!(sample.default_injection().is_some());
    }

    #[test]
    fn test_sample_set_default() {
        let mut sample = Sample::empty();
        sample.set_default(TemporalInjection::dirac(5.0, 0.1));
        assert!(sample.default_injection().is_some());
    }

    #[test]
    fn test_sample_set_species() {
        let mut sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
        sample.set_species("B", TemporalInjection::rectangle(5.0, 15.0, 0.05));
        assert!(sample.species_injection("B").is_some());
        assert!(sample.species_injection("A").is_none());
    }

    #[test]
    fn test_sample_as_map_compatible_with_set_injections() {
        let mut sample = Sample::uniform(TemporalInjection::dirac(5.0, 0.1));
        sample.set_species("Malic", TemporalInjection::gaussian(10.0, 2.0, 0.05));

        let map = sample.as_map();
        assert!(map.contains_key(&None));
        assert!(map.contains_key(&Some("Malic".to_string())));
        assert_eq!(map.len(), 2);
    }
}
