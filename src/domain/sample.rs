//! Échantillon — injection profiles at the column inlet.
//!
//! Ce module définit [`Sample`], le type qui regroupe les profils d'injection
//! temporelle à l'entrée de la colonne ($z = 0$) pour une ou plusieurs espèces.
//!
//! # Design
//!
//! `Sample` utilise le même schéma clé/valeur que [`PhysicalModel::set_injections`] :
//!
//! | Clé | Signification |
//! |-----|--------------|
//! | `None` | Profil par défaut — appliqué à toutes les espèces sans override |
//! | `Some(name)` | Override par espèce nommée |
//!
//! Ce schéma est identique à celui de [`HashMap<Option<String>, TemporalInjection>`]
//! déjà utilisé dans `LangmuirSingle` et `LangmuirMulti`, et s'intègre
//! directement avec [`PhysicalModel::set_injections`].
//!
//! # Example
//!
//! ```rust
//! use chrom_rs::domain::Sample;
//! use chrom_rs::models::TemporalInjection;
//!
//! // Injection gaussienne pour toutes les espèces
//! let sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
//!
//! // Override par espèce
//! let mut sample = Sample::uniform(TemporalInjection::gaussian(10.0, 2.0, 0.1));
//! sample.set_species("B", TemporalInjection::rectangle(5.0, 15.0, 0.05));
//! ```

use crate::models::TemporalInjection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Sample
// =============================================================================

/// Profils d'injection à l'entrée de la colonne — inlet injection profiles.
///
/// Regroupe les [`TemporalInjection`] pour une ou plusieurs espèces.
/// La clé `None` définit le profil par défaut (toutes les espèces) ;
/// `Some(name)` surcharge le profil pour une espèce nommée.
///
/// # Example
///
/// ```rust
/// use chrom_rs::domain::Sample;
/// use chrom_rs::models::TemporalInjection;
///
/// // Profil unique pour toutes les espèces
/// let sample = Sample::uniform(TemporalInjection::dirac(5.0, 0.1));
/// assert!(sample.default_injection().is_some());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Profils d'injection indexés par espèce — injection profiles indexed by species.
    ///
    /// - `None`      → profil par défaut pour toutes les espèces
    /// - `Some(s)`   → override pour l'espèce nommée `s`
    pub injections: HashMap<Option<String>, TemporalInjection>,
}

impl Sample {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Crée un `Sample` vide — creates an empty sample.
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

    /// Crée un `Sample` avec un profil unique pour toutes les espèces.
    ///
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

    /// Définit ou remplace le profil par défaut (toutes les espèces).
    ///
    /// Sets or replaces the default injection profile (all species).
    pub fn set_default(&mut self, injection: TemporalInjection) {
        self.injections.insert(None, injection);
    }

    /// Définit ou remplace le profil d'une espèce nommée.
    ///
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

    /// Retourne le profil par défaut, s'il existe.
    ///
    /// Returns the default injection profile, if any.
    pub fn default_injection(&self) -> Option<&TemporalInjection> {
        self.injections.get(&None)
    }

    /// Retourne le profil d'une espèce nommée, s'il existe.
    ///
    /// Returns the injection profile for a named species, if any.
    pub fn species_injection(&self, name: &str) -> Option<&TemporalInjection> {
        self.injections.get(&Some(name.to_string()))
    }

    /// Retourne `true` si aucun profil n'est défini.
    ///
    /// Returns `true` if no injection profile is defined.
    pub fn is_empty(&self) -> bool {
        self.injections.is_empty()
    }

    /// Retourne une référence à la map interne, compatible avec
    /// [`PhysicalModel::set_injections`].
    ///
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
