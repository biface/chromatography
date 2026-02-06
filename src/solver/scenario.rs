//! Simulation scenario definition
//!
//! A scenario combines a physical model with boundary conditions.
use crate::physics::traits::PhysicalModel;
use crate::solver::boundary::DomainBoundaries;

/// Simulation scenario
///
/// Defines a specific case to simulate:
/// - Physical model (equations)
/// - Boundary conditions (domain boundaries)
///
/// # Design
///
/// The same scenario can be solved with different numerical methods.
/// This is the "WHAT to solve" (not "HOW to solve").
///
/// # Examples
///
/// ```rust
/// // Define scenario
/// let scenario = Scenario::new(model, boundaries);
///
/// // Solve with different methods
/// let result1 = euler_solver.solve(&scenario, &config1)?;
/// let result2 = newton_solver.solve(&scenario, &config2)?;
/// ```

pub struct Scenario {
    /// Physical model (equations)
    pub model: Box<dyn PhysicalModel>,

    /// Conditions and boundaries
    pub conditions: DomainBoundaries
}

impl Scenario {

    /// Create a scenario
    pub fn new(model: Box<dyn PhysicalModel>, conditions: DomainBoundaries) -> Self {
        Self { model, conditions }
    }

    /// Verifying scenario content (mainly boundaries)
    pub fn validate(&self) -> Result<(), String> {
        self.conditions.validate()
    }

    /// Get model name
    pub fn get_model_name(&self) -> &str {
        self.model.name()
    }

    /// n-dim resolution
    pub fn ndim(&self) -> usize {
        self.conditions.ndim()
    }

    /// spatial dimension
    pub fn sdim(&self) -> usize {
        self.conditions.sdim()
    }

    /// time dependant equations
    pub fn is_time_dependent(&self) -> bool {
        self.conditions.is_time_dependent()
    }
}

impl std::fmt::Debug for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scenario")
        .field("name", &self.get_model_name())
        .field("dimension", &self.ndim())
        .field("spatial dim", &self.sdim())
        .field("is time dependent", &self.is_time_dependent())
        .field("Boundaries / conditions", &self.conditions)
        .finish()
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::traits::{PhysicalState, PhysicalModel};
    use crate::solver::boundary::{DomainBoundaries, DimensionBoundary, TimeAxisConvention};

    // Mocking a Physical model
    struct MockModel {
        content: Vec<PhysicalState>,
        model_name: String
    }

    impl MockModel {
        fn new(name: &str) -> Self {
            Self {
                content: vec![PhysicalState::empty()],
                model_name: name.to_string()
            }
        }
    }

    impl PhysicalModel for MockModel {
        fn points(&self) -> usize {
            10
        }

        fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
            if !self.content.is_empty() {
                self.content.last().unwrap().clone() + state.clone()
            } else { 
                state.clone()
            }
        }

        fn setup_initial_state(&self) -> PhysicalState {
            PhysicalState::empty()
        }

        fn name(&self) -> &str {
            self.model_name.as_str()
        }
    }

    // ============================================================================
    // Helper Functions - Utilisant l'API réelle
    // ============================================================================

    /// Temporal-only scenario (ODE)
    fn create_temporal_scenario() -> Scenario {
        let model = Box::new(MockModel::new("TemporalModel"));
        let initial = PhysicalState::empty();
        let boundaries = DomainBoundaries::temporal(initial);
        Scenario::new(model, boundaries)
    }

    /// 1D static spatial scenario
    fn create_1d_spatial_scenario() -> Scenario {
        let model = Box::new(MockModel::new("1D_SpatialModel"));

        let x_left = PhysicalState::empty();
        let x_right = PhysicalState::empty();

        let boundaries = DomainBoundaries::spatial(
            &["x"],
            vec![x_left],
            vec![x_right],
        );

        Scenario::new(model, boundaries)
    }

    /// 1D static hybrid (spatial and time) scenario
    fn create_1d_time_scenario() -> Scenario {
        let model = Box::new(MockModel::new("1D_TimeModel"));

        let x_left = PhysicalState::empty();
        let x_right = PhysicalState::empty();
        let initial = PhysicalState::empty();

        let boundaries = DomainBoundaries::mixed(
            &["x"],
            vec![x_left],
            vec![x_right],
            initial,
        );

        Scenario::new(model, boundaries)
    }

    /// 2D Spatial static scenario
    fn create_2d_spatial_scenario() -> Scenario {
        let model = Box::new(MockModel::new("2D_SpatialModel"));

        let boundaries = DomainBoundaries::spatial(
            &["x", "y"],
            vec![PhysicalState::empty(), PhysicalState::empty()],
            vec![PhysicalState::empty(), PhysicalState::empty()],
        );

        Scenario::new(model, boundaries)
    }

    /// 3D static hybrid scenario
    fn create_3d_time_scenario() -> Scenario {
        let model = Box::new(MockModel::new("3D_TimeModel"));

        let boundaries = DomainBoundaries::mixed(
            &["x", "y", "z"],
            vec![
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
            ],
            vec![
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
            ],
            PhysicalState::empty(),
        );

        Scenario::new(model, boundaries)
    }

    /// advanced custom scenario
    fn create_custom_scenario(
        name: &str,
        dimensions: Vec<DimensionBoundary>,
        convention: TimeAxisConvention,
    ) -> Scenario {
        let model = Box::new(MockModel::new(name));
        let boundaries = DomainBoundaries::create(dimensions, Some(convention));
        Scenario::new(model, boundaries)
    }

    // ============================================================================
    // 1. Constructors & Basic Accessors
    // ============================================================================

    #[test]
    fn test_scenario_creation_basic() {
        let model = Box::new(MockModel::new("TestModel"));
        assert_eq!(model.points(), 10);

        let initial = PhysicalState::empty();
        let boundaries = DomainBoundaries::temporal(initial);
        let scenario = Scenario::new(model, boundaries);

        assert_eq!(scenario.get_model_name(), "TestModel");
    }

    #[test]
    fn test_scenario_creation_with_different_models() {
        let names = vec!["Model1", "Model2", "ComplexModel_v2.0"];

        for name in names {
            let model = Box::new(MockModel::new(name));
            let boundaries = DomainBoundaries::temporal(PhysicalState::empty());
            let scenario = Scenario::new(model, boundaries);

            assert_eq!(scenario.get_model_name(), name);
        }
    }

    #[test]
    fn test_get_model_name_various_scenarios() {
        let scenario_temporal = create_temporal_scenario();
        assert_eq!(scenario_temporal.get_model_name(), "TemporalModel");

        let scenario_1d = create_1d_spatial_scenario();
        assert_eq!(scenario_1d.get_model_name(), "1D_SpatialModel");

        let scenario_2d = create_2d_spatial_scenario();
        assert_eq!(scenario_2d.get_model_name(), "2D_SpatialModel");
    }

    // ============================================================================
    // 2. Validation Tests
    // ============================================================================

    #[test]
    fn test_validate_valid_scenarios() {
        // Test tous les scenarios créés par helpers
        let scenarios = vec![
            create_temporal_scenario(),
            create_1d_spatial_scenario(),
            create_1d_time_scenario(),
            create_2d_spatial_scenario(),
            create_3d_time_scenario(),
        ];

        for scenario in scenarios {
            assert!(
                scenario.validate().is_ok(),
                "Scenario '{}' should be valid",
                scenario.get_model_name()
            );
        }
    }

    #[test]
    fn test_validate_empty_boundaries() {
        // Boundaries vides -> erreur de validation
        let model = Box::new(MockModel::new("EmptyBoundariesModel"));
        let boundaries = DomainBoundaries::new(vec![]);
        let scenario = Scenario::new(model, boundaries);

        let result = scenario.validate();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Dimension boundaries cannot be empty."
        );
    }

    #[test]
    fn test_validate_dimension_without_states() {
        // DimensionBoundary sans états -> erreur
        let model = Box::new(MockModel::new("NoStatesModel"));

        let empty_dim = DimensionBoundary::new("x", vec![]);
        let boundaries = DomainBoundaries::new(vec![empty_dim]);
        let scenario = Scenario::new(model, boundaries);

        let result = scenario.validate();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Dimensions 'x' must have at least one boundary state"
        );
    }

    #[test]
    fn test_validate_duplicate_dimension_names() {
        // Deux dimensions avec même nom -> erreur
        let model = Box::new(MockModel::new("DuplicateModel"));

        let dim1 = DimensionBoundary::new(
            "x",
            vec![PhysicalState::empty(), PhysicalState::empty()],
        );
        let dim2 = DimensionBoundary::new(
            "x",  // Même nom !
            vec![PhysicalState::empty(), PhysicalState::empty()],
        );

        let boundaries = DomainBoundaries::new(vec![dim1, dim2]);
        let scenario = Scenario::new(model, boundaries);

        let result = scenario.validate();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "It is impossible to store two dimensions with the same name."
        );
    }

    // ============================================================================
    // 3. Dimension Queries
    // ============================================================================

    #[test]
    fn test_ndim_temporal_only() {
        // Temporal-only -> ndim = 1
        let scenario = create_temporal_scenario();
        assert_eq!(scenario.ndim(), 1);
    }

    #[test]
    fn test_ndim_1d_spatial() {
        // 1D spatial -> ndim = 1
        let scenario = create_1d_spatial_scenario();
        assert_eq!(scenario.ndim(), 1);
    }

    #[test]
    fn test_ndim_1d_time() {
        // 1D spatial + time -> ndim = 2
        let scenario = create_1d_time_scenario();
        assert_eq!(scenario.ndim(), 2);
    }

    #[test]
    fn test_ndim_2d_spatial() {
        // 2D spatial -> ndim = 2
        let scenario = create_2d_spatial_scenario();
        assert_eq!(scenario.ndim(), 2);
    }

    #[test]
    fn test_ndim_3d_time() {
        // 3D spatial + time -> ndim = 4
        let scenario = create_3d_time_scenario();
        assert_eq!(scenario.ndim(), 4);
    }

    #[test]
    fn test_sdim_temporal_only() {
        // Temporal-only -> sdim = 0 (pas de dimension spatiale)
        let scenario = create_temporal_scenario();
        assert_eq!(scenario.sdim(), 0);
    }

    #[test]
    fn test_sdim_1d_spatial() {
        // 1D spatial -> sdim = 1
        let scenario = create_1d_spatial_scenario();
        assert_eq!(scenario.sdim(), 1);
    }

    #[test]
    fn test_sdim_1d_time() {
        // 1D spatial + time -> sdim = 1
        let scenario = create_1d_time_scenario();
        assert_eq!(scenario.sdim(), 1);
    }

    #[test]
    fn test_sdim_2d_spatial() {
        // 2D spatial -> sdim = 2
        let scenario = create_2d_spatial_scenario();
        assert_eq!(scenario.sdim(), 2);
    }

    #[test]
    fn test_sdim_3d_time() {
        // 3D spatial + time -> sdim = 3
        let scenario = create_3d_time_scenario();
        assert_eq!(scenario.sdim(), 3);
    }

    #[test]
    fn test_ndim_sdim_consistency() {
        // Vérifier que ndim >= sdim toujours
        let scenarios = vec![
            create_temporal_scenario(),
            create_1d_spatial_scenario(),
            create_1d_time_scenario(),
            create_2d_spatial_scenario(),
            create_3d_time_scenario(),
        ];

        for scenario in scenarios {
            assert!(
                scenario.ndim() >= scenario.sdim(),
                "Model '{}': ndim ({}) should be >= sdim ({})",
                scenario.get_model_name(),
                scenario.ndim(),
                scenario.sdim()
            );
        }
    }

    // ============================================================================
    // 4. Time Dependency Tests
    // ============================================================================

    #[test]
    fn test_is_time_dependent_temporal_only() {
        // Temporal-only -> time-dependent
        let scenario = create_temporal_scenario();
        assert!(scenario.is_time_dependent());
    }

    #[test]
    fn test_is_time_dependent_spatial_only() {
        // Spatial seulement -> NOT time-dependent
        let scenario_1d = create_1d_spatial_scenario();
        assert!(!scenario_1d.is_time_dependent());

        let scenario_2d = create_2d_spatial_scenario();
        assert!(!scenario_2d.is_time_dependent());
    }

    #[test]
    fn test_is_time_dependent_mixed() {
        // Mixed (spatial + temporal) -> time-dependent
        let scenario_1d = create_1d_time_scenario();
        assert!(scenario_1d.is_time_dependent());

        let scenario_3d = create_3d_time_scenario();
        assert!(scenario_3d.is_time_dependent());
    }

    #[test]
    fn test_time_dependency_with_different_conventions() {
        // Test avec différentes conventions de temps
        let model = Box::new(MockModel::new("ConventionTestModel"));

        // Convention Last (time-dependent)
        let boundaries_last = DomainBoundaries::create(
            vec![
                DimensionBoundary::new("x", vec![
                    PhysicalState::empty(),
                    PhysicalState::empty()
                ]),
                DimensionBoundary::new("t", vec![PhysicalState::empty()]),
            ],
            Some(TimeAxisConvention::Last),
        );
        let scenario_last = Scenario::new(model, boundaries_last);
        assert!(scenario_last.is_time_dependent());

        // Convention None (NOT time-dependent)
        let model2 = Box::new(MockModel::new("ConventionTestModel2"));
        let boundaries_none = DomainBoundaries::create(
            vec![
                DimensionBoundary::new("x", vec![
                    PhysicalState::empty(),
                    PhysicalState::empty()
                ]),
            ],
            Some(TimeAxisConvention::None),
        );
        let scenario_none = Scenario::new(model2, boundaries_none);
        assert!(!scenario_none.is_time_dependent());
    }

    // ============================================================================
    // 5. Debug Formatting Tests
    // ============================================================================

    #[test]
    fn test_debug_format_basic() {
        let scenario = create_temporal_scenario();
        let debug_str = format!("{:?}", scenario);

        // Vérifier présence des champs principaux
        assert!(debug_str.contains("Scenario"));
        assert!(debug_str.contains("name"));
        assert!(debug_str.contains("dimension"));
    }

    #[test]
    fn test_debug_format_contains_model_name() {
        let scenarios = vec![
            ("TemporalModel", create_temporal_scenario()),
            ("1D_SpatialModel", create_1d_spatial_scenario()),
            ("2D_SpatialModel", create_2d_spatial_scenario()),
        ];

        for (expected_name, scenario) in scenarios {
            let debug_str = format!("{:?}", scenario);
            assert!(
                debug_str.contains(expected_name),
                "Debug string should contain model name '{}'",
                expected_name
            );
        }
    }

    #[test]
    fn test_debug_format_contains_dimensions() {
        let scenario = create_2d_spatial_scenario();
        let debug_str = format!("{:?}", scenario);

        // Vérifier présence dimensions (ndim = 2, sdim = 2)
        assert!(debug_str.contains("dimension"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("spatial dim"));
    }

    #[test]
    fn test_debug_format_time_dependency_true() {
        let scenario = create_1d_time_scenario();
        let debug_str = format!("{:?}", scenario);

        assert!(debug_str.contains("is time dependent"));
        assert!(debug_str.contains("true"));
    }

    #[test]
    fn test_debug_format_time_dependency_false() {
        let scenario = create_2d_spatial_scenario();
        let debug_str = format!("{:?}", scenario);

        assert!(debug_str.contains("is time dependent"));
        assert!(debug_str.contains("false"));
    }

    #[test]
    fn test_debug_format_contains_boundaries() {
        let scenario = create_1d_time_scenario();
        let debug_str = format!("{:?}", scenario);

        // Les boundaries doivent être incluses
        assert!(
            debug_str.contains("Boundaries") || debug_str.contains("conditions"),
            "Debug should contain boundaries information"
        );
    }

    // ============================================================================
    // 6. Integration Tests
    // ============================================================================

    #[test]
    fn test_scenario_workflow_complete() {
        // Workflow complet : création -> validation -> queries -> debug

        let model = Box::new(MockModel::new("WorkflowModel"));
        let boundaries = DomainBoundaries::spatial(
            &["x"],
            vec![PhysicalState::empty()],
            vec![PhysicalState::empty()],
        );

        // 1. Création
        let scenario = Scenario::new(model, boundaries);

        // 2. Validation
        assert!(scenario.validate().is_ok());

        // 3. Queries
        assert_eq!(scenario.get_model_name(), "WorkflowModel");
        assert_eq!(scenario.ndim(), 1);
        assert_eq!(scenario.sdim(), 1);
        assert!(!scenario.is_time_dependent());

        // 4. Debug
        let debug_str = format!("{:?}", scenario);
        assert!(debug_str.contains("WorkflowModel"));
    }

    #[test]
    fn test_multiple_scenarios_different_configs() {
        // Créer plusieurs scenarios et vérifier cohérence
        let scenarios = vec![
            ("Temporal", create_temporal_scenario(), 1, 0, true),
            ("1D Spatial", create_1d_spatial_scenario(), 1, 1, false),
            ("1D+Time", create_1d_time_scenario(), 2, 1, true),
            ("2D Spatial", create_2d_spatial_scenario(), 2, 2, false),
            ("3D+Time", create_3d_time_scenario(), 4, 3, true),
        ];

        for (name, scenario, expected_ndim, expected_sdim, expected_time_dep) in scenarios {
            // Validation
            assert!(
                scenario.validate().is_ok(),
                "{} scenario should be valid", name
            );

            // Dimensions
            assert_eq!(
                scenario.ndim(), expected_ndim,
                "{} scenario: wrong ndim", name
            );
            assert_eq!(
                scenario.sdim(), expected_sdim,
                "{} scenario: wrong sdim", name
            );

            // Time dependency
            assert_eq!(
                scenario.is_time_dependent(), expected_time_dep,
                "{} scenario: wrong time dependency", name
            );
        }
    }

    #[test]
    fn test_scenario_reusability() {
        // Vérifier qu'on peut appeler les méthodes plusieurs fois
        let scenario = create_1d_time_scenario();

        for _ in 0..5 {
            assert_eq!(scenario.get_model_name(), "1D_TimeModel");
            assert_eq!(scenario.ndim(), 2);
            assert_eq!(scenario.sdim(), 1);
            assert!(scenario.is_time_dependent());
            assert!(scenario.validate().is_ok());
        }
    }

    #[test]
    fn test_scenario_consistency_after_multiple_operations() {
        // Vérifier cohérence après multiples appels
        let scenario = create_3d_time_scenario();

        // Appels multiples
        let ndim1 = scenario.ndim();
        let ndim2 = scenario.ndim();
        assert_eq!(ndim1, ndim2, "ndim should be consistent");

        let sdim1 = scenario.sdim();
        let sdim2 = scenario.sdim();
        assert_eq!(sdim1, sdim2, "sdim should be consistent");

        let time_dep1 = scenario.is_time_dependent();
        let time_dep2 = scenario.is_time_dependent();
        assert_eq!(time_dep1, time_dep2, "time dependency should be consistent");

        // Validation multiple
        assert!(scenario.validate().is_ok());
        assert!(scenario.validate().is_ok());
    }

    // ============================================================================
    // 7. Edge Cases & Boundary Conditions
    // ============================================================================

    #[test]
    fn test_very_high_dimensions() {
        // Test avec beaucoup de dimensions spatiales
        let model = Box::new(MockModel::new("HighDimModel"));

        let names: Vec<&str> = vec!["dim0", "dim1", "dim2", "dim3", "dim4"];
        let states_begin = vec![
            PhysicalState::empty(),
            PhysicalState::empty(),
            PhysicalState::empty(),
            PhysicalState::empty(),
            PhysicalState::empty(),
        ];
        let states_end = states_begin.clone();

        let boundaries = DomainBoundaries::spatial(&names, states_begin, states_end);
        let scenario = Scenario::new(model, boundaries);

        assert_eq!(scenario.ndim(), 5);
        assert_eq!(scenario.sdim(), 5);
        assert!(!scenario.is_time_dependent());
        assert!(scenario.validate().is_ok());
    }

    #[test]
    fn test_model_name_with_special_characters() {
        // Test avec noms contenant caractères spéciaux
        let special_names = vec![
            "Model-2.0",
            "Test_Model",
            "Model (v1)",
            "Modèle UTF-8",  // UTF-8
        ];

        for name in special_names {
            let model = Box::new(MockModel::new(name));
            let boundaries = DomainBoundaries::temporal(PhysicalState::empty());
            let scenario = Scenario::new(model, boundaries);

            assert_eq!(scenario.get_model_name(), name);
            assert!(scenario.validate().is_ok());

            // Vérifier que Debug ne panic pas
            let debug_str = format!("{:?}", scenario);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_scenario_with_single_boundary_state() {
        // Test avec une seule boundary state par dimension
        let model = Box::new(MockModel::new("SingleStateModel"));

        let dim = DimensionBoundary::new(
            "t",
            vec![PhysicalState::empty()],
        );

        let boundaries = DomainBoundaries::new(vec![dim]);
        let scenario = Scenario::new(model, boundaries);

        assert!(scenario.validate().is_ok());
        assert_eq!(scenario.ndim(), 1);
    }

    #[test]
    fn test_scenario_with_many_boundary_states() {
        // Test avec plusieurs boundary states (pas juste 1 ou 2)
        let model = Box::new(MockModel::new("ManyStatesModel"));

        let states = vec![
            PhysicalState::empty(),
            PhysicalState::empty(),
            PhysicalState::empty(),
            PhysicalState::empty(),
        ];

        let dim = DimensionBoundary::new("x", states);
        let boundaries = DomainBoundaries::new(vec![dim]);
        let scenario = Scenario::new(model, boundaries);

        assert!(scenario.validate().is_ok());
    }

    #[test]
    fn test_mixed_dimension_names() {
        // Test avec noms de dimensions variés
        let model = Box::new(MockModel::new("MixedNamesModel"));

        let boundaries = DomainBoundaries::spatial(
            &["x", "y", "z", "r", "theta", "phi"],  // Mix cartésien/polaire
            vec![
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
            ],
            vec![
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
                PhysicalState::empty(),
            ],
        );

        let scenario = Scenario::new(model, boundaries);

        assert_eq!(scenario.ndim(), 6);
        assert_eq!(scenario.sdim(), 6);
        assert!(scenario.validate().is_ok());
    }

    #[test]
    fn test_time_axis_convention_first() {
        // Test avec convention First
        let model = Box::new(MockModel::new("FirstConventionModel"));

        let t_dim = DimensionBoundary::new("t", vec![PhysicalState::empty()]);
        let x_dim = DimensionBoundary::new(
            "x",
            vec![PhysicalState::empty(), PhysicalState::empty()],
        );

        let boundaries = DomainBoundaries::create(
            vec![t_dim, x_dim],
            Some(TimeAxisConvention::First),
        );

        let scenario = Scenario::new(model, boundaries);

        assert_eq!(scenario.ndim(), 2);
        assert_eq!(scenario.sdim(), 1);
        assert!(scenario.is_time_dependent());
    }

    #[test]
    fn test_time_axis_convention_index() {
        // Test avec convention Index
        let model = Box::new(MockModel::new("IndexConventionModel"));

        let dims = vec![
            DimensionBoundary::new(
                "x",
                vec![PhysicalState::empty(), PhysicalState::empty()],
            ),
            DimensionBoundary::new("t", vec![PhysicalState::empty()]),
            DimensionBoundary::new(
                "y",
                vec![PhysicalState::empty(), PhysicalState::empty()],
            ),
        ];

        let boundaries = DomainBoundaries::create(
            dims,
            Some(TimeAxisConvention::Index(1)),  // t est à l'index 1
        );

        let scenario = Scenario::new(model, boundaries);

        assert_eq!(scenario.ndim(), 3);
        assert_eq!(scenario.sdim(), 2);
        assert!(scenario.is_time_dependent());
    }

    #[test]
    fn test_debug_output_readability() {
        // Vérifier que le Debug est lisible pour tous types de scenarios
        let scenarios = vec![
            create_temporal_scenario(),
            create_1d_spatial_scenario(),
            create_1d_time_scenario(),
            create_2d_spatial_scenario(),
            create_3d_time_scenario(),
        ];

        for scenario in scenarios {
            let debug_str = format!("{:?}", scenario);

            // Vérifier format de base
            assert!(debug_str.contains("Scenario"));
            assert!(!debug_str.is_empty());
            assert!(debug_str.len() > 50);  // Doit être assez détaillé
        }
    }
}