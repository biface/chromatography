//! Common utilities for integration tests

pub mod mock_models;
pub mod test_helpers;

// Re-export commonly used items
pub use mock_models::{ExponentialDecay, ConstantGrowth, LinearTransport};
pub use test_helpers::{
    assert_states_close,
    compute_l2_error,
    create_simple_scenario,
    relative_error,
};