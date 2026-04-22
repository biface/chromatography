//! Common utilities for integration tests

pub mod mock_models;
pub mod test_helpers;

// Re-export commonly used items
#[allow(unused_imports)]
pub use mock_models::{ConstantGrowth, ExponentialDecay, LinearTransport};
#[allow(unused_imports)]
pub use test_helpers::{
    assert_states_close, compute_l2_error, create_simple_scenario, relative_error,
};
