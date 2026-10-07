pub mod equation_model;
pub mod partition;
pub mod precomputed_model;
pub mod simulator;
pub mod threshold_tuner;
pub mod types;

pub use equation_model::EquationModel;
pub use partition::get_dataset_partition_indices;
pub use precomputed_model::PrecomputedAlphaModel;
pub use simulator::BacktestSimulator;
pub use threshold_tuner::{ThresholdTuner, TunerResult};
pub use types::{BacktestConfig, ExitReason, Position, SignalAction, TradeLog};
