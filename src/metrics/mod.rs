pub mod autotuning;
pub mod backtest_report;
pub mod ranking;
pub mod smoothness;
pub mod terminal_report;

pub use autotuning::{
    calculate_autotuning_score, calculate_capital_max_dd_ratio, calculate_series_r2,
    calculate_slope_ratio, evaluate_equity_linearity, evaluate_report_astro_fitness,
    evaluate_report_sylva_fitness, LinearityMetrics,
};
pub use backtest_report::{calculate_metrics, BacktestReport};
pub use ranking::{compute_multicriteria_rankings, CandidateRankScore};
pub use smoothness::{calculate_equity_smoothness, SmoothnessMetrics};
pub use terminal_report::print_backtest_summary;
