pub mod data_exporter;
pub mod dual_dashboard;
pub mod grid_dashboard;
pub mod html_template;
pub mod models;

pub use data_exporter::{export_equity_series, ExportedChartData};
pub use dual_dashboard::generate_dual_dashboard;
pub use grid_dashboard::{generate_gbdt_grid_dashboard, GbdtGridCandidateReport, GbdtGridDashboardSummary};
pub use html_template::render_html_dashboard;
pub use models::{DashboardModelParams, DashboardSummaryMetrics};
