use crate::dashboard::data_exporter::export_equity_series;
use crate::dashboard::html_template::render_html_dashboard;
use crate::dashboard::models::{DashboardModelParams, DashboardSummaryMetrics};
use crate::data::db::Kline;
use crate::metrics::backtest_report::BacktestReport;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;

pub fn generate_dual_dashboard(
    model_name: &str,
    tf: &str,
    klines: &[Kline],
    report_nom: &BacktestReport,
    report_pct: &BacktestReport,
    leverage: f64,
    capital_percent: f64,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let dashboard_dir = format!("dashboard/{}", tf);
    fs::create_dir_all(&dashboard_dir)?;

    let chart_data = export_equity_series(klines, &report_nom.equity_curve, &report_pct.equity_curve);

    let params = DashboardModelParams {
        model_name: model_name.to_string(),
        tf: tf.to_string(),
        leverage,
        capital_percent,
        initial_capital: report_nom.initial_capital,
        is_start_date: "2020-04-20".to_string(),
        oos_start_date: "2023-01-01".to_string(),
        is_start_idx: 0,
        oos_start_idx: (klines.len() * 7) / 10,
    };

    let summary = DashboardSummaryMetrics {
        capital_final_nom: report_nom.final_capital,
        net_profit_nom: report_nom.net_profit,
        total_return_nom_pct: report_nom.total_return_pct,
        max_dd_nom_pct: report_nom.max_drawdown_pct,
        max_dd_nom_dollar: report_nom.max_drawdown_amount,
        winrate_nom: report_nom.win_rate_pct,
        profit_factor_nom: report_nom.profit_factor,
        total_trades_nom: report_nom.total_trades,
        longs_count_nom: report_nom.total_longs,
        shorts_count_nom: report_nom.total_shorts,
        liquidations_nom: report_nom.liquidations,
        sharpe_nom: report_nom.sharpe_ratio,
        sortino_nom: report_nom.sortino_ratio,
        max_stagnation_nom: report_nom.max_stagnation_bars,

        capital_final_pct: report_pct.final_capital,
        net_profit_pct: report_pct.net_profit,
        total_return_pct_pct: report_pct.total_return_pct,
        max_dd_pct_pct: report_pct.max_drawdown_pct,
        max_dd_pct_dollar: report_pct.max_drawdown_amount,
        winrate_pct: report_pct.win_rate_pct,
        profit_factor_pct: report_pct.profit_factor,
        total_trades_pct: report_pct.total_trades,
        longs_count_pct: report_pct.total_longs,
        shorts_count_pct: report_pct.total_shorts,
        liquidations_pct: report_pct.liquidations,
        sharpe_pct: report_pct.sharpe_ratio,
        sortino_pct: report_pct.sortino_ratio,
        max_stagnation_pct: report_pct.max_stagnation_bars,
    };

    let html_content = render_html_dashboard(&params, &summary, &chart_data);

    let clean_model_name = sanitize_slug(model_name);
    let filename = format!("dashboard_dual_{}_{}.html", tf, clean_model_name);
    let full_path = format!("{}/{}", dashboard_dir, filename);

    let mut file = File::create(&full_path)?;
    file.write_all(html_content.as_bytes())?;

    println!("  🌐 Dashboard HTML interactivo generado en: {}", full_path);
    Ok(full_path)
}

fn sanitize_slug(name: &str) -> String {
    let mut clean = String::with_capacity(name.len());
    let mut last_was_under = false;
    for c in name.chars() {
        if c.is_alphanumeric() {
            clean.push(c.to_ascii_lowercase());
            last_was_under = false;
        } else if !last_was_under {
            clean.push('_');
            last_was_under = true;
        }
    }
    clean.trim_matches('_').to_string()
}
