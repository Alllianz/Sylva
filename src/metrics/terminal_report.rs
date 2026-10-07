use crate::metrics::backtest_report::BacktestReport;
use chrono::{TimeZone, Utc};

pub fn print_backtest_summary(model_name: &str, timeframe: &str, report: &BacktestReport) {
    println!("\n=======================================================");
    println!("  RESULTADOS DEL BACKTEST: {}", model_name);
    println!("  Timeframe: {}", timeframe);
    println!("=======================================================");
    println!(" Capital Inicial:     ${:>12.2}", report.initial_capital);
    println!(" Capital Final:       ${:>12.2}", report.final_capital);
    println!(" PnL Neto:            ${:>12.2} ({:+.2}%)", report.net_profit, report.total_return_pct);
    println!(" Total de Operaciones: {:>11}", report.total_trades);
    println!(" Ganadoras / Perdedoras:{:>5} / {:<5} (WinRate: {:.2}%)", report.winning_trades, report.losing_trades, report.win_rate_pct);
    println!(" Profit Factor:        {:>11.2}", report.profit_factor);
    println!(" Ganancia Bruta:      ${:>12.2}", report.gross_profit);
    println!(" Pérdida Bruta:       ${:>12.2}", report.gross_loss);
    println!(" Comisiones Totales:  ${:>12.2}", report.total_fees);
    println!(" Max Drawdown:        ${:>12.2} ({:.2}%)", report.max_drawdown_amount, report.max_drawdown_pct);
    println!(" Max Estancamiento:    {:>11} barras", report.max_stagnation_bars);
    println!(" Sharpe Ratio:         {:>11.2}", report.sharpe_ratio);
    println!(" Sortino Ratio:        {:>11.2}", report.sortino_ratio);
    println!(" Longs / Shorts / Liq: {:>5} / {:<5} / {:<4}", report.total_longs, report.total_shorts, report.liquidations);
    println!("-------------------------------------------------------");

    if !report.trades.is_empty() {
        println!(" Últimas 5 operaciones registradas:");
        let take_count = report.trades.len().min(5);
        for trade in &report.trades[report.trades.len() - take_count..] {
            let entry_dt = Utc.timestamp_millis_opt(trade.entry_timestamp).unwrap();
            let exit_dt = Utc.timestamp_millis_opt(trade.exit_timestamp).unwrap();
            let dir = if trade.is_long { "LONG" } else { "SHORT" };
            println!(
                "  #{} [{}] In: {} (${:.2}) -> Out: {} (${:.2}) | Net: ${:+.2} ({:+.2}%) | Motivo: {:?}",
                trade.trade_id,
                dir,
                entry_dt.format("%Y-%m-%d %H:%M"),
                trade.entry_price,
                exit_dt.format("%Y-%m-%d %H:%M"),
                trade.exit_price,
                trade.net_pnl,
                trade.return_pct,
                trade.exit_reason
            );
        }
    }
    println!("=======================================================\n");
}
