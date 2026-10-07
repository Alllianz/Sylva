use crate::engine::types::{ExitReason, TradeLog};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub initial_capital: f64,
    pub final_capital: f64,
    pub net_profit: f64,
    pub total_return_pct: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate_pct: f64,
    pub profit_factor: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub total_fees: f64,
    pub max_drawdown_amount: f64,
    pub max_drawdown_pct: f64,
    pub max_stagnation_bars: usize,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub total_longs: usize,
    pub total_shorts: usize,
    pub liquidations: usize,
    pub trades: Vec<TradeLog>,
    pub equity_curve: Vec<(i64, f64)>, // (timestamp, equity)
}

pub fn calculate_metrics(
    initial_capital: f64,
    final_capital: f64,
    trades: Vec<TradeLog>,
    equity_curve: Vec<(i64, f64)>,
    max_stagnation_bars: usize,
) -> BacktestReport {
    let total_trades = trades.len();
    let mut winning_trades = 0;
    let mut losing_trades = 0;
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    let mut total_fees = 0.0;
    let mut total_longs = 0;
    let mut total_shorts = 0;
    let mut liquidations = 0;

    let mut returns = Vec::with_capacity(total_trades);

    for t in &trades {
        total_fees += t.total_fees;
        if t.is_long {
            total_longs += 1;
        } else {
            total_shorts += 1;
        }

        if t.exit_reason == ExitReason::Liquidation {
            liquidations += 1;
        }

        if t.net_pnl > 0.0 {
            winning_trades += 1;
            gross_profit += t.net_pnl;
        } else {
            losing_trades += 1;
            gross_loss += t.net_pnl.abs();
        }

        returns.push(t.return_pct);
    }

    let net_profit = final_capital - initial_capital;
    let total_return_pct = if initial_capital > 0.0 {
        (net_profit / initial_capital) * 100.0
    } else {
        0.0
    };

    let win_rate_pct = if total_trades > 0 {
        (winning_trades as f64 / total_trades as f64) * 100.0
    } else {
        0.0
    };

    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // Calcular Max Drawdown sobre la curva de balance
    let mut peak = initial_capital;
    let mut max_dd_amount = 0.0;
    let mut max_dd_pct = 0.0;

    for &(_, eq) in &equity_curve {
        if eq > peak {
            peak = eq;
        }
        let dd = peak - eq;
        if dd > max_dd_amount {
            max_dd_amount = dd;
        }
        if peak > 0.0 {
            let dd_pct = (dd / peak) * 100.0;
            if dd_pct > max_dd_pct {
                max_dd_pct = dd_pct;
            }
        }
    }

    // Sharpe y Sortino
    let mean_ret = if !returns.is_empty() {
        returns.iter().sum::<f64>() / (returns.len() as f64)
    } else {
        0.0
    };

    let variance = if returns.len() > 1 {
        returns
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / ((returns.len() - 1) as f64)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let sharpe_ratio = if std_dev > 1e-8 {
        (mean_ret / std_dev) * (returns.len() as f64).sqrt()
    } else {
        0.0
    };

    let downside_variance = if returns.len() > 1 {
        returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>()
            / ((returns.len() - 1) as f64)
    } else {
        0.0
    };
    let downside_std_dev = downside_variance.sqrt();

    let sortino_ratio = if downside_std_dev > 1e-8 {
        (mean_ret / downside_std_dev) * (returns.len() as f64).sqrt()
    } else {
        0.0
    };

    BacktestReport {
        initial_capital,
        final_capital,
        net_profit,
        total_return_pct,
        total_trades,
        winning_trades,
        losing_trades,
        win_rate_pct,
        profit_factor,
        gross_profit,
        gross_loss,
        total_fees,
        max_drawdown_amount: max_dd_amount,
        max_drawdown_pct: max_dd_pct,
        max_stagnation_bars,
        sharpe_ratio,
        sortino_ratio,
        total_longs,
        total_shorts,
        liquidations,
        trades,
        equity_curve,
    }
}

impl BacktestReport {
    /// Reduce la curva de equidad a un número ligero de puntos (ej. 100) y descarta los logs detallados de trades
    /// para evitar agotar la memoria RAM durante búsquedas evolutivas masivas (reducción del 99.8% de memoria).
    pub fn downsample_for_summary(&mut self, max_points: usize) {
        self.trades = Vec::new();
        if self.equity_curve.len() > max_points && max_points > 1 {
            let n = self.equity_curve.len();
            let step = (n - 1) as f64 / (max_points - 1) as f64;
            let mut sampled = Vec::with_capacity(max_points);
            for i in 0..max_points {
                let idx = ((i as f64 * step).round() as usize).min(n - 1);
                sampled.push(self.equity_curve[idx]);
            }
            self.equity_curve = sampled;
        }
    }

    /// Elimina completamente los buffers pesados (trades y curva) reteniendo únicamente las métricas escalares.
    pub fn strip_heavy_buffers(&mut self) {
        self.trades = Vec::new();
        self.equity_curve = Vec::new();
    }
}

