use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalAction {
    Flat,
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExitReason {
    SignalReversal,
    TakeProfit,
    StopLoss,
    Liquidation,
    MaxHoldingTime,
    EndOfBacktest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub is_long: bool,
    pub entry_price: f64,
    pub amount: f64,
    pub entry_step: usize,
    pub entry_timestamp: i64,
    pub stop_loss: f64,
    pub take_profit: Option<f64>,
    pub liquidation_price: f64,
    pub initial_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeLog {
    pub trade_id: usize,
    pub is_long: bool,
    pub entry_price: f64,
    pub exit_price: f64,
    pub amount: f64,
    pub entry_step: usize,
    pub exit_step: usize,
    pub entry_timestamp: i64,
    pub exit_timestamp: i64,
    pub gross_pnl: f64,
    pub total_fees: f64,
    pub net_pnl: f64,
    pub return_pct: f64,
    pub exit_reason: ExitReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub leverage: f64,
    pub position_size_pct: f64, // p.ej. 0.10 para 10% del capital
    pub fee_rate: f64,          // p.ej. 0.0005 (0.05%)
    pub max_holding_bars: usize,
    pub use_compound: bool,     // Interés compuesto vs balance nominal fijo
    pub atr_sl_multiplier: f64, // Multiplicador ATR para Stop Loss (ej. 1.5)
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            leverage: 10.0,
            position_size_pct: 0.10,
            fee_rate: 0.0005,
            max_holding_bars: 24,
            use_compound: false,
            atr_sl_multiplier: 1.5,
        }
    }
}
