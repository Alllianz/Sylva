use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Long,
    Short,
    Flat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSignalEvaluation {
    pub signal: SignalType,
    pub p_long: f32,
    pub p_short: f32,
    pub p_flat: f32,
    pub dynamic_thr: f32,
    pub last_price: f64,
    pub candle_timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountBalanceInfo {
    pub id: u32,
    pub timeframe: String,
    pub api_key: String,
    pub api_secret: String,
    pub leverage: u32,
    pub exchange: String,
    pub use_testnet: bool,
    pub wallet_balance: f64,
    pub available_margin: f64,
    pub user_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedAccountPool {
    pub accounts: Vec<AccountBalanceInfo>,
    pub total_wallet_balance: f64,
    pub total_available_margin: f64,
    pub last_updated: i64,
}

impl UnifiedAccountPool {
    pub fn new() -> Self {
        Self {
            accounts: Vec::new(),
            total_wallet_balance: 0.0,
            total_available_margin: 0.0,
            last_updated: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.accounts.is_empty()
    }
}

impl Default for UnifiedAccountPool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedPositionState {
    pub timeframe: String,
    pub direction: String, // "LONG", "SHORT"
    pub entry_price: f64,
    pub amount: f64,
    pub leverage: f64,
    pub capital_percent: f64,
    pub open_timestamp_ms: i64,
    pub entry_time_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedTradeRecord {
    pub timeframe: String,
    pub direction: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub entry_time_utc: String,
    pub exit_time_utc: String,
    pub leverage: f64,
    pub pnl_percent: f64,
    pub roe_percent: f64,
    pub capital_percent: f64,
    pub pnl_usd: f64,
    pub total_fees: f64,
    pub capital_before: f64,
    pub capital_after: f64,
    pub is_win: bool,
    pub account_id: u32,
}
