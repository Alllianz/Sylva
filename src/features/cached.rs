use crate::data::db::Kline;
use crate::features::indicators::{
    calculate_atr, calculate_ema, calculate_ema_200_hl2, calculate_macd_hist,
    calculate_rsi, calculate_volume_sma_10000, calculate_volume_smas,
};

#[derive(Clone, Debug)]
pub struct CachedIndicators {
    pub volume_sma_10000: Vec<f64>,
    pub volume_sma_5: Vec<f64>,
    pub volume_sma_20: Vec<f64>,
    pub macd_hist: Vec<f64>,
    pub ema_20: Vec<f64>,
    pub ema_200_hl2: Vec<f64>,
    pub atr_5: Vec<f64>,
    pub atr_10: Vec<f64>,
    pub atr_50: Vec<f64>,
    pub rsi_14: Vec<f64>,
}

impl CachedIndicators {
    pub fn new(klines: &[Kline]) -> Self {
        let volume_sma_10000 = calculate_volume_sma_10000(klines);
        let (volume_sma_5, volume_sma_20) = calculate_volume_smas(klines);
        let macd_hist = calculate_macd_hist(klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let ema_20 = calculate_ema(&closes, 20);
        let ema_200_hl2 = calculate_ema_200_hl2(klines);
        let atr_5 = calculate_atr(klines, 5);
        let atr_10 = calculate_atr(klines, 10);
        let atr_50 = calculate_atr(klines, 50);
        let rsi_14 = calculate_rsi(klines, 14);

        Self {
            volume_sma_10000,
            volume_sma_5,
            volume_sma_20,
            macd_hist,
            ema_20,
            ema_200_hl2,
            atr_5,
            atr_10,
            atr_50,
            rsi_14,
        }
    }
}
