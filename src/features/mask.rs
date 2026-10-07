use crate::features::names::TOTAL_FEATURES;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureMask {
    pub active_mask: Vec<bool>,
}

impl Default for FeatureMask {
    fn default() -> Self {
        Self {
            active_mask: vec![true; TOTAL_FEATURES],
        }
    }
}

impl FeatureMask {
    pub fn new_all_active() -> Self {
        Self {
            active_mask: vec![true; TOTAL_FEATURES],
        }
    }

    pub fn from_mask(mask: [bool; TOTAL_FEATURES]) -> Self {
        Self {
            active_mask: mask.to_vec(),
        }
    }

    pub fn new_microstructure_5() -> Self {
        let mut mask = vec![false; TOTAL_FEATURES];
        for i in 0..5 {
            mask[i] = true;
        }
        Self { active_mask: mask }
    }

    pub fn new_30_features() -> Self {
        let mut mask = vec![false; TOTAL_FEATURES];
        for i in 0..30.min(TOTAL_FEATURES) {
            mask[i] = true;
        }
        Self { active_mask: mask }
    }

    pub fn new_32_features() -> Self {
        let mut mask = vec![false; TOTAL_FEATURES];
        for i in 0..32.min(TOTAL_FEATURES) {
            mask[i] = true;
        }
        Self { active_mask: mask }
    }

    /// Conjunto de 13 características activas seleccionadas para GBDT Allianz.
    /// Poda permanente de 19 variables:
    /// Podadas (19): F1, F2, F3, F7, F8, F12, F15, F19, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F32.
    /// Activas (13): F4, F5, F6, F9, F10, F11, F13, F14, F16, F17, F18, F20, F31.
    pub fn new_allianz_custom() -> Self {
        let mut mask = vec![true; TOTAL_FEATURES];
        let pruned_indices = [
            0,  // F1: Long-term Rel Volume (SMA 10k)
            1,  // F2: Normalized Total Range
            2,  // F3: Upper Wick Ratio
            6,  // F7: OLS Linear Slope (10 Bars)
            7,  // F8: Donchian Stochastic Pos (10 Bars)
            11, // F12: Markov Current State (St / 5.0)
            14, // F15: Dist to Macro EMA-200 (HL/2)
            18, // F19: Volume Adjusted Amplitude
            20, // F21: Directional Signed Volume Force
            21, // F22: ATR(10) Normalized / Close
            22, // F23: MACD Histogram / Close
            23, // F24: Volatility Ratio (ATR5 / ATR50)
            24, // F25: Channel Width (10 Bars)
            25, // F26: Intra-Candle Volatility ROC
            26, // F27: RSI-14 Centered ([-0.5, +0.5])
            27, // F28: Parkinson Extreme Volatility
            28, // F29: Weekly Cycle Sine
            29, // F30: Weekly Cycle Cosine
            31, // F32: Account Commercial Position State
        ];
        for &idx in &pruned_indices {
            if idx < TOTAL_FEATURES {
                mask[idx] = false;
            }
        }
        Self { active_mask: mask }
    }

    #[inline(always)]
    pub fn is_active(&self, idx: usize) -> bool {
        if idx < TOTAL_FEATURES {
            self.active_mask[idx]
        } else {
            false
        }
    }

    pub fn set_active(&mut self, idx: usize, active: bool) {
        if idx < TOTAL_FEATURES {
            self.active_mask[idx] = active;
        }
    }

    pub fn active_count(&self) -> usize {
        self.active_mask.iter().filter(|&&a| a).count()
    }

    pub fn active_indices(&self) -> Vec<usize> {
        self.active_mask
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a)
            .map(|(i, _)| i)
            .collect()
    }
}
