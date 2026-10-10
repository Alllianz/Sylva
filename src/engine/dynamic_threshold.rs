use crate::features::CachedIndicators;
use serde::{Deserialize, Serialize};

/// Modalidad de umbrales para disparo de señales cuantitativas
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdMode {
    /// Umbrales estáticos fijos calibrados exclusivamente en In-Sample
    Static,
    /// Umbrales dinámicos modulados por el ratio de volatilidad causal ATR5 / ATR50
    DynamicAtrRatio,
}

impl Default for ThresholdMode {
    fn default() -> Self {
        ThresholdMode::DynamicAtrRatio
    }
}

/// Calcula el factor de escala de volatilidad causal en el índice `curr_idx`.
/// Ratio de expansión/compresión: ATR(5) / ATR(50).
/// Acotado estrictamente entre [0.5, 2.5] para estabilidad matemática (0% Look-Ahead Bias).
#[inline(always)]
pub fn calculate_volatility_factor(
    curr_idx: usize,
    cached: Option<&CachedIndicators>,
) -> f32 {
    if let Some(c) = cached {
        if curr_idx < c.atr_5.len() && curr_idx < c.atr_50.len() {
            let a5 = c.atr_5[curr_idx];
            let a50 = c.atr_50[curr_idx];
            if a50 > 1e-6 {
                let ratio = (a5 / a50) as f32;
                return ratio.clamp(0.5, 2.5);
            }
        }
    }
    1.0
}
