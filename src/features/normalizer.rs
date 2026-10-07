use crate::data::db::Kline;
use crate::features::cached::CachedIndicators;
use crate::features::mask::FeatureMask;
use crate::features::names::TOTAL_FEATURES;
use crate::features::raw::compute_raw_features;

/// Normalizador de Características Cuantitativas mediante Z-Score Rodante Causal (0% Look-Ahead Bias)
pub struct FeatureNormalizer {
    pub rolling_window: usize,
    pub current_position_val: f32,
    pub mask: Option<FeatureMask>,
}

impl FeatureNormalizer {
    pub fn new(rolling_window: usize) -> Self {
        Self {
            rolling_window: rolling_window.max(20),
            current_position_val: 0.0,
            mask: None,
        }
    }

    pub fn with_mask(mut self, mask: FeatureMask) -> Self {
        self.mask = Some(mask);
        self
    }

    pub fn set_position_state(&mut self, pos_val: f32) {
        self.current_position_val = pos_val;
    }

    /// Calcula las 32 características normalizadas en la barra `t` usando solo la ventana rodante causal `[t-W+1 ..= t]`
    pub fn normalize(
        &self,
        klines: &[Kline],
        t: usize,
        cached: &CachedIndicators,
    ) -> [f32; TOTAL_FEATURES] {
        if t < self.rolling_window || t >= klines.len() {
            return [0.0f32; TOTAL_FEATURES];
        }

        let start_w = t + 1 - self.rolling_window;
        let mut raw_history = Vec::with_capacity(self.rolling_window);

        for step in start_w..=t {
            let feat = compute_raw_features(klines, step, cached, self.current_position_val);
            raw_history.push(feat);
        }

        let mut normalized = [0.0f32; TOTAL_FEATURES];
        let inv_w = 1.0 / (self.rolling_window as f32);

        for col in 0..TOTAL_FEATURES {
            // Si la feature está podada por la máscara, queda en 0.0
            if let Some(ref m) = self.mask {
                if !m.is_active(col) {
                    normalized[col] = 0.0;
                    continue;
                }
            }

            // No normalizar el canal de estado comercial (F32) porque ya está en {-1, 0, 1}
            if col == TOTAL_FEATURES - 1 {
                normalized[col] = raw_history.last().unwrap()[col];
                continue;
            }

            let mut sum = 0.0f32;
            for row in &raw_history {
                sum += row[col];
            }
            let mean = sum * inv_w;

            let mut sum_sq_diff = 0.0f32;
            for row in &raw_history {
                let diff = row[col] - mean;
                sum_sq_diff += diff * diff;
            }
            let mut std_dev = (sum_sq_diff * inv_w).sqrt();
            if std_dev < 1e-6 {
                std_dev = 1.0;
            }

            let current_raw = raw_history.last().unwrap()[col];
            normalized[col] = (current_raw - mean) / std_dev;
        }

        normalized
    }
}
