use crate::data::db::Kline;
use crate::engine::dynamic_threshold::{calculate_volatility_factor, ThresholdMode};
use crate::engine::equation_model::EquationModel;
use crate::engine::types::SignalAction;
use crate::features::cached::CachedIndicators;
use crate::features::mask::FeatureMask;
use crate::features::names::TOTAL_FEATURES;
use crate::features::normalizer::FeatureNormalizer;
use crate::trees::online::gbdt::OnlineGbdtModel;
use std::sync::Arc;

/// Envoltorio de Online GBDT para el motor de backtesting y trading en vivo con aprendizaje continuo
pub struct OnlineGbdtEquationModel {
    pub model: OnlineGbdtModel,
    pub threshold_long: f32,
    pub threshold_short: f32,
    pub rolling_window: usize,
    pub current_position_val: f32,
    pub feature_mask: Option<FeatureMask>,
    pub enable_online_learning: bool,
    pub prev_features: Option<[f32; TOTAL_FEATURES]>,
    pub prev_close: Option<f64>,
    pub last_predicted_alpha: f32,
    pub cached_indicators: Option<Arc<CachedIndicators>>,
    pub threshold_mode: ThresholdMode,
}

impl OnlineGbdtEquationModel {
    pub fn new(
        model: OnlineGbdtModel,
        threshold_long: f32,
        threshold_short: f32,
        rolling_window: usize,
    ) -> Self {
        Self {
            model,
            threshold_long,
            threshold_short,
            rolling_window: rolling_window.max(20),
            current_position_val: 0.0,
            feature_mask: None,
            enable_online_learning: true,
            prev_features: None,
            prev_close: None,
            last_predicted_alpha: 0.0,
            cached_indicators: None,
            threshold_mode: ThresholdMode::default(),
        }
    }

    pub fn with_mask(mut self, mask: FeatureMask) -> Self {
        self.feature_mask = Some(mask);
        self
    }

    pub fn with_online_learning(mut self, enabled: bool) -> Self {
        self.enable_online_learning = enabled;
        self
    }

    pub fn with_cached_indicators(mut self, cached: Arc<CachedIndicators>) -> Self {
        self.cached_indicators = Some(cached);
        self
    }

    pub fn with_threshold_mode(mut self, mode: ThresholdMode) -> Self {
        self.threshold_mode = mode;
        self
    }
}

impl EquationModel for OnlineGbdtEquationModel {
    fn name(&self) -> &str {
        "Online GBDT (Streaming Decision Trees - Adaptive Boosting)"
    }

    fn evaluate(&mut self, history: &[Kline]) -> SignalAction {
        let n = history.len();
        if n < self.rolling_window {
            return SignalAction::Flat;
        }

        let curr_idx = n - 1;
        let curr_kline = &history[curr_idx];
        let curr_close = curr_kline.close;

        // 1. --- APRENDIZAJE ONLINE CAUSAL (0% LOOK-AHEAD BIAS) ---
        // Al cerrar la vela en t, se revela el retorno real de la predicción efectuada en t-1
        if self.enable_online_learning {
            if let (Some(prev_feats), Some(prev_c)) = (self.prev_features, self.prev_close) {
                if prev_c > 0.0 {
                    let actual_return = ((curr_close - prev_c) / prev_c) as f32;
                    self.model.update_step(&prev_feats, actual_return);
                }
            }
        }

        // 2. --- EXTRACCIÓN CAUSAL DE CARACTERÍSTICAS PARA t ---
        let normalizer = FeatureNormalizer::new(self.rolling_window);
        let normalizer = if let Some(ref m) = self.feature_mask {
            normalizer.with_mask(m.clone())
        } else {
            normalizer
        };

        let feats = if let Some(ref cached) = self.cached_indicators {
            normalizer.normalize(history, curr_idx, cached)
        } else {
            let cached = CachedIndicators::new(history);
            normalizer.normalize(history, curr_idx, &cached)
        };

        // 3. --- INFERENCIA ADAPTATIVA ONLINE ---
        let predicted_alpha = self.model.predict(&feats);
        self.last_predicted_alpha = predicted_alpha;

        // Guardar estado causal para el siguiente paso streaming
        self.prev_features = Some(feats);
        self.prev_close = Some(curr_close);

        // 4. --- DISPARO DE SEÑALES (CON UMBRALES ADAPTATIVOS POR VOLATILIDAD) ---
        let vol_factor = match self.threshold_mode {
            ThresholdMode::DynamicAtrRatio => calculate_volatility_factor(curr_idx, self.cached_indicators.as_deref()),
            ThresholdMode::Static => 1.0,
        };
        let eff_threshold_long = self.threshold_long * vol_factor;
        let eff_threshold_short = self.threshold_short * vol_factor;

        if predicted_alpha > eff_threshold_long {
            self.current_position_val = 1.0;
            SignalAction::Buy
        } else if predicted_alpha < eff_threshold_short {
            self.current_position_val = -1.0;
            SignalAction::Sell
        } else {
            SignalAction::Flat
        }
    }

    fn reset(&mut self) {
        self.current_position_val = 0.0;
        self.prev_features = None;
        self.prev_close = None;
        self.last_predicted_alpha = 0.0;
    }
}
