use crate::data::db::Kline;
use crate::engine::equation_model::EquationModel;
use crate::engine::types::SignalAction;
use crate::features::cached::CachedIndicators;
use crate::features::mask::FeatureMask;
use crate::features::normalizer::FeatureNormalizer;
use crate::trees::linear::elastic_net::ElasticNetModel;
use std::sync::Arc;

/// Envoltorio de Elastic Net para el motor de backtesting y simulación continua
pub struct ElasticNetEquationModel {
    pub model: ElasticNetModel,
    pub threshold_long: f32,
    pub threshold_short: f32,
    pub rolling_window: usize,
    pub current_position_val: f32,
    pub feature_mask: Option<FeatureMask>,
    pub cached_indicators: Option<Arc<CachedIndicators>>,
}

impl ElasticNetEquationModel {
    pub fn new(
        model: ElasticNetModel,
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
            cached_indicators: None,
        }
    }

    pub fn with_mask(mut self, mask: FeatureMask) -> Self {
        self.feature_mask = Some(mask);
        self
    }

    pub fn with_cached_indicators(mut self, cached: Arc<CachedIndicators>) -> Self {
        self.cached_indicators = Some(cached);
        self
    }
}

impl EquationModel for ElasticNetEquationModel {
    fn name(&self) -> &str {
        "Elastic Net Regularized (L1 Lasso + L2 Ridge)"
    }

    fn evaluate(&mut self, history: &[Kline]) -> SignalAction {
        let n = history.len();
        if n < self.rolling_window {
            return SignalAction::Flat;
        }

        let curr_idx = n - 1;
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

        let predicted_return = self.model.predict(&feats);

        if predicted_return > self.threshold_long {
            self.current_position_val = 1.0;
            SignalAction::Buy
        } else if predicted_return < self.threshold_short {
            self.current_position_val = -1.0;
            SignalAction::Sell
        } else {
            SignalAction::Flat
        }
    }

    fn reset(&mut self) {
        self.current_position_val = 0.0;
    }
}
