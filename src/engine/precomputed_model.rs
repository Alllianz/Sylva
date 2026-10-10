use crate::data::db::Kline;
use crate::engine::equation_model::EquationModel;
use crate::engine::types::SignalAction;
use std::sync::Arc;

/// Modelo de inferencia ultra-rápido basado en vector continuo de señales Alpha precomputadas (en GPU o CPU)
/// Elimina al 100% el re-cálculo redundante de variables y el recorrido del árbol durante la calibración de umbrales.
#[derive(Clone, Debug)]
pub struct PrecomputedAlphaModel {
    pub alphas: Arc<Vec<f32>>,
    pub start_step: usize,
    pub threshold_long: f32,
    pub threshold_short: f32,
    pub vol_ratios: Option<Arc<Vec<f32>>>,
}

impl PrecomputedAlphaModel {
    pub fn new(
        alphas: Arc<Vec<f32>>,
        start_step: usize,
        threshold_long: f32,
        threshold_short: f32,
    ) -> Self {
        Self {
            alphas,
            start_step,
            threshold_long,
            threshold_short,
            vol_ratios: None,
        }
    }

    pub fn with_vol_ratios(mut self, vol_ratios: Arc<Vec<f32>>) -> Self {
        self.vol_ratios = Some(vol_ratios);
        self
    }
}

impl EquationModel for PrecomputedAlphaModel {
    fn name(&self) -> &str {
        "Precomputed Alpha Model (Zero Overhead Sweeper)"
    }

    #[inline(always)]
    fn evaluate(&mut self, history: &[Kline]) -> SignalAction {
        let n = history.len();
        if n == 0 {
            return SignalAction::Flat;
        }

        let curr_step = n - 1;
        if curr_step < self.start_step {
            return SignalAction::Flat;
        }

        let rel_idx = curr_step - self.start_step;
        if rel_idx < self.alphas.len() {
            let alpha = self.alphas[rel_idx];
            let factor = match &self.vol_ratios {
                Some(vr) if rel_idx < vr.len() => vr[rel_idx],
                _ => 1.0,
            };
            let eff_long = self.threshold_long * factor;
            let eff_short = self.threshold_short * factor;

            if alpha > eff_long {
                SignalAction::Buy
            } else if alpha < eff_short {
                SignalAction::Sell
            } else {
                SignalAction::Flat
            }
        } else {
            SignalAction::Flat
        }
    }
}
