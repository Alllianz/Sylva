use crate::data::db::Kline;
use crate::engine::types::SignalAction;

/// Trait común para cualquier modelo o sistema de predicción causal sobre árboles o modelos tabulares.
/// Garantiza estricta causalidad: solo recibe el slice de klines disponible hasta la vela en curso.
pub trait EquationModel: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&mut self, history: &[Kline]) -> SignalAction;
    fn reset(&mut self) {}
}
