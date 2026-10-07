use crate::data::db::Kline;
use crate::engine::equation_model::EquationModel;
use crate::engine::types::SignalAction;
use crate::live_trading::types::{LiveSignalEvaluation, SignalType};

pub fn evaluate_live_signal<M: EquationModel>(
    model: &mut M,
    klines: &[Kline],
    threshold_long: f32,
    _threshold_short: f32,
) -> Option<LiveSignalEvaluation> {
    if klines.is_empty() {
        return None;
    }

    let last_kline = klines.last()?;
    let action = model.evaluate(klines);

    let signal = match action {
        SignalAction::Buy => SignalType::Long,
        SignalAction::Sell => SignalType::Short,
        SignalAction::Flat => SignalType::Flat,
    };

    let p_long = if signal == SignalType::Long { 1.0 } else { 0.0 };
    let p_short = if signal == SignalType::Short { 1.0 } else { 0.0 };

    Some(LiveSignalEvaluation {
        signal,
        p_long,
        p_short,
        p_flat: if signal == SignalType::Flat { 1.0 } else { 0.0 },
        dynamic_thr: threshold_long,
        last_price: last_kline.close,
        candle_timestamp: last_kline.timestamp,
    })
}
