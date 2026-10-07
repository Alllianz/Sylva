use crate::data::db::Kline;
use crate::engine::equation_model::EquationModel;
use crate::engine::types::{BacktestConfig, ExitReason, Position, SignalAction, TradeLog};
use crate::features::indicators::calculate_atr;
use crate::metrics::backtest_report::{calculate_metrics, BacktestReport};

/// Motor de Backtesting Causal Ultra-Rápido sin Look-Ahead Bias
pub struct BacktestSimulator {
    pub config: BacktestConfig,
}

impl BacktestSimulator {
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Ejecuta el backtest en la ventana [start_idx..end_idx] usando klines[0..start_idx] como precarga causal
    pub fn run_range<M: EquationModel>(
        &self,
        model: &mut M,
        klines: &[Kline],
        start_idx: usize,
        end_idx: usize,
    ) -> BacktestReport {
        let total_n = klines.len();
        if total_n == 0 || start_idx >= end_idx || start_idx >= total_n {
            return calculate_metrics(
                self.config.initial_capital,
                self.config.initial_capital,
                Vec::new(),
                Vec::new(),
                0,
            );
        }

        let atr_values = calculate_atr(klines, 14);
        let mut capital = self.config.initial_capital;
        let mut peak_capital = capital;
        let mut current_stagnation = 0;
        let mut max_stagnation = 0;

        let mut current_position: Option<Position> = None;
        let mut trades: Vec<TradeLog> = Vec::new();
        let target_end = end_idx.min(total_n);
        let mut equity_curve: Vec<(i64, f64)> = Vec::with_capacity(target_end - start_idx);
        let mut trade_counter = 0;

        for step in start_idx..target_end {
            let kline = &klines[step];
            let price = kline.close;
            let atr = if step < atr_values.len() && atr_values[step] > 0.0 {
                atr_values[step]
            } else {
                price * 0.015
            };

            // 1. --- CHEQUEO DE SALIDAS EN LA VELA ACTUAL ---
            if let Some(pos) = current_position.take() {
                let mut is_closed = false;
                let mut exit_p = 0.0;
                let mut reason = ExitReason::SignalReversal;

                if pos.is_long {
                    // LONG: Liquidación o Stop Loss
                    if kline.open <= pos.liquidation_price {
                        is_closed = true;
                        exit_p = pos.liquidation_price;
                        reason = ExitReason::Liquidation;
                    } else if kline.open <= pos.stop_loss {
                        is_closed = true;
                        exit_p = kline.open;
                        reason = ExitReason::StopLoss;
                    } else if kline.low <= pos.liquidation_price {
                        is_closed = true;
                        exit_p = pos.liquidation_price;
                        reason = ExitReason::Liquidation;
                    } else if kline.low <= pos.stop_loss {
                        is_closed = true;
                        exit_p = pos.stop_loss;
                        reason = ExitReason::StopLoss;
                    } else if let Some(tp) = pos.take_profit {
                        if kline.high >= tp {
                            is_closed = true;
                            exit_p = tp;
                            reason = ExitReason::TakeProfit;
                        }
                    }

                    if !is_closed && step.saturating_sub(pos.entry_step) >= self.config.max_holding_bars {
                        is_closed = true;
                        exit_p = price;
                        reason = ExitReason::MaxHoldingTime;
                    }

                    if is_closed {
                        trade_counter += 1;
                        let entry_fee = pos.entry_price * pos.amount * self.config.fee_rate;
                        let exit_fee = exit_p * pos.amount * self.config.fee_rate;
                        let total_fees = entry_fee + exit_fee;
                        let gross_pnl = (exit_p - pos.entry_price) * pos.amount;
                        let net_pnl = gross_pnl - total_fees;

                        capital += net_pnl;
                        if capital < 0.0 {
                            capital = 0.0;
                        }

                        let ret_pct = if pos.initial_margin > 0.0 {
                            (net_pnl / pos.initial_margin) * 100.0
                        } else {
                            0.0
                        };

                        trades.push(TradeLog {
                            trade_id: trade_counter,
                            is_long: true,
                            entry_price: pos.entry_price,
                            exit_price: exit_p,
                            amount: pos.amount,
                            entry_step: pos.entry_step,
                            exit_step: step,
                            entry_timestamp: pos.entry_timestamp,
                            exit_timestamp: kline.timestamp,
                            gross_pnl,
                            total_fees,
                            net_pnl,
                            return_pct: ret_pct,
                            exit_reason: reason,
                        });
                    } else {
                        current_position = Some(pos);
                    }
                } else {
                    // SHORT: Liquidación o Stop Loss
                    if kline.open >= pos.liquidation_price {
                        is_closed = true;
                        exit_p = pos.liquidation_price;
                        reason = ExitReason::Liquidation;
                    } else if kline.open >= pos.stop_loss {
                        is_closed = true;
                        exit_p = kline.open;
                        reason = ExitReason::StopLoss;
                    } else if kline.high >= pos.liquidation_price {
                        is_closed = true;
                        exit_p = pos.liquidation_price;
                        reason = ExitReason::Liquidation;
                    } else if kline.high >= pos.stop_loss {
                        is_closed = true;
                        exit_p = pos.stop_loss;
                        reason = ExitReason::StopLoss;
                    } else if let Some(tp) = pos.take_profit {
                        if kline.low <= tp {
                            is_closed = true;
                            exit_p = tp;
                            reason = ExitReason::TakeProfit;
                        }
                    }

                    if !is_closed && step.saturating_sub(pos.entry_step) >= self.config.max_holding_bars {
                        is_closed = true;
                        exit_p = price;
                        reason = ExitReason::MaxHoldingTime;
                    }

                    if is_closed {
                        trade_counter += 1;
                        let entry_fee = pos.entry_price * pos.amount * self.config.fee_rate;
                        let exit_fee = exit_p * pos.amount * self.config.fee_rate;
                        let total_fees = entry_fee + exit_fee;
                        let gross_pnl = (pos.entry_price - exit_p) * pos.amount;
                        let net_pnl = gross_pnl - total_fees;

                        capital += net_pnl;
                        if capital < 0.0 {
                            capital = 0.0;
                        }

                        let ret_pct = if pos.initial_margin > 0.0 {
                            (net_pnl / pos.initial_margin) * 100.0
                        } else {
                            0.0
                        };

                        trades.push(TradeLog {
                            trade_id: trade_counter,
                            is_long: false,
                            entry_price: pos.entry_price,
                            exit_price: exit_p,
                            amount: pos.amount,
                            entry_step: pos.entry_step,
                            exit_step: step,
                            entry_timestamp: pos.entry_timestamp,
                            exit_timestamp: kline.timestamp,
                            gross_pnl,
                            total_fees,
                            net_pnl,
                            return_pct: ret_pct,
                            exit_reason: reason,
                        });
                    } else {
                        current_position = Some(pos);
                    }
                }
            }

            // 2. --- EVALUACIÓN CAUSAL DEL MODELO (Solo datos hasta 'step') ---
            let signal = model.evaluate(&klines[0..=step]);

            // 3. --- PROCESAMIENTO DE NUEVAS ENTRADAS O REVERSIONES ---
            match signal {
                SignalAction::Buy => {
                    if let Some(pos) = current_position.take() {
                        if !pos.is_long {
                            // Cerrar Short por reversión
                            trade_counter += 1;
                            let entry_fee = pos.entry_price * pos.amount * self.config.fee_rate;
                            let exit_fee = price * pos.amount * self.config.fee_rate;
                            let total_fees = entry_fee + exit_fee;
                            let gross_pnl = (pos.entry_price - price) * pos.amount;
                            let net_pnl = gross_pnl - total_fees;
                            capital += net_pnl;
                            if capital < 0.0 {
                                capital = 0.0;
                            }
                            let ret_pct = if pos.initial_margin > 0.0 {
                                (net_pnl / pos.initial_margin) * 100.0
                            } else {
                                0.0
                            };
                            trades.push(TradeLog {
                                trade_id: trade_counter,
                                is_long: false,
                                entry_price: pos.entry_price,
                                exit_price: price,
                                amount: pos.amount,
                                entry_step: pos.entry_step,
                                exit_step: step,
                                entry_timestamp: pos.entry_timestamp,
                                exit_timestamp: kline.timestamp,
                                gross_pnl,
                                total_fees,
                                net_pnl,
                                return_pct: ret_pct,
                                exit_reason: ExitReason::SignalReversal,
                            });
                        } else {
                            current_position = Some(pos);
                        }
                    }

                    if current_position.is_none() && capital > 10.0 {
                        let base_capital = if self.config.use_compound {
                            capital
                        } else {
                            self.config.initial_capital
                        };
                        let margin = base_capital * self.config.position_size_pct;
                        let notional = margin * self.config.leverage;
                        let amount = notional / price;

                        let sl_dist = atr * self.config.atr_sl_multiplier;
                        let stop_loss = price - sl_dist;
                        let liq_price = price * (1.0 - (1.0 / self.config.leverage) * 0.95);

                        current_position = Some(Position {
                            is_long: true,
                            entry_price: price,
                            amount,
                            entry_step: step,
                            entry_timestamp: kline.timestamp,
                            stop_loss,
                            take_profit: None,
                            liquidation_price: liq_price,
                            initial_margin: margin,
                        });
                    }
                }
                SignalAction::Sell => {
                    if let Some(pos) = current_position.take() {
                        if pos.is_long {
                            // Cerrar Long por reversión
                            trade_counter += 1;
                            let entry_fee = pos.entry_price * pos.amount * self.config.fee_rate;
                            let exit_fee = price * pos.amount * self.config.fee_rate;
                            let total_fees = entry_fee + exit_fee;
                            let gross_pnl = (price - pos.entry_price) * pos.amount;
                            let net_pnl = gross_pnl - total_fees;
                            capital += net_pnl;
                            if capital < 0.0 {
                                capital = 0.0;
                            }
                            let ret_pct = if pos.initial_margin > 0.0 {
                                (net_pnl / pos.initial_margin) * 100.0
                            } else {
                                0.0
                            };
                            trades.push(TradeLog {
                                trade_id: trade_counter,
                                is_long: true,
                                entry_price: pos.entry_price,
                                exit_price: price,
                                amount: pos.amount,
                                entry_step: pos.entry_step,
                                exit_step: step,
                                entry_timestamp: pos.entry_timestamp,
                                exit_timestamp: kline.timestamp,
                                gross_pnl,
                                total_fees,
                                net_pnl,
                                return_pct: ret_pct,
                                exit_reason: ExitReason::SignalReversal,
                            });
                        } else {
                            current_position = Some(pos);
                        }
                    }

                    if current_position.is_none() && capital > 10.0 {
                        let base_capital = if self.config.use_compound {
                            capital
                        } else {
                            self.config.initial_capital
                        };
                        let margin = base_capital * self.config.position_size_pct;
                        let notional = margin * self.config.leverage;
                        let amount = notional / price;

                        let sl_dist = atr * self.config.atr_sl_multiplier;
                        let stop_loss = price + sl_dist;
                        let liq_price = price * (1.0 + (1.0 / self.config.leverage) * 0.95);

                        current_position = Some(Position {
                            is_long: false,
                            entry_price: price,
                            amount,
                            entry_step: step,
                            entry_timestamp: kline.timestamp,
                            stop_loss,
                            take_profit: None,
                            liquidation_price: liq_price,
                            initial_margin: margin,
                        });
                    }
                }
                SignalAction::Flat => {}
            }

            // 4. --- CALCULAR EQUIDAD DE MERCADO (Mark-To-Market) ---
            let mut unrealized_pnl = 0.0;
            if let Some(ref pos) = current_position {
                if pos.is_long {
                    unrealized_pnl = (price - pos.entry_price) * pos.amount;
                } else {
                    unrealized_pnl = (pos.entry_price - price) * pos.amount;
                }
            }

            let current_equity = (capital + unrealized_pnl).max(0.0);
            equity_curve.push((kline.timestamp, current_equity));

            if current_equity > peak_capital {
                peak_capital = current_equity;
                current_stagnation = 0;
            } else {
                current_stagnation += 1;
                if current_stagnation > max_stagnation {
                    max_stagnation = current_stagnation;
                }
            }
        }

        calculate_metrics(
            self.config.initial_capital,
            capital,
            trades,
            equity_curve,
            max_stagnation,
        )
    }
}
