use crate::dashboard::grid_dashboard::{generate_gbdt_grid_dashboard, GbdtGridCandidateReport, GbdtGridDashboardSummary};
use crate::data::db::Kline;
use crate::engine::simulator::BacktestSimulator;
use crate::engine::threshold_tuner::ThresholdTuner;
use crate::engine::types::BacktestConfig;
use crate::features::FeatureMask;
use crate::metrics::autotuning::{calculate_capital_max_dd_ratio, calculate_slope_ratio, evaluate_equity_linearity};
use crate::metrics::backtest_report::BacktestReport;
use crate::metrics::ranking::compute_multicriteria_rankings;
use crate::metrics::smoothness::calculate_equity_smoothness;
use crate::trees::batch::gbdt::{GbdtConfig, GbdtModel, GbdtTrainer};
use crate::trees::batch::model_wrapper::GbdtEquationModel;
use crate::trees::batch::purged_cv::{evaluate_predictions, PurgedCrossValidator};
use crate::trees::dataset::TabularDataset;
use std::error::Error;

/// Configuración del Auto-Tuner de GBDT
#[derive(Clone, Debug)]
pub struct GbdtAutoTuningConfig {
    pub n_folds: usize,
    pub embargo_pct: f32,
    pub rolling_window: usize,
    pub target_horizon: usize,
    pub candidate_configs: Vec<GbdtConfig>,
}

impl Default for GbdtAutoTuningConfig {
    fn default() -> Self {
        // Espacio de búsqueda balanceado para máxima velocidad y prevención de sobreajuste
        let depths = [2, 3, 4, 5, 6];
        let min_leaves = [15, 25, 40, 60];
        let n_trees_list = [25, 40, 60];
        let lrs = [0.02, 0.03, 0.05];
        let subsamples = [0.75, 0.85];

        let mut configs = Vec::new();
        for &d in &depths {
            for &leaf in &min_leaves {
                for &trees in &n_trees_list {
                    for &lr in &lrs {
                        for &sub in &subsamples {
                            // Filtro de consistencia: si profundidad es 2 o 3, learning rate de 0.03 o 0.05
                            // Si profundidad es 5 o 6, menor learning rate y más regularización
                            if d >= 5 && lr > 0.03 {
                                continue;
                            }
                            configs.push(GbdtConfig {
                                n_trees: trees,
                                max_depth: d,
                                learning_rate: lr,
                                l2_reg: if d >= 4 { 2.0 } else { 1.0 },
                                l1_reg: 0.01,
                                min_samples_leaf: leaf,
                                subsample: sub,
                                colsample_bytree: if d >= 4 { 0.7 } else { 0.8 },
                                gamma: 0.0001,
                            });
                        }
                    }
                }
            }
        }

        Self {
            n_folds: 5,
            embargo_pct: 0.01,
            rolling_window: 100,
            target_horizon: 1,
            candidate_configs: configs,
        }
    }
}

/// Resultado de la auto-optimización
pub struct GbdtAutoTuningResult {
    pub champion_model: GbdtModel,
    pub champion_config: GbdtConfig,
    pub champion_thr_long: f32,
    pub champion_thr_short: f32,
    pub champion_is_fitness: f64,
    pub champion_oof_fitness: f64,
    pub candidate_reports: Vec<GbdtGridCandidateReport>,
    pub total_evaluated: usize,
}

pub struct GbdtAutoTuner {
    pub config: GbdtAutoTuningConfig,
    pub backtest_config_nom: BacktestConfig,
    pub backtest_config_pct: BacktestConfig,
}

impl GbdtAutoTuner {
    pub fn new(
        config: GbdtAutoTuningConfig,
        backtest_config_nom: BacktestConfig,
        backtest_config_pct: BacktestConfig,
    ) -> Self {
        Self {
            config,
            backtest_config_nom,
            backtest_config_pct,
        }
    }

    /// Ejecuta el proceso autónomo de auto-optimización anti-degradación
    pub fn auto_optimize(
        &self,
        klines: &[Kline],
        is_dataset: &TabularDataset,
        is_start_idx: usize,
        oos_start_idx: usize,
        tf: &str,
        mask: Option<&FeatureMask>,
    ) -> Result<GbdtAutoTuningResult, Box<dyn Error + Send + Sync>> {
        let colors = [
            "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899", 
            "#06b6d4", "#f97316", "#14b8a6", "#6366f1", "#a855f7", 
            "#84cc16", "#ef4444", "#eab308", "#0ea5e9", "#d946ef", 
            "#22c55e", "#64748b", "#fb7185", "#38bdf8", "#4ade80"
        ];

        let cv = PurgedCrossValidator::new(self.config.n_folds, self.config.embargo_pct);
        let splits = cv.split(is_dataset.len(), self.config.target_horizon);
        let tuner = ThresholdTuner::new(self.backtest_config_nom.clone());
        let sim_nom = BacktestSimulator::new(self.backtest_config_nom.clone());
        let sim_pct = BacktestSimulator::new(self.backtest_config_pct.clone());

        let total_candidates = self.config.candidate_configs.len();
        let feat_info = if let Some(m) = mask {
            format!("MÁSCARA PERSONALIZADA ({} Variables Activas)", m.active_count())
        } else {
            "TODAS LAS 34 VARIABLES CUANTITATIVAS".to_string()
        };

        println!(
            "\n=== 🧬 MOTOR DE AUTO-OPTIMIZACIÓN AUTÓNOMA GBDT [{}] ===",
            feat_info
        );
        println!(
            "  • Explorando {} arquitecturas de árboles en validación cruzada depurada (Purged 5-Fold CV + 1% Embargo)...",
            total_candidates
        );
        println!("  • Optimizando simultáneamente por Sylva EVO Fitness y consistencia entre folds.\n");

        #[derive(Clone)]
        struct CandidateEval {
            config: GbdtConfig,
            cv_mse: f32,
            cv_mda: f32,
            cv_ic: f32,
            _mse_std: f32,
            _anti_degradation_score: f64,
            is_fitness: f64,
            thr_long: f32,
            thr_short: f32,
            model: GbdtModel,
            report_nom: BacktestReport,
            report_pct: BacktestReport,
        }

        let mut evaluated_candidates: Vec<CandidateEval> = Vec::with_capacity(total_candidates);

        for (idx, gbdt_cfg) in self.config.candidate_configs.iter().enumerate() {
            let mut fold_mses = Vec::with_capacity(splits.len());
            let mut fold_mdas = Vec::with_capacity(splits.len());
            let mut fold_ics = Vec::with_capacity(splits.len());

            for split in &splits {
                let fold_train = is_dataset.subset(&split.train_indices);
                let fold_test = is_dataset.subset(&split.test_indices);

                let trainer = GbdtTrainer::new(gbdt_cfg.clone());
                let fold_model = trainer.fit(&fold_train, mask)?;
                let preds: Vec<f32> = fold_test
                    .samples
                    .iter()
                    .map(|s| fold_model.predict(&s.features))
                    .collect();
                let targets: Vec<f32> = fold_test.samples.iter().map(|s| s.target).collect();
                let (mse, mda, rank_ic) = evaluate_predictions(&preds, &targets);

                fold_mses.push(mse);
                fold_mdas.push(mda);
                fold_ics.push(rank_ic);
            }

            let mean_mse: f32 = fold_mses.iter().sum::<f32>() / fold_mses.len() as f32;
            let mean_mda: f32 = (fold_mdas.iter().sum::<f32>() / fold_mdas.len() as f32) * 100.0;
            let mean_ic: f32 = fold_ics.iter().sum::<f32>() / fold_ics.len() as f32;

            // Varianza entre folds (Medida de Estabilidad Anti-Degradación)
            let var_mse: f32 = fold_mses
                .iter()
                .map(|&m| (m - mean_mse).powi(2))
                .sum::<f32>()
                / fold_mses.len() as f32;
            let std_mse = var_mse.sqrt();

            // Ajuste sobre In-Sample completo y calibración de umbrales con Sylva EVO
            let full_trainer = GbdtTrainer::new(gbdt_cfg.clone());
            let final_gbdt = full_trainer.fit(is_dataset, mask)?;

            let tune_res = tuner.tune(
                klines,
                is_start_idx,
                oos_start_idx,
                &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                |thr_l, thr_s| {
                    let mut m = GbdtEquationModel::new(
                        final_gbdt.clone(),
                        thr_l,
                        thr_s,
                        self.config.rolling_window,
                    );
                    if let Some(fmask) = mask {
                        m = m.with_mask(fmask.clone());
                    }
                    m
                },
            );

            let mut model_nom = GbdtEquationModel::new(
                final_gbdt.clone(),
                tune_res.best_threshold_long,
                tune_res.best_threshold_short,
                self.config.rolling_window,
            );
            let mut model_pct = GbdtEquationModel::new(
                final_gbdt.clone(),
                tune_res.best_threshold_long,
                tune_res.best_threshold_short,
                self.config.rolling_window,
            );
            if let Some(fmask) = mask {
                model_nom = model_nom.with_mask(fmask.clone());
                model_pct = model_pct.with_mask(fmask.clone());
            }

            let rep_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, klines.len());
            let rep_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, klines.len());

            // Score Anti-Degradación: Premia alto Fitness y penaliza alta inestabilidad de MSE
            let stability_penalty = (std_mse / (mean_mse + 1e-6)).min(0.5) as f64;
            let anti_degradation_score = tune_res.best_fitness * (1.0 - stability_penalty);

            if (idx + 1) % 10 == 0 || idx == 0 || idx == total_candidates - 1 {
                println!(
                    "  • Evaluando [{:>3}/{}]: Depth:{:>2} | Leaf:{:>2} | Trees:{:>2} | LR:{:.2} ➔ CV MSE:{:.6} | MDA:{:5.2}% | IS Fitness:{:>6.2} | Score Anti-Deg:{:>6.2}",
                    idx + 1, total_candidates, gbdt_cfg.max_depth, gbdt_cfg.min_samples_leaf, gbdt_cfg.n_trees, gbdt_cfg.learning_rate, mean_mse, mean_mda, tune_res.best_fitness, anti_degradation_score
                );
            }

            evaluated_candidates.push(CandidateEval {
                config: gbdt_cfg.clone(),
                cv_mse: mean_mse,
                cv_mda: mean_mda,
                cv_ic: mean_ic,
                _mse_std: std_mse,
                _anti_degradation_score: anti_degradation_score,
                is_fitness: tune_res.best_fitness,
                thr_long: tune_res.best_threshold_long,
                thr_short: tune_res.best_threshold_short,
                model: final_gbdt,
                report_nom: rep_nom,
                report_pct: rep_pct,
            });
        }

        // 1. Ranking Multicriterio por Puestos de Sylva EVO Max (Sin Look-Ahead Bias)
        let rankings = compute_multicriteria_rankings(
            &evaluated_candidates,
            |c| format!("{}-{}-{}-{:.4}", c.config.max_depth, c.config.min_samples_leaf, c.config.n_trees, c.config.learning_rate),
            |c| {
                let mid_is = (c.report_nom.equity_curve.len() / 2).max(1);
                let half1 = c.report_nom.equity_curve[mid_is - 1].1 - c.report_nom.initial_capital;
                let half2 = c.report_nom.final_capital - c.report_nom.equity_curve[mid_is - 1].1;
                let s1 = half1 / (mid_is as f64);
                let s2 = half2 / ((c.report_nom.equity_curve.len() - mid_is) as f64);
                calculate_slope_ratio(s1, s2)
            },
            |c| calculate_capital_max_dd_ratio(c.report_nom.final_capital, c.report_nom.max_drawdown_amount),
            |c| evaluate_equity_linearity(&c.report_nom.equity_curve, oos_start_idx, is_start_idx).composite_linearity_score,
            |c| {
                let lin = evaluate_equity_linearity(&c.report_nom.equity_curve, oos_start_idx, is_start_idx);
                let smooth = calculate_equity_smoothness(&c.report_nom.equity_curve, lin.composite_linearity_score);
                smooth.smoothness_score
            },
        );

        // 2. Construir reportes para el dashboard con la información de Puestos
        let mut candidate_reports = Vec::with_capacity(evaluated_candidates.len());
        for (i, cand) in evaluated_candidates.iter().enumerate() {
            let key = format!("{}-{}-{}-{:.4}", cand.config.max_depth, cand.config.min_samples_leaf, cand.config.n_trees, cand.config.learning_rate);
            let (r_slope, r_cap_dd, r_r2, r_smooth, weighted_rank, sylva_rank) = if let Some(r) = rankings.get(&key) {
                (r.rank_slope, r.rank_cap_dd, r.rank_r2, r.rank_smoothness, r.weighted_avg_rank, r.sylva_rank_fitness)
            } else {
                (999, 999, 999, 999, 999.0, 0.0)
            };

            candidate_reports.push(GbdtGridCandidateReport {
                max_depth: cand.config.max_depth,
                min_samples_leaf: cand.config.min_samples_leaf,
                n_trees: cand.config.n_trees,
                target_horizon: self.config.target_horizon,
                cv_mse: cand.cv_mse,
                cv_mda: cand.cv_mda,
                cv_ic: cand.cv_ic,
                best_thr_long: cand.thr_long,
                best_thr_short: cand.thr_short,
                is_sylva_fitness: cand.is_fitness,
                rank_slope: r_slope,
                rank_cap_dd: r_cap_dd,
                rank_r2: r_r2,
                rank_smoothness: r_smooth,
                weighted_avg_rank: weighted_rank,
                sylva_rank_fitness: sylva_rank,
                raw_slope_ratio: 0.0,
                raw_cap_dd_ratio: 0.0,
                raw_r2_score: 0.0,
                raw_smoothness_score: 0.0,
                report_nom: cand.report_nom.clone(),
                report_pct: cand.report_pct.clone(),
                color: colors[i % colors.len()].to_string(),
            });
        }

        // 3. Ordenar candidatos por Puesto Ponderado de Sylva EVO ascendente (Menor promedio de puesto = Mejor arquitectura)
        candidate_reports.sort_by(|a, b| {
            a.weighted_avg_rank
                .partial_cmp(&b.weighted_avg_rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!("\n=======================================================================================================================================");
        println!("                         🏆 LEADERBOARD DE AUTO-OPTIMIZACIÓN GBDT (RANKING POR PUESTOS SYLVA EVO MAX)                         ");
        println!("=======================================================================================================================================");
        println!(
            "  {:<5} | {:<5} | {:<10} | {:<5} | {:<5} | {:>10} | {:>12} | {:>12} | {:>10} | {:>10} | {:>10}",
            "Puesto", "Depth", "MinSamples", "Trees", "LR", "Rank Pond.", "Rank R² (2x)", "Rank Slope", "Rank Cap/DD", "Rank Smooth", "Net Profit"
        );
        println!("---------------------------------------------------------------------------------------------------------------------------------------");

        for (rank, cand) in candidate_reports.iter().take(10).enumerate() {
            let medal = match rank {
                0 => "🥇 1º",
                1 => "🥈 2º",
                2 => "🥉 3º",
                _ => "   ",
            };
            println!(
                "  {:<2} {} | {:>5} | {:>10} | {:>5} | {:>5.2} | {:>10.2} | {:>12} | {:>12} | {:>10} | {:>10} | ${:>+9.2}",
                rank + 1, medal, cand.max_depth, cand.min_samples_leaf, cand.n_trees, 0.03, cand.weighted_avg_rank, format!("#{}", cand.rank_r2), format!("#{}", cand.rank_slope), format!("#{}", cand.rank_cap_dd), format!("#{}", cand.rank_smoothness), cand.report_nom.net_profit
            );
        }
        println!("=======================================================================================================================================");

        let champion = &candidate_reports[0];

        // Buscar el modelo entrenado del campeón
        let champ_model = evaluated_candidates
            .iter()
            .find(|c| c.config.max_depth == champion.max_depth && c.config.min_samples_leaf == champion.min_samples_leaf && c.config.n_trees == champion.n_trees)
            .map(|c| c.model.clone())
            .unwrap_or_else(|| evaluated_candidates[0].model.clone());

        let champ_cfg = evaluated_candidates
            .iter()
            .find(|c| c.config.max_depth == champion.max_depth && c.config.min_samples_leaf == champion.min_samples_leaf && c.config.n_trees == champion.n_trees)
            .map(|c| c.config.clone())
            .unwrap_or_else(|| evaluated_candidates[0].config.clone());

        let summary_grid = GbdtGridDashboardSummary {
            tf: tf.to_string(),
            is_start_idx,
            oos_start_idx,
            initial_capital: self.backtest_config_nom.initial_capital,
            candidates: candidate_reports.clone(),
        };
        let _ = generate_gbdt_grid_dashboard(&summary_grid, klines);

        Ok(GbdtAutoTuningResult {
            champion_model: champ_model,
            champion_config: champ_cfg,
            champion_thr_long: champion.best_thr_long,
            champion_thr_short: champion.best_thr_short,
            champion_is_fitness: champion.is_sylva_fitness,
            champion_oof_fitness: champion.weighted_avg_rank,
            candidate_reports,
            total_evaluated: total_candidates,
        })
    }
}
