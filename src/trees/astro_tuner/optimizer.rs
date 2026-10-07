use crate::data::db::Kline;
use crate::dashboard::grid_dashboard::{generate_gbdt_grid_dashboard, GbdtGridCandidateReport, GbdtGridDashboardSummary};
use crate::engine::simulator::BacktestSimulator;
use crate::engine::types::BacktestConfig;
use crate::features::CachedIndicators;
use crate::features::FeatureMask;
use crate::metrics::ranking::compute_multicriteria_rankings;
use crate::trees::astro_tuner::evaluator::AstroTreeEvaluator;
use crate::trees::astro_tuner::sampler::AstroTreeSampler;
use crate::trees::astro_tuner::types::{
    AstroTreeAutoTuningConfig, AstroTreeAutoTuningResult, AstroTreeCandidateConfig,
    AstroTreeModelType, AstroTreeTrial,
};
use crate::trees::dataset::TabularDataset;
use crate::trees::gpu::GpuTreeEngine;
use crate::trees::online::gbdt::OnlineGbdtTrainer;
use crate::trees::online::model_wrapper::OnlineGbdtEquationModel;
use rayon::prelude::*;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Optimizador Evolutivo Automático para Árboles (Astro EVO Tree Engine)
pub struct AstroTreeOptimizer {
    pub config: AstroTreeAutoTuningConfig,
    pub backtest_config_nom: BacktestConfig,
    pub backtest_config_pct: BacktestConfig,
    pub cached: Arc<CachedIndicators>,
    pub gpu_engine: Option<Arc<GpuTreeEngine>>,
}

impl AstroTreeOptimizer {
    pub fn new(
        config: AstroTreeAutoTuningConfig,
        backtest_config_nom: BacktestConfig,
        backtest_config_pct: BacktestConfig,
        cached: Arc<CachedIndicators>,
    ) -> Self {
        let use_gpu = config.use_gpu;
        Self::new_with_engine(config, backtest_config_nom, backtest_config_pct, cached, use_gpu)
    }

    pub fn new_with_engine(
        config: AstroTreeAutoTuningConfig,
        backtest_config_nom: BacktestConfig,
        backtest_config_pct: BacktestConfig,
        cached: Arc<CachedIndicators>,
        use_gpu: bool,
    ) -> Self {
        let gpu_engine = if use_gpu {
            match GpuTreeEngine::new() {
                Ok(gpu) => {
                    println!("  🎮 [GPU] Acelerador gráfico activo: {}", gpu.hardware_info());
                    Some(Arc::new(gpu))
                }
                Err(e) => {
                    println!("  ⚠️ [GPU] No disponible ({}), conmutando automáticamente a CPU Rayon multihilo", e);
                    None
                }
            }
        } else {
            println!("  💻 [CPU] Modo Determinista CPU Seleccionado (32 Hilos Rayon - Cero Dependencia Gráfica)");
            None
        };

        Self {
            config,
            backtest_config_nom,
            backtest_config_pct,
            cached,
            gpu_engine,
        }
    }

    /// Ejecuta el proceso completo de auto-optimización evolutiva determinista acelerada por GPU
    pub fn optimize(
        &self,
        klines: &[Kline],
        is_dataset: &TabularDataset,
        is_start_idx: usize,
        oos_start_idx: usize,
        tf: &str,
        mask: Option<&FeatureMask>,
    ) -> Result<AstroTreeAutoTuningResult, Box<dyn Error + Send + Sync>> {
        let num_threads = rayon::current_num_threads();
        let model_label = match self.config.model_type {
            AstroTreeModelType::OnlineConventional => "Online GBDT (Streaming Hoeffding Trees)",
            AstroTreeModelType::BayesianOnline => "Bayesian Online Trees (NIG Posterior & Incertidumbre)",
        };

        let feature_label = match mask {
            Some(m) if m.active_count() <= 5 => "Microestructura Pura (F1..F5)",
            Some(m) if m.active_count() <= 30 => "Conjunto Reducido de 30 Features (F1..F30 Mercado Puro)",
            _ => "Red Completa de 32 Variables Cuantitativas",
        };

        let hardware_status = match &self.gpu_engine {
            Some(gpu) => format!("🎮 ACELERACIÓN GPU: {}", gpu.hardware_info()),
            None => format!("💻 CPU MULTIHILO ({} CORES)", num_threads),
        };

        println!("\n=======================================================================================================================================");
        println!("               🧬 AUTO-OPTIMIZACIÓN EVOLUTIVA ASTRO EVO - {}", hardware_status);
        println!("=======================================================================================================================================");
        println!("  • Modelo: {}", model_label);
        println!("  • Espacio de Variables: {}", feature_label);
        println!("  • Generaciones: {} | Población por Gen: {} | Pruebas Iniciales Exploratorias: {} | Hilos Rayon: {}",
            self.config.generations, self.config.population_size, self.config.initial_exploratory_trials, num_threads
        );
        if let Some(gpu) = &self.gpu_engine {
            println!("  • Pipeline Gráfico: Inferencia Vectorial en VRAM + Calibración de Umbrales Instantánea ({})", gpu.hardware_info());
        }
        println!("---------------------------------------------------------------------------------------------------------------------------------------");

        let mut evaluator = AstroTreeEvaluator::new(
            self.backtest_config_nom.clone(),
            self.backtest_config_pct.clone(),
            self.config.rolling_window,
            Arc::clone(&self.cached),
        );
        if let Some(ref gpu) = self.gpu_engine {
            evaluator = evaluator.with_gpu_engine(Arc::clone(gpu));
        }

        let mut sampler = AstroTreeSampler::new(self.config.seed, self.config.model_type);
        let mut all_trials: Vec<AstroTreeTrial> = Vec::new();

        let colors = [
            "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899",
            "#06b6d4", "#f97316", "#14b8a6", "#6366f1", "#a855f7",
            "#84cc16", "#ef4444", "#eab308", "#0ea5e9", "#d946ef",
            "#22c55e", "#64748b", "#fb7185", "#38bdf8", "#4ade80",
        ];

        // 1. --- FASE 1: EXPLORACIÓN GLOBAL INICIAL DETERMINISTA ---
        let exploratory_candidates: Vec<(usize, AstroTreeCandidateConfig, String)> = (0..self.config.initial_exploratory_trials)
            .map(|i| {
                let candidate = sampler.sample_exploratory();
                let color = colors[i % colors.len()].to_string();
                (i + 1, candidate, color)
            })
            .collect();

        let phase1_engine = if self.gpu_engine.is_some() { "GPU Pipeline + Rayon" } else { "32 Hilos CPU" };
        println!("\n--- FASE 1: Exploración Inicial Determinista ({} Candidatos en {}) ---", self.config.initial_exploratory_trials, phase1_engine);

        let progress = AtomicUsize::new(0);
        let total_exploratory = self.config.initial_exploratory_trials;

        let exploratory_trials: Vec<AstroTreeTrial> = exploratory_candidates
            .into_par_iter()
            .map(|(trial_idx, candidate, color)| {
                let res = evaluator.evaluate_candidate(
                    &candidate,
                    trial_idx,
                    0,
                    klines,
                    is_dataset,
                    is_start_idx,
                    oos_start_idx,
                    mask,
                    color,
                );
                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 10 == 0 || done == total_exploratory || done == 1 {
                    println!("  ⏳ [Fase 1] Exploración: {}/{} candidatos evaluados...", done, total_exploratory);
                }
                res
            })
            .collect::<Result<Vec<_>, _>>()?;

        for trial in &exploratory_trials {
            println!(
                "  • Exploración [{:>2}/{}]: Depth: {} | Trees: {:>2} | LR: {:.3} | Decay: {:.4} ➔ Fitness: {:>6.2} | Profit: +${:.2} ({:+.2}%)",
                trial.trial_idx, self.config.initial_exploratory_trials, trial.config.max_depth, trial.config.n_trees, trial.config.learning_rate, trial.config.decay_factor, trial.is_astro_fitness, trial.report_nom.net_profit, trial.report_nom.total_return_pct
            );
        }
        all_trials.extend(exploratory_trials);

        // 2. --- FASE 2: BUCLE EVOLUTIVO DETERMINISTA PARALELO ASTRO EVO ---
        println!("\n--- FASE 2: Bucle Evolutivo Determinista Astro EVO ({} Generaciones x {} Individuos) ---", self.config.generations, self.config.population_size);

        for gen_idx in 1..=self.config.generations {
            all_trials.sort_by(|a, b| {
                b.is_astro_fitness
                    .partial_cmp(&a.is_astro_fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.trial_idx.cmp(&b.trial_idx))
            });

            let top_parents: Vec<AstroTreeCandidateConfig> = all_trials.iter().take(4).map(|t| t.config.clone()).collect();
            let best_historical_fitness = all_trials[0].is_astro_fitness;

            let phase2_engine = if self.gpu_engine.is_some() { "GPU + Rayon" } else { "32 Cores CPU" };
            println!(
                "\n  🧬 GENERACIÓN {:>2}/{} | Mejor Fitness Actual: {:>6.2} (Depth={}, Trees={}, Decay={:.4}) [{}]",
                gen_idx, self.config.generations, best_historical_fitness, all_trials[0].config.max_depth, all_trials[0].config.n_trees, all_trials[0].config.decay_factor, phase2_engine
            );

            let base_idx = all_trials.len();
            let gen_candidates: Vec<(usize, AstroTreeCandidateConfig, String)> = (0..self.config.population_size)
                .map(|ind_idx| {
                    let candidate = if sampler.next_f32() < 0.20 {
                        sampler.sample_exploratory()
                    } else {
                        let parent_idx = (sampler.next_u32() as usize) % top_parents.len();
                        sampler.mutate_parent(&top_parents[parent_idx])
                    };
                    let color = colors[(base_idx + ind_idx) % colors.len()].to_string();
                    (base_idx + ind_idx + 1, candidate, color)
                })
                .collect();

            let gen_progress = AtomicUsize::new(0);
            let total_gen = self.config.population_size;

            let gen_trials: Vec<AstroTreeTrial> = gen_candidates
                .into_par_iter()
                .map(|(trial_idx, candidate, color)| {
                    let res = evaluator.evaluate_candidate(
                        &candidate,
                        trial_idx,
                        gen_idx,
                        klines,
                        is_dataset,
                        is_start_idx,
                        oos_start_idx,
                        mask,
                        color,
                    );
                    let done = gen_progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 10 == 0 || done == total_gen || done == 1 {
                        println!("     ⏳ [Gen {}/{}] Evolución: {}/{} individuos evaluados...", gen_idx, self.config.generations, done, total_gen);
                    }
                    res
                })
                .collect::<Result<Vec<_>, _>>()?;

            for (ind_idx, trial) in gen_trials.iter().enumerate() {
                if (ind_idx + 1) % 5 == 0 || ind_idx == gen_trials.len() - 1 {
                    println!(
                        "     └─ Individuo [{:>2}/{}]: Depth: {} | Trees: {:>2} | LR: {:.3} | Decay: {:.4} ➔ Fitness: {:>6.2} | Profit: +${:.2}",
                        ind_idx + 1, self.config.population_size, trial.config.max_depth, trial.config.n_trees, trial.config.learning_rate, trial.config.decay_factor, trial.is_astro_fitness, trial.report_nom.net_profit
                    );
                }
            }

            all_trials.extend(gen_trials);
        }

        // 3. --- FASE 3: RANKING MULTICRITERIO POR PUESTOS DE ASTRO EVO ---
        println!("\n  📊 Calculando Ranking Multicriterio por Puestos Astro EVO (R² 2x, Slope Ratio, Cap/DD, Smoothness)...");
        let mut candidate_reports: Vec<GbdtGridCandidateReport> = all_trials.iter().map(|t| t.candidate_report.clone()).collect();

        let rankings = compute_multicriteria_rankings(
            &candidate_reports,
            |c| format!("{}-{}-{}-{:.4}", c.max_depth, c.min_samples_leaf, c.n_trees, c.best_thr_long),
            |c| c.raw_slope_ratio,
            |c| c.raw_cap_dd_ratio,
            |c| c.raw_r2_score,
            |c| c.raw_smoothness_score,
        );

        for (idx, trial) in all_trials.iter_mut().enumerate() {
            let key = format!("{}-{}-{}-{:.4}", trial.config.max_depth, trial.config.min_samples_leaf, trial.config.n_trees, trial.best_threshold_long);
            if let Some(r_score) = rankings.get(&key) {
                trial.candidate_report.rank_slope = r_score.rank_slope;
                trial.candidate_report.rank_cap_dd = r_score.rank_cap_dd;
                trial.candidate_report.rank_r2 = r_score.rank_r2;
                trial.candidate_report.rank_smoothness = r_score.rank_smoothness;
                trial.candidate_report.weighted_avg_rank = r_score.weighted_avg_rank;
                trial.candidate_report.astro_rank_fitness = r_score.astro_rank_fitness;
                candidate_reports[idx] = trial.candidate_report.clone();
            }
        }

        all_trials.sort_by(|a, b| {
            a.candidate_report
                .weighted_avg_rank
                .partial_cmp(&b.candidate_report.weighted_avg_rank)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.trial_idx.cmp(&b.trial_idx))
        });

        let mut champion = all_trials[0].clone();

        // Para el Campeón Absoluto, reconstruir la curva completa de alta resolución y lista de trades
        // para el reporte del terminal y los dashboards interactivos (únicamente 1 modelo):
        let sim_nom = BacktestSimulator::new(self.backtest_config_nom.clone());
        let sim_pct = BacktestSimulator::new(self.backtest_config_pct.clone());
        let champion_online_cfg = champion.config.to_online_config();
        if let Ok(champ_trainer) = OnlineGbdtTrainer::new(champion_online_cfg).fit_stream(is_dataset, mask) {
            let mut model_nom = OnlineGbdtEquationModel::new(
                champ_trainer.clone(),
                champion.best_threshold_long,
                champion.best_threshold_short,
                self.config.rolling_window,
            ).with_cached_indicators(Arc::clone(&self.cached));
            let mut model_pct = OnlineGbdtEquationModel::new(
                champ_trainer,
                champion.best_threshold_long,
                champion.best_threshold_short,
                self.config.rolling_window,
            ).with_cached_indicators(Arc::clone(&self.cached));
            if let Some(fmask) = mask {
                model_nom = model_nom.with_mask(fmask.clone());
                model_pct = model_pct.with_mask(fmask.clone());
            }
            champion.report_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, klines.len());
            champion.report_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, klines.len());
        }

        println!("\n=======================================================================================================================================");
        println!("                             🏆 CAMPEÓN ABSOLUTO ASTRO EVO MAX (MEJOR PUESTO PONDERADO MULTICRITERIO)                                 ");
        println!("=======================================================================================================================================");
        println!("  • Arquitectura: Profundidad={} | Árboles={} | Muestras Mínimas Hoja={}", champion.config.max_depth, champion.config.n_trees, champion.config.min_samples_leaf);
        println!("  • Hiperparámetros: Learning Rate={:.3} | Factor Olvido δ={:.4} | Regularización L2={:.2}", champion.config.learning_rate, champion.config.decay_factor, champion.config.l2_reg);
        println!("  • Umbrales Calibrados: Long={:+.4} | Short={:+.4}", champion.best_threshold_long, champion.best_threshold_short);
        println!("  • Métricas IS: Puesto Ponderado={:.2} (R²: #{} | Slope: #{} | DD: #{} | Smooth: #{})",
            champion.candidate_report.weighted_avg_rank, champion.candidate_report.rank_r2, champion.candidate_report.rank_slope, champion.candidate_report.rank_cap_dd, champion.candidate_report.rank_smoothness
        );
        println!("  • Rendimiento Nominal: +${:.2} ({:+.2}%) | WinRate: {:.1}% | ProfitFactor: {:.2} | MaxDD: {:.2}%",
            champion.report_nom.net_profit, champion.report_nom.total_return_pct, champion.report_nom.win_rate_pct, champion.report_nom.profit_factor, champion.report_nom.max_drawdown_pct
        );
        println!("  • Rendimiento Compuesto: +${:.2} ({:+.2}%) | MaxDD: {:.2}% | Liquidaciones: {}",
            champion.report_pct.net_profit, champion.report_pct.total_return_pct, champion.report_pct.max_drawdown_pct, champion.report_pct.liquidations
        );
        println!("=======================================================================================================================================\n");

        let total_evaluated = all_trials.len();
        let top_candidates: Vec<GbdtGridCandidateReport> = candidate_reports.iter().take(30).cloned().collect();
        let grid_summary = GbdtGridDashboardSummary {
            tf: tf.to_string(),
            is_start_idx,
            oos_start_idx,
            initial_capital: self.backtest_config_nom.initial_capital,
            candidates: top_candidates,
        };
        let _ = generate_gbdt_grid_dashboard(&grid_summary, klines);

        Ok(AstroTreeAutoTuningResult {
            champion_trial: champion,
            all_trials,
            candidate_reports,
            total_evaluated,
        })
    }
}
