use crate::dashboard::grid_dashboard::GbdtGridCandidateReport;
use crate::metrics::backtest_report::BacktestReport;
use crate::trees::bayesian::types::BayesianTreeConfig;
use crate::trees::online::types::OnlineGbdtConfig;
use serde::{Deserialize, Serialize};

/// Tipo de modelo de árboles para la auto-optimización
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AstroTreeModelType {
    /// Streaming GBDT con cotas de Hoeffding y olvido exponencial
    OnlineConventional,
    /// Árboles Bayesianos Online con distribución Normal-Inverse-Gamma y cuantificación de incertidumbre
    BayesianOnline,
}

/// Hiperparámetros de un candidato generado por la optimización evolutiva Astro EVO
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AstroTreeCandidateConfig {
    pub model_type: AstroTreeModelType,
    pub max_depth: usize,
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub learning_rate: f32,
    pub decay_factor: f32,
    pub l2_reg: f32,
    pub l1_reg: f32,
    pub grace_period: usize,
    pub split_confidence: f32,
    pub colsample_bytree: f32,
    pub prior_precision: f32,
    pub uncertainty_penalty_kappa: f32,
}

impl AstroTreeCandidateConfig {
    pub fn to_online_config(&self) -> OnlineGbdtConfig {
        OnlineGbdtConfig {
            n_trees: self.n_trees,
            max_depth: self.max_depth,
            learning_rate: self.learning_rate,
            decay_factor: self.decay_factor,
            l2_reg: self.l2_reg,
            l1_reg: self.l1_reg,
            min_samples_split: self.min_samples_leaf,
            grace_period: self.grace_period,
            split_confidence: self.split_confidence,
            tie_threshold: 0.05,
            colsample_bytree: self.colsample_bytree,
            gamma: 0.0001,
        }
    }

    pub fn to_bayesian_config(&self) -> BayesianTreeConfig {
        BayesianTreeConfig {
            n_trees: self.n_trees,
            max_depth: self.max_depth,
            learning_rate: self.learning_rate,
            decay_factor: self.decay_factor,
            prior_mean: 0.0,
            prior_precision: self.prior_precision,
            prior_shape: 2.5,
            prior_scale: 1.0,
            min_samples_leaf: self.min_samples_leaf,
            uncertainty_penalty_kappa: self.uncertainty_penalty_kappa,
            use_thompson_sampling: false,
            confidence_level: 0.95,
            colsample_bytree: self.colsample_bytree,
        }
    }
}

/// Registro individual de una prueba/trial evolutivo
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AstroTreeTrial {
    pub trial_idx: usize,
    pub generation_idx: usize,
    pub config: AstroTreeCandidateConfig,
    pub best_threshold_long: f32,
    pub best_threshold_short: f32,
    pub is_astro_fitness: f64,
    pub report_nom: BacktestReport,
    pub report_pct: BacktestReport,
    pub candidate_report: GbdtGridCandidateReport,
}

/// Configuración del optimizador evolutivo Astro EVO para árboles
#[derive(Clone, Debug)]
pub struct AstroTreeAutoTuningConfig {
    pub model_type: AstroTreeModelType,
    pub population_size: usize,
    pub generations: usize,
    pub initial_exploratory_trials: usize,
    pub mutation_rate: f32,
    pub rolling_window: usize,
    pub target_horizon: usize,
    pub seed: u64,
    pub use_gpu: bool,
}

impl Default for AstroTreeAutoTuningConfig {
    fn default() -> Self {
        Self {
            model_type: AstroTreeModelType::OnlineConventional,
            population_size: 20,
            generations: 8,
            initial_exploratory_trials: 15,
            mutation_rate: 0.30,
            rolling_window: 100,
            target_horizon: 4,
            seed: 987654321,
            use_gpu: true,
        }
    }
}

/// Resultado global de la auto-optimización Astro EVO
pub struct AstroTreeAutoTuningResult {
    pub champion_trial: AstroTreeTrial,
    pub all_trials: Vec<AstroTreeTrial>,
    pub candidate_reports: Vec<GbdtGridCandidateReport>,
    pub total_evaluated: usize,
}
