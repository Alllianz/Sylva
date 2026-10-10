use crate::dashboard::grid_dashboard::GbdtGridCandidateReport;
use crate::engine::dynamic_threshold::ThresholdMode;
use crate::metrics::backtest_report::BacktestReport;
use crate::trees::bayesian::types::BayesianTreeConfig;
use crate::trees::online::types::OnlineGbdtConfig;
use serde::{Deserialize, Serialize};

/// Tipo de modelo de árboles para la auto-optimización
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SylvaTreeModelType {
    /// Streaming GBDT con cotas de Hoeffding y olvido exponencial
    OnlineConventional,
    /// Árboles Bayesianos Online con distribución Normal-Inverse-Gamma y cuantificación de incertidumbre
    BayesianOnline,
}

// Alias para compatibilidad
pub type AstroTreeModelType = SylvaTreeModelType;

/// Hiperparámetros de un candidato generado por la optimización evolutiva Sylva EVO
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SylvaTreeCandidateConfig {
    pub model_type: SylvaTreeModelType,
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
    pub target_horizon: usize,
}

pub type AstroTreeCandidateConfig = SylvaTreeCandidateConfig;

impl SylvaTreeCandidateConfig {
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
pub struct SylvaTreeTrial {
    pub trial_idx: usize,
    pub generation_idx: usize,
    pub config: SylvaTreeCandidateConfig,
    pub best_threshold_long: f32,
    pub best_threshold_short: f32,
    pub is_sylva_fitness: f64,
    pub report_nom: BacktestReport,
    pub report_pct: BacktestReport,
    pub candidate_report: GbdtGridCandidateReport,
}

pub type AstroTreeTrial = SylvaTreeTrial;

/// Configuración del optimizador evolutivo Sylva EVO para árboles
#[derive(Clone, Debug)]
pub struct SylvaTreeAutoTuningConfig {
    pub model_type: SylvaTreeModelType,
    pub population_size: usize,
    pub generations: usize,
    pub initial_exploratory_trials: usize,
    pub mutation_rate: f32,
    pub rolling_window: usize,
    pub candidate_horizons: Vec<usize>,
    pub seed: u64,
    pub use_gpu: bool,
    pub threshold_mode: ThresholdMode,
}

pub type AstroTreeAutoTuningConfig = SylvaTreeAutoTuningConfig;

impl Default for SylvaTreeAutoTuningConfig {
    fn default() -> Self {
        Self {
            model_type: SylvaTreeModelType::OnlineConventional,
            population_size: 20,
            generations: 8,
            initial_exploratory_trials: 15,
            mutation_rate: 0.30,
            rolling_window: 100,
            candidate_horizons: vec![1, 2, 4],
            seed: 987654321,
            use_gpu: true,
            threshold_mode: ThresholdMode::DynamicAtrRatio,
        }
    }
}

/// Resultado global de la auto-optimización Sylva EVO
pub struct SylvaTreeAutoTuningResult {
    pub champion_trial: SylvaTreeTrial,
    pub all_trials: Vec<SylvaTreeTrial>,
    pub candidate_reports: Vec<GbdtGridCandidateReport>,
    pub total_evaluated: usize,
}

pub type AstroTreeAutoTuningResult = SylvaTreeAutoTuningResult;
