/// Configuración de hiperparámetros para Bayesian Online Boosted Trees
#[derive(Clone, Debug)]
pub struct BayesianTreeConfig {
    /// Número de árboles aditivos en el ensamble bayesiano
    pub n_trees: usize,
    /// Profundidad máxima permitida por árbol
    pub max_depth: usize,
    /// Tasa de aprendizaje (shrinkage)
    pub learning_rate: f32,
    /// Factor de olvido exponencial bayesiano (\delta \in (0, 1])
    pub decay_factor: f32,
    /// Media a priori en cada hoja (\mu_0)
    pub prior_mean: f32,
    /// Precisión/fuerza a priori (\kappa_0)
    pub prior_precision: f32,
    /// Parámetro de forma a priori Normal-Inverse-Gamma (\alpha_0)
    pub prior_shape: f32,
    /// Parámetro de escala a priori Normal-Inverse-Gamma (\beta_0)
    pub prior_scale: f32,
    /// Mínimo de muestras por hoja
    pub min_samples_leaf: usize,
    /// Penalización por incertidumbre epistémica (\kappa_{risk} en \hat{\mu} - \kappa \cdot \hat{\sigma})
    pub uncertainty_penalty_kappa: f32,
    /// Usar muestreo de Thompson en lugar de media a posteriori
    pub use_thompson_sampling: bool,
    /// Nivel de confianza para intervalos creíbles (ej. 0.95)
    pub confidence_level: f32,
    /// Fracción de submuestreo de características por árbol
    pub colsample_bytree: f32,
}

impl Default for BayesianTreeConfig {
    fn default() -> Self {
        Self {
            n_trees: 25,
            max_depth: 3,
            learning_rate: 0.03,
            decay_factor: 0.995,
            prior_mean: 0.0,
            prior_precision: 1.0,
            prior_shape: 2.5,
            prior_scale: 1.0,
            min_samples_leaf: 20,
            uncertainty_penalty_kappa: 0.5,
            use_thompson_sampling: false,
            confidence_level: 0.95,
            colsample_bytree: 0.8,
        }
    }
}

/// Estructura de inferencia bayesiana con distribución a posteriori
#[derive(Clone, Debug)]
pub struct BayesianPrediction {
    /// Media a posteriori del retorno esperado: \mathbb{E}[Y | x]
    pub mean: f32,
    /// Varianza a posteriori total (Incertidumbre Epistémica + Aleatoria): \mathbb{V}ar[Y | x]
    pub variance: f32,
    /// Desviación estándar de incertidumbre (\sigma)
    pub std_dev: f32,
    /// Límite inferior del intervalo creíble bayesiano al 95%
    pub lower_bound: f32,
    /// Límite superior del intervalo creíble bayesiano al 95%
    pub upper_bound: f32,
    /// Probabilidad a posteriori de retorno estrictamente positivo: P(Y > 0 | x)
    pub prob_positive: f32,
    /// Alpha ajustado por riesgo e incertidumbre: \hat{\mu} - \text{sign}(\hat{\mu}) \cdot \kappa \cdot \hat{\sigma}
    pub adjusted_alpha: f32,
}

impl Default for BayesianPrediction {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 1.0,
            std_dev: 1.0,
            lower_bound: -1.96,
            upper_bound: 1.96,
            prob_positive: 0.5,
            adjusted_alpha: 0.0,
        }
    }
}
