use crate::features::TOTAL_FEATURES;

/// Configuración de hiperparámetros para Online Gradient Boosted Decision Trees (Streaming GBDT)
#[derive(Clone, Debug)]
pub struct OnlineGbdtConfig {
    /// Número de árboles aditivos en el ensamble online
    pub n_trees: usize,
    /// Profundidad máxima de cada árbol
    pub max_depth: usize,
    /// Tasa de aprendizaje (shrinkage eta)
    pub learning_rate: f32,
    /// Factor de olvido exponencial para adaptación no estacionaria (\delta \in (0, 1])
    pub decay_factor: f32,
    /// Regularización L2 sobre los pesos de las hojas (\lambda)
    pub l2_reg: f32,
    /// Regularización L1 para dispersión de hojas (\alpha)
    pub l1_reg: f32,
    /// Mínimo de muestras acumuladas en una hoja antes de evaluar una división
    pub min_samples_split: usize,
    /// Período de gracia: número de muestras entre evaluaciones sucesivas de división
    pub grace_period: usize,
    /// Nivel de confianza para el bound de Hoeffding (1 - \delta_{hoeff})
    pub split_confidence: f32,
    /// Umbral para empates en la cota de Hoeffding (\tau)
    pub tie_threshold: f32,
    /// Fracción de submuestreo de variables por árbol (0.5..1.0)
    pub colsample_bytree: f32,
    /// Umbral de ganancia mínima de división (\gamma)
    pub gamma: f32,
}

impl Default for OnlineGbdtConfig {
    fn default() -> Self {
        Self {
            n_trees: 30,
            max_depth: 3,
            learning_rate: 0.03,
            decay_factor: 0.995,
            l2_reg: 1.0,
            l1_reg: 0.01,
            min_samples_split: 30,
            grace_period: 15,
            split_confidence: 0.05, // delta = 0.05 (95% confianza)
            tie_threshold: 0.05,
            colsample_bytree: 0.8,
            gamma: 0.0001,
        }
    }
}

/// Estadísticas acumuladas y recursivas en cada hoja streaming
#[derive(Clone, Debug)]
pub struct OnlineLeafStats {
    /// Suma acumulada de gradientes de primer orden con decaimiento: G = \sum g_i
    pub sum_g: f32,
    /// Suma acumulada de hessianos de segundo orden con decaimiento: H = \sum h_i
    pub sum_h: f32,
    /// Número de muestras observadas en esta hoja
    pub sample_count: usize,
    /// Contador de muestras desde la última evaluación de división
    pub samples_since_split_eval: usize,
    /// Mínimos y máximos observados por variable para generar candidatos de corte adaptativos
    pub feature_min: [f32; TOTAL_FEATURES],
    pub feature_max: [f32; TOTAL_FEATURES],
    /// Suma de g y h acumulada en bins/cuantiles para cada característica candidata
    pub split_g_left: [[f32; 8]; TOTAL_FEATURES],
    pub split_h_left: [[f32; 8]; TOTAL_FEATURES],
    pub split_g_right: [[f32; 8]; TOTAL_FEATURES],
    pub split_h_right: [[f32; 8]; TOTAL_FEATURES],
}

impl Default for OnlineLeafStats {
    fn default() -> Self {
        Self {
            sum_g: 0.0,
            sum_h: 0.0,
            sample_count: 0,
            samples_since_split_eval: 0,
            feature_min: [f32::MAX; TOTAL_FEATURES],
            feature_max: [f32::MIN; TOTAL_FEATURES],
            split_g_left: [[0.0; 8]; TOTAL_FEATURES],
            split_h_left: [[0.0; 8]; TOTAL_FEATURES],
            split_g_right: [[0.0; 8]; TOTAL_FEATURES],
            split_h_right: [[0.0; 8]; TOTAL_FEATURES],
        }
    }
}

impl OnlineLeafStats {
    /// Actualiza las estadísticas con una nueva muestra, aplicando factor de olvido exponencial
    pub fn update(&mut self, features: &[f32; TOTAL_FEATURES], g: f32, h: f32, decay: f32) {
        // Aplicar decaimiento exponencial a la historia acumulada
        self.sum_g = self.sum_g * decay + g;
        self.sum_h = self.sum_h * decay + h;
        self.sample_count += 1;
        self.samples_since_split_eval += 1;

        // Actualizar rangos y bins por variable
        for feat_idx in 0..TOTAL_FEATURES {
            let val = features[feat_idx];
            if val < self.feature_min[feat_idx] {
                self.feature_min[feat_idx] = val;
            }
            if val > self.feature_max[feat_idx] {
                self.feature_max[feat_idx] = val;
            }

            let min_v = self.feature_min[feat_idx];
            let max_v = self.feature_max[feat_idx];
            let range = max_v - min_v;

            if range > 1e-6 {
                // 8 cuantiles de división rápida
                for bin in 0..8 {
                    let threshold = min_v + range * ((bin as f32 + 1.0) / 9.0);
                    if val <= threshold {
                        self.split_g_left[feat_idx][bin] = self.split_g_left[feat_idx][bin] * decay + g;
                        self.split_h_left[feat_idx][bin] = self.split_h_left[feat_idx][bin] * decay + h;
                    } else {
                        self.split_g_right[feat_idx][bin] = self.split_g_right[feat_idx][bin] * decay + g;
                        self.split_h_right[feat_idx][bin] = self.split_h_right[feat_idx][bin] * decay + h;
                    }
                }
            }
        }
    }

    /// Calcula el peso óptimo regularizado de la hoja
    pub fn compute_weight(&self, l1_reg: f32, l2_reg: f32) -> f32 {
        if self.sum_h + l2_reg <= 1e-7 {
            return 0.0;
        }

        if l1_reg > 0.0 {
            if self.sum_g > l1_reg {
                -(self.sum_g - l1_reg) / (self.sum_h + l2_reg)
            } else if self.sum_g < -l1_reg {
                -(self.sum_g + l1_reg) / (self.sum_h + l2_reg)
            } else {
                0.0
            }
        } else {
            -self.sum_g / (self.sum_h + l2_reg)
        }
    }

    /// Calcula la ganancia de puntuación del nodo
    pub fn compute_score(&self, l1_reg: f32, l2_reg: f32) -> f32 {
        if self.sum_h + l2_reg <= 1e-7 {
            return 0.0;
        }

        let g_adj = if l1_reg > 0.0 {
            if self.sum_g > l1_reg {
                self.sum_g - l1_reg
            } else if self.sum_g < -l1_reg {
                self.sum_g + l1_reg
            } else {
                0.0
            }
        } else {
            self.sum_g
        };

        (g_adj * g_adj) / (self.sum_h + l2_reg)
    }
}

/// Muestra para procesamiento streaming online
#[derive(Clone, Debug)]
pub struct OnlineSample {
    pub features: [f32; TOTAL_FEATURES],
    pub target: f32,
    pub timestamp: i64,
}

/// Métricas de rendimiento y aprendizaje online en streaming
#[derive(Clone, Debug, Default)]
pub struct OnlineMetrics {
    pub total_samples: usize,
    pub running_mse: f32,
    pub running_mae: f32,
    pub running_hit_ratio: f32,
    pub running_directional_pnl: f32,
    pub tree_count: usize,
    pub total_leaves: usize,
}
