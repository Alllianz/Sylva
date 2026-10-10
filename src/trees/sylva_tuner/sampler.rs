use crate::trees::sylva_tuner::types::{SylvaTreeCandidateConfig, SylvaTreeModelType};

/// Muestreador y Mutador Evolutivo por Bloques Coordinados (Inspirado en el motor de Sylva EVO)
pub struct SylvaTreeSampler {
    pub seed: u64,
    pub model_type: SylvaTreeModelType,
    pub candidate_horizons: Vec<usize>,
}

pub type AstroTreeSampler = SylvaTreeSampler;

impl SylvaTreeSampler {
    pub fn new(seed: u64, model_type: SylvaTreeModelType, candidate_horizons: Vec<usize>) -> Self {
        let candidate_horizons = if candidate_horizons.is_empty() {
            vec![1, 2, 4]
        } else {
            candidate_horizons
        };
        Self { seed, model_type, candidate_horizons }
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.seed >> 32) as u32
    }

    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32
    }

    #[inline(always)]
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Genera una muestra exploratoria inicial respetando la relación estructural entre bloques
    pub fn sample_exploratory(&mut self) -> SylvaTreeCandidateConfig {
        // Bloque 1: Arquitectura y Capacidad del Árbol (Profundidad, Árboles, Hojas Mínimas)
        let max_depth = (2 + (self.next_u32() % 5)) as usize; // Profundidad entre 2 y 6
        let n_trees = match max_depth {
            2 => (20 + (self.next_u32() % 35)) as usize, // 20..54
            3 => (20 + (self.next_u32() % 30)) as usize, // 20..49
            4 => (15 + (self.next_u32() % 25)) as usize, // 15..39
            _ => (10 + (self.next_u32() % 20)) as usize, // 10..29 para árboles profundos
        };
        let min_samples_leaf = (15 + (self.next_u32() % 35)) as usize; // 15..49

        // Bloque 2: Dinámica Temporal y Tasa de Aprendizaje (Log-space)
        let log_lr = match max_depth {
            2 | 3 => -4.0 + (self.next_f32() * 1.8), // e^-4.0 .. e^-2.2 (0.018 .. 0.11)
            _ => -4.6 + (self.next_f32() * 1.5),     // e^-4.6 .. e^-3.1 (0.010 .. 0.045)
        };
        let learning_rate = log_lr.exp().clamp(0.008, 0.12);

        // Factor de olvido exponencial robusto (\delta \in [0.9950, 0.9998]) para preservar memoria estructural
        let decay_factor = 0.9950 + (self.next_f32() * 0.0048);

        // Bloque 3: Regularización L1 / L2 y Período de Gracia
        let log_l2 = -1.5 + (self.next_f32() * 3.5); // 0.22 .. 7.38
        let l2_reg = log_l2.exp().clamp(0.1, 10.0);
        let l1_reg = if self.next_f32() < 0.4 { 0.0 } else { 0.001 * (1.0 + self.next_f32() * 9.0) };

        let grace_period = (10 + (self.next_u32() % 25)) as usize; // 10..34
        let split_confidence = 0.01 + (self.next_f32() * 0.09); // 0.01..0.10
        let colsample_bytree = if max_depth >= 4 { 0.70 + self.next_f32() * 0.25 } else { 0.80 + self.next_f32() * 0.20 };

        // Bloque 4: Inferencia Bayesiana & Incertidumbre
        let prior_precision = 0.2 + (self.next_f32() * 3.8); // 0.2..4.0
        let uncertainty_penalty_kappa = self.next_f32() * 1.8; // 0.0..1.8

        let h_idx = (self.next_u32() as usize) % self.candidate_horizons.len();
        let target_horizon = self.candidate_horizons[h_idx];

        SylvaTreeCandidateConfig {
            model_type: self.model_type,
            max_depth,
            n_trees,
            min_samples_leaf,
            learning_rate,
            decay_factor,
            l2_reg,
            l1_reg,
            grace_period,
            split_confidence,
            colsample_bytree,
            prior_precision,
            uncertainty_penalty_kappa,
            target_horizon,
        }
    }

    /// Mutación Adaptativa por Bloques Coordinados de un Candidato Padre Exitoso
    pub fn mutate_parent(&mut self, parent: &SylvaTreeCandidateConfig) -> SylvaTreeCandidateConfig {
        let mut child = parent.clone();
        let block_to_mutate = (self.next_u32() % 5) as usize;

        match block_to_mutate {
            0 => {
                // Mutación Bloque 1: Arquitectura (Depth, Trees, Min Leaf)
                let delta_d = (self.next_gaussian() * 0.8).round() as i32;
                child.max_depth = ((child.max_depth as i32) + delta_d).clamp(2, 6) as usize;

                let delta_t = (self.next_gaussian() * 5.0).round() as i32;
                child.n_trees = ((child.n_trees as i32) + delta_t).clamp(10, 60) as usize;

                let delta_leaf = (self.next_gaussian() * 4.0).round() as i32;
                child.min_samples_leaf = ((child.min_samples_leaf as i32) + delta_leaf).clamp(10, 60) as usize;
            }
            1 => {
                // Mutación Bloque 2: Dinámica Temporal (Learning Rate & Olvido Exponencial en espacio log)
                let log_lr = child.learning_rate.ln() + self.next_gaussian() * 0.25;
                child.learning_rate = log_lr.exp().clamp(0.005, 0.12);

                // Mutación de decay factor
                let delta_decay = self.next_gaussian() * 0.001;
                child.decay_factor = (child.decay_factor + delta_decay).clamp(0.9940, 0.9999);
            }
            2 => {
                // Mutación Bloque 3: Regularización (L1, L2, Colsample, Grace)
                let log_l2 = child.l2_reg.ln() + self.next_gaussian() * 0.35;
                child.l2_reg = log_l2.exp().clamp(0.05, 15.0);

                if self.next_f32() < 0.3 {
                    child.l1_reg = (child.l1_reg + self.next_gaussian() * 0.005).max(0.0);
                }

                let delta_grace = (self.next_gaussian() * 3.0).round() as i32;
                child.grace_period = ((child.grace_period as i32) + delta_grace).clamp(8, 45) as usize;
            }
            3 => {
                // Mutación Bloque 4: Parámetros Bayesianos & Incertidumbre
                let delta_prec = self.next_gaussian() * 0.4;
                child.prior_precision = (child.prior_precision + delta_prec).clamp(0.1, 5.0);

                let delta_kappa = self.next_gaussian() * 0.25;
                child.uncertainty_penalty_kappa = (child.uncertainty_penalty_kappa + delta_kappa).clamp(0.0, 2.5);
            }
            _ => {
                // Mutación Bloque 5: Horizonte Causal Multi-Vela
                let h_idx = (self.next_u32() as usize) % self.candidate_horizons.len();
                child.target_horizon = self.candidate_horizons[h_idx];
            }
        }

        child
    }
}
