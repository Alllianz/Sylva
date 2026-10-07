/// Hoja bayesiana con distribución a posteriori conjugada Normal-Inverse-Gamma (NIG)
#[derive(Clone, Debug)]
pub struct BayesianLeaf {
    /// Media a posteriori (\mu)
    pub mu: f32,
    /// Precisión / peso de las observaciones (\kappa)
    pub kappa: f32,
    /// Parámetro de forma a posteriori (\alpha)
    pub alpha: f32,
    /// Parámetro de escala a posteriori (\beta)
    pub beta: f32,
    /// Número de muestras observadas en la hoja
    pub sample_count: usize,
}

impl BayesianLeaf {
    /// Crea una nueva hoja bayesiana con los hiperparámetros a priori NIG
    pub fn new(prior_mean: f32, prior_precision: f32, prior_shape: f32, prior_scale: f32) -> Self {
        Self {
            mu: prior_mean,
            kappa: prior_precision.max(0.01),
            alpha: prior_shape.max(1.1),
            beta: prior_scale.max(0.01),
            sample_count: 0,
        }
    }

    /// Actualización streaming exacta de la distribución conjugada Normal-Inverse-Gamma
    pub fn update(&mut self, y: f32, decay: f32) {
        self.sample_count += 1;
        let d = decay.clamp(0.90, 1.0);

        let kappa_prev = self.kappa * d;
        let mu_prev = self.mu;
        let alpha_prev = self.alpha * d;
        let beta_prev = self.beta * d;

        // Actualización de parámetros posteriores
        let kappa_new = kappa_prev + 1.0;
        let mu_new = (kappa_prev * mu_prev + y) / kappa_new;
        let alpha_new = alpha_prev + 0.5;

        let diff = y - mu_prev;
        let beta_new = beta_prev + (kappa_prev * diff * diff) / (2.0 * kappa_new);

        self.kappa = kappa_new;
        self.mu = mu_new;
        self.alpha = alpha_new.max(1.1);
        self.beta = beta_new.max(0.001);
    }

    /// Retorna la media predictiva y la varianza total (epistémica + aleatoria)
    pub fn predict(&self) -> (f32, f32) {
        let mean = self.mu;
        // Varianza de la distribución t de Student a posteriori
        let var = if self.alpha > 1.0 {
            (self.beta / (self.alpha - 1.0)) * (1.0 + 1.0 / self.kappa)
        } else {
            self.beta * (1.0 + 1.0 / self.kappa)
        };
        (mean, var.max(1e-6))
    }

    /// Muestreo de Thompson a partir de la distribución predictiva a posteriori
    pub fn sample_posterior(&self, rng_state: &mut u64) -> f32 {
        let (mean, var) = self.predict();
        let std_dev = var.sqrt();
        let z = sample_standard_normal(rng_state);
        mean + std_dev * z
    }
}

#[inline(always)]
fn fast_rand_uniform(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u = ((*state >> 32) as u32) as f32 / 4294967296.0;
    u.clamp(1e-7, 0.9999999)
}

/// Muestreo Normal estándar usando la transformada Box-Muller
#[inline(always)]
fn sample_standard_normal(state: &mut u64) -> f32 {
    let u1 = fast_rand_uniform(state);
    let u2 = fast_rand_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}
