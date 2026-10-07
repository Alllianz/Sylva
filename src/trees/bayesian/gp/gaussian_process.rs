use super::kernel::Kernel;
use std::error::Error;

/// Regresor de Procesos Gaussianos (Gaussian Process Regression - GPR)
/// Implementa descomposición de Cholesky L L^T = K + σ_n² I para resolución exacta y numéricamente estable.
#[derive(Debug, Clone)]
pub struct GaussianProcessRegressor {
    pub kernel: Box<dyn Kernel>,
    pub noise_variance: f64,
    pub x_train: Vec<Vec<f64>>,
    pub y_train: Vec<f64>,
    pub l_cholesky: Vec<Vec<f64>>,
    pub alpha: Vec<f64>,
    pub is_fitted: bool,
}

impl GaussianProcessRegressor {
    pub fn new(kernel: Box<dyn Kernel>, noise_variance: f64) -> Self {
        Self {
            kernel,
            noise_variance: noise_variance.max(1e-8),
            x_train: Vec::new(),
            y_train: Vec::new(),
            l_cholesky: Vec::new(),
            alpha: Vec::new(),
            is_fitted: false,
        }
    }

    /// Ajusta el Proceso Gaussiano con el conjunto de observaciones {(x_i, y_i)}
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let n = x.len();
        if n == 0 || n != y.len() {
            return Err("Dimensiones inválidas para ajustar el Proceso Gaussiano".into());
        }

        // 1. Construir matriz de covarianza K + σ_n² I
        let mut k_matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let cov = self.kernel.compute(&x[i], &x[j]);
                k_matrix[i][j] = cov;
                k_matrix[j][i] = cov;
            }
            k_matrix[i][i] += self.noise_variance;
        }

        // 2. Factorización de Cholesky L L^T = K con Jitter adaptativo en caso de singularidad
        let mut l_factor = vec![vec![0.0f64; n]; n];
        let mut success = false;

        for attempt in 0..6 {
            if attempt > 0 {
                let jitter = 10.0_f64.powi(attempt - 6) * 1e-2;
                for i in 0..n {
                    k_matrix[i][i] += jitter;
                }
            }

            let mut cholesky_ok = true;
            for i in 0..n {
                for j in 0..=i {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l_factor[i][k] * l_factor[j][k];
                    }

                    if i == j {
                        let val = k_matrix[i][i] - sum;
                        if val <= 1e-12 {
                            cholesky_ok = false;
                            break;
                        }
                        l_factor[i][j] = val.sqrt();
                    } else {
                        l_factor[i][j] = (k_matrix[i][j] - sum) / l_factor[j][j];
                    }
                }
                if !cholesky_ok {
                    break;
                }
            }

            if cholesky_ok {
                success = true;
                break;
            }
        }

        if !success {
            return Err("Fallo en la descomposición de Cholesky del Proceso Gaussiano (matriz no definida positiva)".into());
        }

        // 3. Resolver L v = y (Sustitución hacia adelante)
        let mut v = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l_factor[i][j] * v[j];
            }
            v[i] = (y[i] - sum) / l_factor[i][i];
        }

        // 4. Resolver L^T alpha = v (Sustitución hacia atrás)
        let mut alpha = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l_factor[j][i] * alpha[j];
            }
            alpha[i] = (v[i] - sum) / l_factor[i][i];
        }

        self.x_train = x;
        self.y_train = y;
        self.l_cholesky = l_factor;
        self.alpha = alpha;
        self.is_fitted = true;

        Ok(())
    }

    /// Realiza la predicción posterior (media μ y varianza σ²) para un punto candidato x*
    pub fn predict(&self, x_star: &[f64]) -> (f64, f64) {
        if !self.is_fitted || self.x_train.is_empty() {
            return (0.0, self.kernel.variance());
        }

        let n = self.x_train.len();
        let mut k_star = Vec::with_capacity(n);
        for x_i in &self.x_train {
            k_star.push(self.kernel.compute(x_i, x_star));
        }

        // Media posterior: μ(x*) = k_*^T α
        let mut mean = 0.0;
        for i in 0..n {
            mean += k_star[i] * self.alpha[i];
        }

        // Varianza posterior: σ²(x*) = k(x*, x*) - v_*^T v_*, donde L v_* = k_*
        let mut v_star = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += self.l_cholesky[i][j] * v_star[j];
            }
            v_star[i] = (k_star[i] - sum) / self.l_cholesky[i][i];
        }

        let mut v_star_norm_sq = 0.0;
        for &val in &v_star {
            v_star_norm_sq += val * val;
        }

        let prior_var = self.kernel.compute(x_star, x_star);
        let variance = (prior_var - v_star_norm_sq).max(1e-9);

        (mean, variance)
    }
}
