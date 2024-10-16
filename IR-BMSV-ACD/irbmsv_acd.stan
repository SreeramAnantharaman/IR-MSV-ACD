// Formula (Paper) to code relevance
// Code  | Paper
// omega = omega;
// delta = delta;
// durpar = alpha;
// mu = mu;
// tau = sigmasq;
// phi = phi;
// Lcorr = R/Rho;

data {
  int T;
  int<lower=1> q;
  int<lower=0> p;
  int<lower=2> k;
  vector[p] sds;
  vector[T] g;
  real<lower=0> alpha_dir;
  matrix[1, p] mean_y;
  matrix[T, p] y;
}

parameters {
  real<lower=0> omega;
  real<lower=0> delta;
  simplex[k] durpar;
  vector[p] mu;
  vector<lower=0, upper=1>[p] tau;
  vector<lower=0, upper=1>[p] phi;
  cholesky_factor_corr[p] Lcorr_raw;  // Raw Cholesky factor
  matrix[T, p] h_std;  // Latent state
}

transformed parameters {
  matrix[T, p] mu_sv;
  matrix[T, p] h;
  for (j in 1:p) {
    mu_sv[1, j] = mu[j];
    h[,j]=h_std[,j]*sqrt(tau[j]);
    h[1,j] /= sqrt(1 - phi[j]*phi[j]);
    h[1,j] += mu_sv[1, j];
    for (t in 2:T) {
      mu_sv[t, j] = mu[j] + phi[j]^(g[t]) * (h[t - 1, j] - mu[j]);
      h[t,j] *= sqrt((1 - phi[j]^(2*g[t])) / (1 - phi[j]*phi[j]));
      h[t,j] += mu_sv[t, j];
    }
  }
  matrix[p, p] Lcorr;    // Lower triangular Cholesky factor
  matrix[p, p] Sigma;    // Covariance matrix
  array[T] vector[p] mv_cov;
  array[T] matrix[p, p] sigma_y;
  
  // Convert raw Cholesky factor to lower triangular matrix
  Lcorr = multiply_lower_tri_self_transpose(Lcorr_raw);
  
  // Construct the covariance matrix Sigma
  Sigma = quad_form_diag(Lcorr, sds);
  
  // Initialize matrices for the covariance of innovations and observations
  for (t in 1:T) {
    mv_cov[t] = to_vector(exp(h[t] / 2));
    sigma_y[t] = quad_form_diag(Sigma,mv_cov[t]);
  }
}

model {
  vector[T] psi;         // Vector for the duration parameters
  
  delta ~ gamma(0.001, 0.001);
  omega ~ gamma(0.001, 0.001);
  durpar ~ dirichlet(rep_vector(alpha_dir, k));
  phi ~ beta(1, 1);
  
  // LKJ prior for the raw Cholesky factor
  Lcorr_raw ~ lkj_corr_cholesky(1.3);
  vector[T] gamma_llpd;
  
  // Prior for the innovations
  for (t in 1:q) {
    psi[t] = omega;
    gamma_llpd[t] = gamma_lpdf(g[t]|delta, delta / psi[t]);
  }
  
  // Model for the innovations
  for (t in (q + 1):T) {
    psi[t] = omega;
    for (i in 1:q) {
      psi[t] += durpar[i] * g[t - i];
    }
    gamma_llpd[t] = gamma_lpdf(g[t]|delta, delta / psi[t]);
  }
  target += sum(gamma_llpd);
  
  // Prior for the state variables
  mu ~ normal(0, 10);
  tau ~ inv_gamma(1, 1);
  to_vector(h_std) ~ std_normal();
  real y_llpd;
  
  // Likelihood for the first observation
  y_llpd = multi_normal_cholesky_lpdf(y[1] | mean_y[1]', cholesky_decompose(sigma_y[1]));

// Accumulate the log-likelihood
for (i in 2:T) {
  y_llpd += multi_normal_cholesky_lpdf(y[i] | mean_y[1]', cholesky_decompose(sigma_y[i]));
}

// Add to the target log probability
target += y_llpd;
}
