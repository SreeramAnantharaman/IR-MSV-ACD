// Formula (Paper) to code relevance
// Code  | Paper
// omega = omega;
// delta = delta;
// durpar = alpha;
// mu = mu;
// sigmasq = sigmasq;
// phi = phi;
// meanar = kappa;
// beta1 = beta; 
// sigmasq_rho = gamma;

data {
  int<lower=1> T;            // Number of time points
  int<lower=1> q;
  int<lower=0> p;
  int<lower=2> k;
  matrix[T, p] y;            // Observations
  vector[T] g;
  real<lower=0> alpha_dir;
  int<lower=1> q_dim;
}

parameters {
  vector[p] mu;              // Mean of log-volatility process
  vector<lower=0, upper=1>[p] sigmasq;  // Volatility of log-volatility process
  vector<lower=0, upper=1>[p] phi;  // Transformed persistence parameter
  matrix[T, p] h_std;        // Standardized log-volatility process
  vector[q_dim] meanar;
  vector<lower=0, upper=1>[q_dim] beta1; // Transformed AR(1) process persistence parameter
  vector<lower=0, upper=1>[q_dim] sigmasq_rho; // Volatility of AR(1) process for q
  matrix[T, q_dim] nu_raw; // Non-centered innovations for q
  real<lower=0> omega;
  real<lower=0> delta;
  simplex[k] durpar;
}

transformed parameters {
  matrix[T, p] mu_sv;
  matrix[T, p] h;
  for (j in 1:p) {
    mu_sv[1, j] = mu[j];
    h[,j]=h_std[,j]*sqrt(sigmasq[j]);
    h[1,j] /= sqrt(1 - phi[j]*phi[j]);
    h[1,j] += mu_sv[1, j];
    for (t in 2:T) {
      mu_sv[t, j] = mu[j] + phi[j]^(g[t]) * (h[t - 1, j] - mu[j]);
      h[t,j] *= sqrt((1 - phi[j]^(2*g[t])) / (1 - phi[j]*phi[j]));
      h[t,j] += mu_sv[t, j];
    }
  }

  matrix[q_dim, T] mu_ar;
  matrix[q_dim, T] q_ar; // Transformed lower triangular elements of L_t
    for (j in 1:(q_dim)) {
    mu_ar[j, 1] = meanar[j];
    q_ar[j,]=to_row_vector(nu_raw[,j]*sqrt(sigmasq_rho[j]));
    q_ar[j,1] /= sqrt(1 - beta1[j]*beta1[j]);
    q_ar[j,1] += mu_ar[j, 1];
    for (t in 2:T) {
      mu_ar[j, t] = meanar[j]+beta1[j]^(g[t]) * (q_ar[j, t - 1]-meanar[j]);
      q_ar[j,t] *= sqrt((1 - beta1[j]^(2*g[t])) / (1 - beta1[j]*beta1[j]));
      q_ar[j,t] += mu_ar[j, t];
    }
  }

  array[T] matrix[p, p] L;         // Time-varying Cholesky factor
  for (t in 1:T) {
    int idx = 1;
    for (i in 1:p) {
      for (j in 1:(i-1)) {
        L[t, i, j] = q_ar[idx, t];
        idx += 1;
      }
      L[t, i, i] = 1.0;
      for (j in (i+1):p) {
        L[t, i, j] = 0.0;
      }
    }
  }
}

model {
  vector[T] psi;         // Vector for the duration parameters
  
  delta ~ gamma(0.001, 0.001);
  omega ~ gamma(0.001, 0.001);
  durpar ~ dirichlet(rep_vector(alpha_dir, k));
  
  vector[T] gamma_llpd;
  
  // Model for the gaps
  for (t in 1:q) {
    psi[t] = omega;
    gamma_llpd[t] = gamma_lpdf(g[t]|delta, delta / psi[t]);
  }
  
  for (t in (q + 1):T) {
    psi[t] = omega;
    for (i in 1:q) {
      psi[t] += durpar[i] * g[t - i];
    }
    gamma_llpd[t] = gamma_lpdf(g[t]|delta, delta / psi[t]);
  }
  target += sum(gamma_llpd);
  
  // Priors
  mu ~ normal(0, 10);
  sigmasq ~ inv_gamma(1, 1);
  phi ~ beta(1, 1);
  meanar ~ normal(0, sqrt(10));
  beta1 ~ beta(1,1);
  sigmasq_rho ~ inv_gamma(2.5, 0.25);
  
  // Non-centered parameterization priors
  to_vector(h_std) ~ std_normal();
  to_vector(nu_raw) ~ std_normal();
  
  // Likelihood
  for (t in 1:T) {
  vector[p] h_t_exp = exp(to_vector(h[t, ])); // Convert row_vector to vector
  matrix[p, p] H_t = diag_matrix(h_t_exp); // Use vector to create diagonal matrix
  matrix[p, p] cov_mat = L[t] * H_t * L[t]';
  target += multi_normal_lpdf(y[t] | rep_vector(0, p), cov_mat); // Use cov_mat
}
}
