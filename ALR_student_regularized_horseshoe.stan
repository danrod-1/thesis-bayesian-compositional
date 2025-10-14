functions {
  vector inverse_alr(vector z) {
    int D = num_elements(z) + 1;
    vector[D] y;
    real z_max = max(z);
    real denom = 0;

    for (d in 1:(D - 1)) {
      y[d] = exp(z[d] - z_max);
      denom += y[d];
    }

    y[D] = exp(-z_max);
    denom += y[D];

    for (d in 1:D) {
      y[d] = y[d] / denom;
    }

    return y;
  }
}



data {
  int<lower=1> N;               // number of training observations
  int<lower=1> D;               // ALR-transformed dimensionality
  int<lower=1> P;               // number of covariates
  matrix[N, P] X;               // design matrix (train)
  matrix[N, D] Z;               // ALR-transformed responses
  int<lower=0> N_new;           // number of test observations
  matrix[N_new, P] X_new;       // design matrix (test)
}

parameters {
  matrix[D, P] z;                          
  vector<lower=0>[P] lambda;                
  real<lower=0> tau_global;                 
  real<lower=0> c2;                         

  vector<lower=0>[D] tau;                   
  cholesky_factor_corr[D] L_corr;           

  real<lower=2> nu;                         
}

transformed parameters {
  matrix[D, P] B;                 
  matrix[N, D] mu;               
  matrix[D, D] L_Sigma;          
  vector[P] lambda_tilde;

  for (j in 1:P) {
    lambda_tilde[j] = sqrt(c2 * square(lambda[j]) / (c2 + square(tau_global * lambda[j])));
  }

  B = z;
  for (p in 1:P) {
    B[:, p] *= lambda_tilde[p];
  }

  L_Sigma = diag_pre_multiply(tau, L_corr);
  mu = X * B';
}

model {
  // Priors on degrees of freedom
  nu ~ gamma(2, 0.15);

 
  to_vector(z) ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_global ~ cauchy(0, 1);
  c2 ~ inv_gamma(1, 1);


  tau ~ normal(0, 1);
  L_corr ~ lkj_corr_cholesky(1.2);

 
  for (n in 1:N) {
    Z[n] ~ multi_student_t_cholesky(nu, mu[n], L_Sigma);
  }
}

generated quantities {
  matrix[N, D] Z_rep;
  matrix[N, D + 1] Y_rep;
  matrix[N_new, D] mu_pred;
  matrix[N_new, D] Z_pred;
  matrix[N_new, D + 1] Y_pred;
  matrix[D, D] Sigma;
  vector[N] log_lik;
  matrix[N, D + 1] Y_hat;
  matrix[N, D + 1] Y_hat_pred;


  Sigma = L_Sigma * L_Sigma';

  for (n in 1:N) {
    log_lik[n] = multi_student_t_cholesky_lpdf(Z[n] | nu, mu[n], L_Sigma);
    Z_rep[n] = (multi_student_t_cholesky_rng(nu, mu[n]', L_Sigma))';
    Y_rep[n] = inverse_alr(Z_rep[n]')';
    Y_hat[n] = inverse_alr(mu[n]')';
  }

  mu_pred = X_new * B';

  for (n in 1:N_new) {
    Z_pred[n] = (multi_student_t_cholesky_rng(nu, mu_pred[n]', L_Sigma))';
    Y_pred[n] = inverse_alr(Z_pred[n]')';
    Y_hat_pred[n] = inverse_alr(mu_pred[n]')';
  }
}



