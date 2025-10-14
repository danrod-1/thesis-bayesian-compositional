data {
  int<lower=1> N;               // number of training observations
  int<lower=2> D;               // number of composition components
  int<lower=1> P;               // number of covariates
  matrix[N, P] X;               // design matrix (train)
  array[N] simplex[D] Y;        // compositional response
  int<lower=0> N_new;           // number of test observations
  matrix[N_new, P] X_new;       // design matrix (test)
}

parameters {
  matrix[D - 1, P] z;                 
  vector<lower=0>[P] lambda;          
  real<lower=0> tau_global;          
  real<lower=0> c2;                   
  real log_theta;                     
}

transformed parameters {
  matrix[D, P] B;                 
  array[N] vector[D] eta;
  array[N] simplex[D] mu;
  real theta = exp(log_theta);

  vector[P] lambda_tilde;
  for (j in 1:P)
    lambda_tilde[j] = sqrt(c2 * square(lambda[j]) / (c2 + square(tau_global * lambda[j])));

  for (k in 1:(D - 1))
    B[k] = z[k] .* lambda_tilde';
  B[D] = rep_row_vector(0, P);  // identifiability constraint

  for (i in 1:N)
    eta[i] = to_vector(X[i] * B');
  for (i in 1:N)
    mu[i] = softmax(eta[i]);
}

model {
  to_vector(z) ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_global ~ cauchy(0, 1);
  c2 ~ inv_gamma(1, 1);
  log_theta ~ normal(0, 3);

  for (i in 1:N)
    Y[i] ~ dirichlet(mu[i] * theta);
}

generated quantities {
  array[N] simplex[D] Y_rep;
  array[N_new] simplex[D] mu_pred;
  array[N_new] simplex[D] Y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    Y_rep[i] = dirichlet_rng(mu[i] * theta);
    log_lik[i] = dirichlet_lpdf(Y[i] | mu[i] * theta);
  }

  for (i in 1:N_new) {
    vector[D] eta_new = to_vector(X_new[i] * B');
    mu_pred[i] = softmax(eta_new);
    Y_pred[i] = dirichlet_rng(mu_pred[i] * theta);
  }
}



