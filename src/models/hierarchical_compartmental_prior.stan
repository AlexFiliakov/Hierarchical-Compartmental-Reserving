
functions {
  // ODE system for compartmental model
  vector compartmentmodel(real t, vector y, vector theta) {
    vector[3] dydt;
    dydt[1] = -theta[1] * y[1];                           // exposure 
    dydt[2] = theta[1] * theta[3] * y[1] - theta[2] * y[2]; // outstanding
    dydt[3] = theta[2] * theta[4] * y[2];                   // paid
    return dydt;
  }

  // Incremental claims process function
  real incrclaimsprocess(real t, real devfreq, real ker, real kp, 
                         real RLR, real RRF, real delta) {
    vector[3] y0;
    array[1] vector[3] y;
    vector[4] theta;
    theta[1] = ker; 
    theta[2] = kp;
    theta[3] = RLR; 
    theta[4] = RRF;
    real out; 

    // Set initial values
    y0[1] = 1; // Exposure
    y0[2] = 0; // Outstanding
    y0[3] = 0; // Paid

    y = ode_rk45(compartmentmodel, y0, 0, rep_array(t, 1), theta);
    out = y[1, 2] * (1 - delta) + y[1, 3] * delta;

    if ((delta > 0) && (t > devfreq)) { // paid greater dev period 1
      // incremental paid
      y = ode_rk45(compartmentmodel, y0, 0, rep_array(t - devfreq, 1), theta);
      out = out - y[1, 3];
    }
    return(out);
  }
}

data {
  int<lower=1> N;                    // Number of observations
  int<lower=1> n_accident_years;     // Number of accident years
  array[N] int accident_year_id;    // Accident year index for each obs
  vector[N] dev;                     // Development period
  vector[N] delta;                   // 0 for outstanding, 1 for paid
  vector[n_accident_years] RateIndex; // Rate indices by accident year
}

generated quantities {
  // Sample parameters from priors
  real oker = normal_rng(0, 1);
  real okp = normal_rng(0, 1);

  // Population-level parameters for hierarchical structure
  real oRLR_mu = normal_rng(0, 1);
  real oRRF_mu = normal_rng(0, 1);
  real<lower=0> oRLR_sd = abs(student_t_rng(10, 0, 0.05));
  real<lower=0> oRRF_sd = abs(student_t_rng(10, 0, 0.05));

  // Generate correlation matrix
  cholesky_factor_corr[2] L_Omega = lkj_corr_cholesky_rng(2, 1);

  // Generate random effects
  vector[n_accident_years] oRLR_raw;
  vector[n_accident_years] oRRF_raw;
  for (i in 1:n_accident_years) {
    oRLR_raw[i] = normal_rng(0, 1);
    oRRF_raw[i] = normal_rng(0, 1);
  }

  // Transform to correlated random effects
  matrix[2, n_accident_years] random_effects;
  random_effects[1] = oRLR_raw';
  random_effects[2] = oRRF_raw';
  random_effects = diag_pre_multiply([oRLR_sd, oRRF_sd]', L_Omega) * random_effects;

  vector[n_accident_years] oRLR = oRLR_mu + random_effects[1]';
  vector[n_accident_years] oRRF = oRRF_mu + random_effects[2]';

  // Sample observation-level parameters
  // In brms with link_sigma = "log", the priors are on log(sigma)
  real log_sigma_outstanding = normal_rng(log(0.05), 0.02);
  real log_sigma_paid = normal_rng(log(0.1), 0.05);
  real<lower=0> sigma_outstanding = exp(log_sigma_outstanding);
  real<lower=0> sigma_paid = exp(log_sigma_paid);

  // Generate prior predictive samples
  vector[N] loss_ratio_rep;

  for (n in 1:N) {
    real ker = 0.1 * exp(oker * 0.05);
    real kp = 0.5 * exp(okp * 0.025);
    real RLR = 0.55 * exp(oRLR[accident_year_id[n]] * 0.025) / RateIndex[accident_year_id[n]];
    real RRF = exp(oRRF[accident_year_id[n]] * 0.05);

    real eta = incrclaimsprocess(dev[n], 1.0, ker, kp, RLR, RRF, delta[n]);
    real sigma = delta[n] == 0 ? sigma_outstanding : sigma_paid;

    loss_ratio_rep[n] = lognormal_rng(log(eta), sigma);
  }
}
