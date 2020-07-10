functions{
    real gev_lpdf(real y, real mu, real sigma, real xi) {
        real t;
        real lp;
        t = xi==0 ? exp((mu - y) / sigma) : pow(1 + xi * ((y - mu ) / sigma), -1 / xi);
        lp = -log(sigma) + (xi + 1) * log(t) - t;
        return lp;
    }
    real gev_rng(real mu, real sigma, real xi){
        real p;
        real z;
        real epsilon;
        p = uniform_rng(0, 1);
        epsilon = 0.0001;
        z = fabs(xi) < epsilon ? -log(-log(p)): expm1(-xi * log(-log(p))) / xi;
        return mu + sigma * z;
    }
}

data {
    int<lower=0> J; // number of sites
    int<lower=0> N; // number of observations
    real<lower=0> log_area[J]; // area of each watershed
    real flood[N]; // the annual-maximum floods
    int site[N]; // the index of each data point
    // pass priors as data to avoid recompiling
    real alpha_1_std;
    real alpha_2_std;
    real beta_1_std;
    real beta_2_std;
    real xi_bar_std;
    real tau_mu_std;
    real tau_sigma_std;
    real tau_xi_std;
}

parameters {
    // scale, loc, shape: one per site
    real<lower=0> mu[J];
    real<lower=0> sigma[J];
    real xi[J];
    // hyperparameters
    real alpha_1;
    real alpha_2;
    real beta_1;
    real beta_2;
    real xi_bar;
    real<lower=0> tau_mu;
    real<lower=0> tau_sigma;
    real<lower=0> tau_xi;
}

model {
    // data model
    for (n in 1:N){
        flood[n] ~ gev(mu[site[n]], sigma[site[n]], xi[site[n]]);
    }
    // hierarchical model = note that stan uses stdev not variance for normal dist
    for (j in 1:J){
        mu[j] ~ lognormal(alpha_1 + alpha_2 * log_area[j], tau_mu);
        sigma[j] ~ lognormal(beta_1 + beta_2 * log_area[j], tau_sigma);
        xi[j] ~ normal(xi_bar, tau_xi);
    }
    // hyperpriors
    alpha_1 ~ normal(0, alpha_1_std);
    alpha_2 ~ normal(0, alpha_2_std);
    beta_1 ~ normal(0, beta_1_std);
    beta_2 ~ normal(0, beta_2_std);
    xi_bar ~ normal(0, xi_bar_std);
    tau_mu ~ normal(0, tau_mu_std);
    tau_sigma ~ normal(0, tau_sigma_std);
    tau_xi ~ normal(0, tau_xi_std);
}

generated quantities {
    real log_lik[N];
    real y_hat[N];
    for (n in 1:N) {
        log_lik[n] = gev_lpdf(flood[n] | mu[site[n]], sigma[site[n]], xi[site[n]]);
        y_hat[n] = gev_rng(mu[site[n]], sigma[site[n]], xi[site[n]]);
    }
}