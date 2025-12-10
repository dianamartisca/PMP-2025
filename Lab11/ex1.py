import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# Load the data
df = pd.read_csv('Prices.csv')

# Prepare the data
y = df['Price'].values  # Sale price
x1 = df['Speed'].values  # Processor frequency in MHz
x2 = np.log(df['HardDrive'].values)  # Natural log of hard disk size in MB

# a) Build the Bayesian regression model with weakly informative priors
with pm.Model() as model:
    speed_data = pm.Data("speed", x1)
    log_hd_data = pm.Data("log_hd", x2)
    
    # Weakly informative priors
    alpha = pm.Normal("alpha", mu=0, sigma=1000)
    beta1 = pm.Normal("beta1", mu=0, sigma=1000)
    beta2 = pm.Normal("beta2", mu=0, sigma=1000)
    sigma = pm.HalfNormal("sigma", sigma=1000)
    
    # Linear mode
    mu = alpha + beta1 * speed_data + beta2 * log_hd_data
    
    # Likelihood
    pm.Normal("price", mu=mu, sigma=sigma, observed=y)
    
    # Sample from posterior
    idata = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42)

# Get posterior means
alpha_mean = idata.posterior["alpha"].mean().values
beta1_mean = idata.posterior["beta1"].mean().values
beta2_mean = idata.posterior["beta2"].mean().values
sigma_mean = idata.posterior["sigma"].mean().values

print(f"\nalpha (Intercept): {alpha_mean:.2f}")
print(f"beta1 (Processor frequency): {beta1_mean:.4f}")
print(f"beta2 (Log hard disk size): {beta2_mean:.2f}")
print(f"sigma (Noise): {sigma_mean:.2f}")

# b) Obtain 95% HDI estimates
coef_hdi = az.hdi(idata, var_names=["beta1", "beta2"], hdi_prob=0.95)
print(f"\n95% HDI for beta1: [{coef_hdi['beta1'].values[0]:.4f}, {coef_hdi['beta1'].values[1]:.4f}]")
print(f"95% HDI for beta2: [{coef_hdi['beta2'].values[0]:.2f}, {coef_hdi['beta2'].values[1]:.2f}]")

# c) Are processor frequency and hard disk size useful predictors?

beta1_hdi = coef_hdi['beta1'].values
beta2_hdi = coef_hdi['beta2'].values

beta1_useful = not (beta1_hdi[0] <= 0 <= beta1_hdi[1])
beta2_useful = not (beta2_hdi[0] <= 0 <= beta2_hdi[1])

print(f"\nProcessor frequency useful? {beta1_useful} (HDI excludes zero: {beta1_useful})")
print(f"Hard disk size useful? {beta2_useful} (HDI excludes zero: {beta2_useful})")

# d) Predict price for computer with 33 MHz and 540 MB hard disk (90% HDI)

speed_new = 33
hd_new = 540
log_hd_new = np.log(hd_new)

# Extract posterior samples
alpha_samples = idata.posterior["alpha"].values.flatten()
beta1_samples = idata.posterior["beta1"].values.flatten()
beta2_samples = idata.posterior["beta2"].values.flatten()
sigma_samples = idata.posterior["sigma"].values.flatten()

# Compute expected price
mu_samples = alpha_samples + beta1_samples * speed_new + beta2_samples * log_hd_new
mu_hdi = np.percentile(mu_samples, [5, 95])

print(f"\nd) Expected price for {speed_new} MHz, {hd_new} MB:")
print(f"   Mean: ${mu_samples.mean():.2f}")
print(f"   90% HDI: [${mu_hdi[0]:.2f}, ${mu_hdi[1]:.2f}]")

# e) Predict sale price with observation noise
rng = np.random.default_rng(42)
price_samples = mu_samples + rng.normal(0, sigma_samples, size=len(mu_samples))
price_hdi = np.percentile(price_samples, [5, 95])

print(f"\ne) Sale price prediction for {speed_new} MHz, {hd_new} MB:")
print(f"   Mean: ${price_samples.mean():.2f}")
print(f"   90% HDI: [${price_hdi[0]:.2f}, ${price_hdi[1]:.2f}]")
