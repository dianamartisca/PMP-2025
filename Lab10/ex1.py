import numpy as np
import pymc as pm
import arviz as az

# Data: advertising expenses (in thousands) and weekly sales revenue (in thousands)
publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 
                      6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0, 
                  15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

# New advertising expense levels to predict
new_publicity = np.array([3.0, 6.5, 9.0, 12.0])

with pm.Model() as model:
    x = pm.Data("publicity", publicity)
    
    # Priors for intercept, slope, and noise
    alpha = pm.Normal("alpha", mu=10, sigma=10)  # intercept
    beta = pm.Normal("beta", mu=2, sigma=5)      # slope
    sigma = pm.HalfNormal("sigma", sigma=5)      # noise/uncertainty
    
    # Linear model
    mu = alpha + beta * x
    
    # Likelihood
    pm.Normal("sales", mu=mu, sigma=sigma, observed=sales)
    
    # Sample from posterior
    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)

# Get posterior means
alpha_mean = idata.posterior["alpha"].mean().values
beta_mean = idata.posterior["beta"].mean().values

print(f"\nIntercept (alpha): {alpha_mean:.4f}")
print(f"Slope (beta):      {beta_mean:.4f}")

# 94% HDIs for coefficients
coef_hdi = az.hdi(idata, var_names=["alpha", "beta"], hdi_prob=0.94)
print("\nCoefficient HDIs (94%):")
print(coef_hdi)

# Predict future revenues for new advertising expenses
# Extract posterior samples for alpha, beta, and sigma
alpha_samples = idata.posterior["alpha"].values.flatten()
beta_samples = idata.posterior["beta"].values.flatten()
sigma_samples = idata.posterior["sigma"].values.flatten()

# Generate predictions for each new advertising level
rng = np.random.default_rng(42)
predictions = []

for pub in new_publicity:
    # Compute mean prediction from linear model for each posterior sample
    mu_pred = alpha_samples + beta_samples * pub
    # Add noise according to sigma
    pred_samples = mu_pred + rng.normal(0, sigma_samples, size=len(mu_pred))
    predictions.append(pred_samples)

predictions = np.array(predictions)  

# 94% predictive intervals for forecasted sales
pred_int = np.percentile(predictions, [3, 97], axis=1)
print("\nPredictive intervals (94% HDI) for new advertising levels:")
print(f"{'Publicity ($k)':<20} {'Lower Bound':<15} {'Upper Bound':<15} {'Mean Prediction':<15}")

for i, pub in enumerate(new_publicity):
    lo = pred_int[0][i]
    hi = pred_int[1][i]
    mean_pred = predictions[i].mean()
    print(f"${pub:>6.1f}k{'':<13} ${lo:>6.2f}k{'':<8} ${hi:>6.2f}k{'':<8} ${mean_pred:>6.2f}k")

print(f"\nFor every 1k increase in advertising, sales increase by ~{beta_mean:.2f}k")
print(f"Base sales (no advertising) would be ~{alpha_mean:.2f}k")