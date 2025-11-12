import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])
x_bar = data.mean()
print(f"Sample mean = {x_bar:.2f}, Sample std = {data.std(ddof=1):.2f}")

# a) Model with weak prior
with pm.Model() as weak_model:
    mu = pm.Normal("mu", mu=x_bar, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
    
    trace_weak = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)
    summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)

# b) 95% HDI
print("\nPosterior summaries (Weak Prior):")
print(summary_weak)

# c) Compare with frequentist estimates
print("\nFrequentist estimates:")
print(f"Mean: {np.mean(data):.2f}")
print(f"SD:   {np.std(data, ddof=1):.2f}")

# d) Model with strong prior
with pm.Model() as strong_model:
    mu = pm.Normal("mu", mu=50, sigma=1)  # strong prior mu ~ N(50, 1*1)
    sigma = pm.HalfNormal("sigma", sigma=10)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
    
    trace_strong = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)
    summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)

print("\nPosterior summaries (Strong Prior):")
print(summary_strong)

az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
plt.suptitle("Posterior with Weak Prior", fontsize=14)
plt.show()

az.plot_posterior(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
plt.suptitle("Posterior with Strong Prior", fontsize=14)
plt.show()
