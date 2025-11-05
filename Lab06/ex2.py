"""
Lab 6, Exercise 2: Bayesian Inference for Call Center Rate

Problem: Estimate average call rate λ using Poisson-Gamma conjugate prior.
- Observed: 180 calls over 10 hours
- Likelihood: Poisson(λ) for number of calls per hour
- Prior: Gamma(α, β) - conjugate prior for Poisson
- Posterior: Gamma(α_post, β_post) - computed analytically

Tasks:
a) Determine the posterior distribution of λ
b) Calculate 94% HDI (Highest Density Interval)
c) Find the most probable value (mode) of λ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az

# ============================================================================
# Observed Data
# ============================================================================
total_calls = 180
total_hours = 10
observed_rate = total_calls / total_hours  # 18 calls per hour

print("="*70)
print("OBSERVED DATA")
print("="*70)
print(f"Total calls: {total_calls}")
print(f"Total hours: {total_hours}")
print(f"Observed average rate: {observed_rate} calls/hour\n")

# ============================================================================
# Prior Distribution: Gamma(α, β)
# ============================================================================
# We use a weakly informative prior: Gamma(1, 1)
# - Mean = α/β = 1/1 = 1
# - Variance = α/β² = 1/1 = 1
# This is a common choice when we have little prior knowledge

alpha_prior = 1
beta_prior = 1

print("="*70)
print("PRIOR DISTRIBUTION")
print("="*70)
print(f"Prior: Gamma(α={alpha_prior}, β={beta_prior})")
print(f"Prior mean: {alpha_prior/beta_prior}")
print(f"Prior mean: {alpha_prior/beta_prior}")
print(f"Prior variance: {alpha_prior/beta_prior**2}\n")

# ============================================================================
# a) Posterior Distribution (Conjugate Prior Property)
# ============================================================================
# For Poisson likelihood with Gamma prior, the posterior is also Gamma:
# Posterior ~ Gamma(α_prior + total_calls, β_prior + total_hours)

alpha_post = alpha_prior + total_calls
beta_post = beta_prior + total_hours

print("="*70)
print("a) POSTERIOR DISTRIBUTION")
print("="*70)
print(f"Posterior: Gamma(α={alpha_post}, β={beta_post})")
print(f"Posterior mean: {alpha_post/beta_post:.4f}")
print(f"Posterior variance: {alpha_post/beta_post**2:.4f}")
print(f"Posterior std dev: {np.sqrt(alpha_post/beta_post**2):.4f}\n")

# Create posterior distribution object
posterior_dist = stats.gamma(a=alpha_post, scale=1/beta_post)

# ============================================================================
# b) 94% HDI (Highest Density Interval)
# ============================================================================
# Sample from posterior and compute HDI using ArviZ
posterior_samples = posterior_dist.rvs(size=10000, random_state=42)
hdi_94 = az.hdi(posterior_samples, hdi_prob=0.94)

print("="*70)
print("b) 94% HDI (HIGHEST DENSITY INTERVAL)")
print("="*70)
print(f"94% HDI: [{hdi_94[0]:.4f}, {hdi_94[1]:.4f}]")
print(f"Interpretation: We are 94% confident that λ is between")
print(f"                {hdi_94[0]:.4f} and {hdi_94[1]:.4f} calls/hour\n")

# ============================================================================
# c) Most Probable Value (Mode)
# ============================================================================
# Mode of Gamma(α, β) = (α - 1) / β for α ≥ 1

if alpha_post >= 1:
    mode = (alpha_post - 1) / beta_post
else:
    mode = 0

print("="*70)
print("c) MOST PROBABLE VALUE (MODE)")
print("="*70)
print(f"Mode of posterior: {mode:.4f} calls/hour")
print(f"Interpretation: The most probable call rate is {mode:.4f}\n")

# ============================================================================
# Visualization
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

lambda_range = np.linspace(0, 30, 1000)
prior_pdf = stats.gamma(a=alpha_prior, scale=1/beta_prior).pdf(lambda_range)
posterior_pdf = stats.gamma(a=alpha_post, scale=1/beta_post).pdf(lambda_range)

# Plot 1: Prior vs Posterior
axes[0].plot(lambda_range, prior_pdf, 'b--', label=f'Prior: Gamma({alpha_prior}, {beta_prior})', linewidth=2)
axes[0].plot(lambda_range, posterior_pdf, 'r-', label=f'Posterior: Gamma({alpha_post}, {beta_post})', linewidth=2)
axes[0].axvline(observed_rate, color='green', linestyle=':', label=f'Observed rate: {observed_rate}', linewidth=2)
axes[0].axvline(mode, color='orange', linestyle='-.', label=f'Mode: {mode:.2f}', linewidth=2)
axes[0].set_xlabel('λ (calls per hour)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Prior and Posterior Distributions', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Posterior with HDI
axes[1].plot(lambda_range, posterior_pdf, 'r-', linewidth=2, label='Posterior')
axes[1].axvline(mode, color='orange', linestyle='-.', label=f'Mode: {mode:.2f}', linewidth=2)
axes[1].axvline(hdi_94[0], color='purple', linestyle='--', label=f'94% HDI: [{hdi_94[0]:.2f}, {hdi_94[1]:.2f}]', linewidth=1.5)
axes[1].axvline(hdi_94[1], color='purple', linestyle='--', linewidth=1.5)
axes[1].fill_between(lambda_range, 0, posterior_pdf, 
                     where=(lambda_range >= hdi_94[0]) & (lambda_range <= hdi_94[1]),
                     alpha=0.3, color='purple', label='94% HDI region')
axes[1].set_xlabel('λ (calls per hour)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Posterior Distribution with 94% HDI', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
