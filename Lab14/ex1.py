import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('date_colesterol.csv')
X = data['Ore_Exercitii'].values.reshape(-1, 1)
y = data['Colesterol'].values

# Prepare polynomial features (quadratic)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

def fit_mixture_of_regressions(X_poly, y, K):
	# Fit a Gaussian Mixture Model to the targets (cholesterol)
	gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
	# Use X_poly and y to initialize responsibilities
	# We'll use a simple EM-like approach
	N = len(y)
	resp = np.ones((N, K)) / K 
	means = np.zeros((K, X_poly.shape[1]))
	variances = np.ones(K)
	weights = np.ones(K) / K
	for iteration in range(20):
		# M-step: Fit regression for each cluster
		for k in range(K):
			# Weighted regression
			reg = LinearRegression()
			reg.fit(X_poly, y, sample_weight=resp[:, k])
			means[k] = reg.coef_
			intercept = reg.intercept_
			y_pred = reg.predict(X_poly)
			variances[k] = np.average((y - y_pred) ** 2, weights=resp[:, k])
		# E-step: Update responsibilities
		for k in range(K):
			reg = LinearRegression()
			reg.coef_ = means[k]
			reg.intercept_ = 0 
			y_pred = X_poly @ means[k]
			# Gaussian likelihood
			resp[:, k] = weights[k] * (1.0 / np.sqrt(2 * np.pi * variances[k])) * np.exp(-0.5 * (y - y_pred) ** 2 / variances[k])
		resp = resp / resp.sum(axis=1, keepdims=True)
		# Update weights
		weights = resp.mean(axis=0)
	# After EM, fit final regressions for each cluster
	coefs = []
	intercepts = []
	for k in range(K):
		reg = LinearRegression()
		reg.fit(X_poly, y, sample_weight=resp[:, k])
		coefs.append(reg.coef_)
		intercepts.append(reg.intercept_)
	return weights, intercepts, coefs, variances


# Model selection using WAIC and LOO
def log_likelihood_point(y_true, means, variances, weights, X_poly):
	# Compute the log-likelihood for each data point under the mixture
	N = len(y_true)
	K = len(weights)
	log_lik = np.zeros((N, K))
	for k in range(K):
		mu = X_poly @ means[k]
		var = variances[k]
		log_lik[:, k] = np.log(weights[k] + 1e-12) + (-0.5 * np.log(2 * np.pi * var) - 0.5 * (y_true - mu) ** 2 / var)
	# LogSumExp for mixture
	return np.logaddexp.reduce(log_lik, axis=1)

def waic(log_lik_points):
	lppd = np.sum(log_lik_points)
	p_waic = np.var(log_lik_points)
	return -2 * (lppd - p_waic)

def loo(log_lik_points):
	# Approximate LOO using log-likelihood points (not exact, but informative)
	return -2 * np.sum(log_lik_points)

waic_scores = []
loo_scores = []
for K in [3, 4, 5]:
	weights, intercepts, coefs, variances = fit_mixture_of_regressions(X_poly, y, K)
	means = np.stack(coefs)
	# Compute log-likelihood for each data point
	log_lik_points = log_likelihood_point(y, means, variances, weights, X_poly)
	waic_score = waic(log_lik_points)
	loo_score = loo(log_lik_points)
	waic_scores.append(waic_score)
	loo_scores.append(loo_score)
	print(f"\nK = {K}")
	for k in range(K):
		print(f"Subpopulation {k+1}:")
		print(f"  Weight: {weights[k]:.3f}")
		print(f"  Regression: Colesterol = {intercepts[k]:.2f} + {coefs[k][0]:.2f}*t + {coefs[k][1]:.2f}*t^2")
		print(f"  Variance: {variances[k]:.2f}")
	print(f"  WAIC: {waic_score:.2f}")
	print(f"  LOO: {loo_score:.2f}")

# Model selection summary and best K printout (after all K are processed)
print("\nModel selection summary:")
for i, K in enumerate([3, 4, 5]):
	print(f"K={K}: WAIC={waic_scores[i]:.2f}, LOO={loo_scores[i]:.2f}")
if len(set(np.round(waic_scores, 2))) == 1 and len(set(np.round(loo_scores, 2))) == 1:
	print("\nAll values of K provide identical WAIC and LOO scores. Any K (3, 4, or 5) can be considered equally best for this data.")
else:
	best_K_waic = [3, 4, 5][np.argmin(waic_scores)]
	best_K_loo = [3, 4, 5][np.argmin(loo_scores)]
	print(f"\nBest K by WAIC: {best_K_waic}")
	print(f"Best K by LOO: {best_K_loo}")
