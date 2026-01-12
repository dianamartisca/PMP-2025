import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

dummy_data = np.loadtxt('./date.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

order = 5

# Create polynomial features
x_1p = np.vstack([x_1**i for i in range(1, order+1)])

# Standardize features
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)

# Standardize target
y_1s = (y_1 - y_1.mean()) / y_1.std()

# Build the model
with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    epsilon = pm.HalfCauchy('epsilon', beta=5)
    
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
    
    trace_p = pm.sample(2000, tune=1000, return_inferencedata=True)

# Plot original data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data (Standardized)')

# a) Perform inference and plot the curve
plt.subplot(1, 2, 2)
plt.scatter(x_1s[0], y_1s, alpha=0.6, label='Data')

# Generate predictions for plotting
x_range = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

# We need to transform x_range back to original scale, create polynomial features, then standardize
x_range_original = x_range * x_1p.std(axis=1, keepdims=True)[0] + x_1p.mean(axis=1, keepdims=True)[0]
x_range_poly = np.vstack([x_range_original**i for i in range(1, order+1)])
x_range_std = (x_range_poly - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)

# Get posterior samples
alpha_post = trace_p.posterior['alpha'].values.flatten()
beta_post = trace_p.posterior['beta'].values.reshape(-1, order)

# Calculate predictions for each posterior sample
y_pred_samples = []
for i in range(len(alpha_post)):
    y_pred_i = alpha_post[i] + np.dot(beta_post[i], x_range_std)
    y_pred_samples.append(y_pred_i)

y_pred_samples = np.array(y_pred_samples)

# Plot mean prediction
y_pred_mean = y_pred_samples.mean(axis=0)
plt.plot(x_range, y_pred_mean, 'r-', linewidth=2, label='Posterior Mean')

# Plot credible interval
y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)
plt.fill_between(x_range, y_pred_lower, y_pred_upper, alpha=0.3, color='red', label='95% CI')

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Polynomial Regression (Order={order})')
plt.legend()
plt.tight_layout()
plt.show()

# b)
# Helper function to run model and return posterior predictive mean and CI
def run_poly_model(beta_sd, label, color):
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=beta_sd, shape=order)
        epsilon = pm.HalfCauchy('epsilon', beta=5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, progressbar=False)
    alpha_post = trace.posterior['alpha'].values.flatten()
    beta_post = trace.posterior['beta'].values.reshape(-1, order)
    y_pred_samples = []
    for i in range(len(alpha_post)):
        y_pred_i = alpha_post[i] + np.dot(beta_post[i], x_range_std)
        y_pred_samples.append(y_pred_i)
    y_pred_samples = np.array(y_pred_samples)
    y_pred_mean = y_pred_samples.mean(axis=0)
    y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)
    plt.plot(x_range, y_pred_mean, color=color, linewidth=2, label=label)
    plt.fill_between(x_range, y_pred_lower, y_pred_upper, alpha=0.15, color=color)
    return trace

# Plot all curves for comparison
plt.figure(figsize=(10, 6))
plt.scatter(x_1s[0], y_1s, alpha=0.5, label='Data', color='k')

# Original (sd=10)
trace_10 = run_poly_model(10, 'beta sd=10', 'red')
# sd=100
trace_100 = run_poly_model(100, 'beta sd=100', 'blue')
# sd=[10,0.1,0.1,0.1,0.1]
trace_vec = run_poly_model(np.array([10,0.1,0.1,0.1,0.1]), 'beta sd=[10,0.1,...]', 'green')

plt.xlabel('x (standardized)')
plt.ylabel('y (standardized)')
plt.title('Effect of beta prior sd on polynomial fit')
plt.legend()
plt.tight_layout()
plt.show()

print('Summary for beta sd=10:')
print(az.summary(trace_10))
print('\nSummary for beta sd=100:')
print(az.summary(trace_100))
print('\nSummary for beta sd=[10,0.1,0.1,0.1,0.1]:')
print(az.summary(trace_vec))
