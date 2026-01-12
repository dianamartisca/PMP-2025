import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

# Generate data (same as before for consistency)
np.random.seed(42)
x_1 = np.linspace(-1.5, 1.5, 100)
y_1 = 0.5 + 2*x_1 - 3*x_1**2 + 1.5*x_1**3 - 0.5*x_1**4 + 0.2*x_1**5 + np.random.normal(0, 0.5, 100)

orders = [1, 2, 3]  # Linear, quadratic, cubic
traces = []
waics = []
loos = []
labels = ['Linear', 'Quadratic', 'Cubic']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(10, 6))
plt.scatter(x_1, y_1, alpha=0.5, label='Data', color='k')

x_range = np.linspace(x_1.min(), x_1.max(), 100)

for idx, order in enumerate(orders):
    # Polynomial features
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    x_range_poly = np.vstack([x_range**i for i in range(1, order+1)])
    x_range_std = (x_range_poly - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        epsilon = pm.HalfCauchy('epsilon', beta=5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y = pm.Normal('y', mu=mu, sigma=epsilon, observed=y_1s)
        trace = pm.sample(
            2000, tune=1000, return_inferencedata=True, progressbar=False
        )
        traces.append(trace)
        waics.append(az.waic(trace))
        loos.append(az.loo(trace))

    # Posterior predictive
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
    plt.plot(x_range, y_pred_mean * y_1.std() + y_1.mean(), color=colors[idx], label=f'{labels[idx]} mean')
    plt.fill_between(x_range, y_pred_lower * y_1.std() + y_1.mean(), y_pred_upper * y_1.std() + y_1.mean(), alpha=0.15, color=colors[idx])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Model Comparison')
plt.legend()
plt.tight_layout()
plt.show()

for idx, order in enumerate(orders):
    print(f"{labels[idx]} model (order={order}):")
    print("  WAIC:", waics[idx].waic)
    print("  LOO:", loos[idx].loo)
    print()

print("Lower WAIC/LOO values indicate better model fit (penalized for complexity).")
