import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
lambdas = [1, 2, 5, 10]

#1
X = {f"X{i+1} (λ={l})": np.random.poisson(lam=l, size=1000) for i, l in enumerate(lambdas)}

df = pd.DataFrame(X)

print(df.head())

print("\nSummary Statistics:")
print(df.describe())

#2
chosen_lambdas = np.random.choice(lambdas, size=1000, replace=True)

values = np.random.poisson(lam=chosen_lambdas)

df = pd.DataFrame({
    'Chosen λ': chosen_lambdas,
    'Poisson Value': values
})

print(df.head())

print("\nAverage value by λ:")
print(df.groupby('Chosen λ')['Poisson Value'].mean())

#a
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
bins = range(0, max(values) + 2)  

# Plot the 4 fixed-λ histograms
for ax, (label, data) in zip(axes[:4], X.items()):
    ax.hist(data, bins=bins, density=True, alpha=0.7, edgecolor='black')
    ax.set_title(f"Poisson({label})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

# Plot the randomized λ
axes[4].hist(values, bins=bins, density=True, color='orange', alpha=0.7, edgecolor='black')
axes[4].set_title("Randomized λ")
axes[4].set_xlabel("Value")

plt.tight_layout()
plt.show()

probabilities_biased = [0.1, 0.1, 0.6, 0.2]   # λ=5 much more likely

chosen_lambdas_biased = np.random.choice(lambdas, size=1000, replace=True, p=probabilities_biased)
values_biased = np.random.poisson(lam=chosen_lambdas_biased)

df_biased = pd.DataFrame({
    'Chosen λ': chosen_lambdas_biased,
    'Poisson Value': values_biased
})

print("\nAverage value by λ (biased to λ=5):")
print(df_biased.groupby('Chosen λ')['Poisson Value'].mean())