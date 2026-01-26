import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('bike_daily.csv')
print(df.head())

'''
1. Briefly explore relationships between predictors and renatls: pairwise scatterplots and comment on likely nonlinearitis (on exam paper).
'''
numeric_cols = ["rentals", "temp_c", "humidity", "wind_kph"]
sns.pairplot(df[numeric_cols])
plt.show()

'''
2.a) Standardize continuos predictors: temp_c, humidity, wind_kph and the target rentals.
'''
# Select relevant columns and dummies for season
df = pd.get_dummies(df, columns=['season'], drop_first=True)
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# Define which columns to standardize
to_scale = ['rentals', 'temp_c', 'humidity', 'wind_kph']
scaler = StandardScaler()
df[[f'{col}_z' for col in to_scale]] = scaler.fit_transform(df[to_scale])

'''
2.b) Build a Bayesian linear regression model for rentals using PYMC.
'''
# Prepare predictors/target
X = df[['temp_c_z', 'humidity_z', 'wind_kph_z', 'is_holiday'] + 
       [col for col in df.columns if col.startswith('season_')]].values
y = df['rentals_z'].values

# Model
with pm.Model() as linear_model:
    # Priors for coefficients and intercept
    beta = pm.Normal('beta', mu=0, sigma=1, shape=X.shape[1])
    intercept = pm.Normal('intercept', mu=0, sigma=1)
    sigma = pm.Exponential('sigma', 1)
    
    # Expected value
    mu = intercept + pm.math.dot(X, beta)
    
    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    
    # Sample from posterior
    trace_linear = pm.sample(
        draws=1500,  # Between 1000-2000
        tune=1000,
        chains=3,    # Between 2-4
        target_accept=0.9,
        random_seed=42
    )

'''
2.c) Build a polynomial regression model, similar to 2.b), by adding an extra term temp_c^2.
'''
# Add polynomial term temp_c^2 for standardized temperature
df['temp_c_z2'] = df['temp_c_z'] ** 2

# Prepare predictors/target for the polynomial model (include temp_c_z2 term now)
X_poly = df[['temp_c_z', 'temp_c_z2', 'humidity_z', 'wind_kph_z', 'is_holiday'] +
            [col for col in df.columns if col.startswith('season_')]].values
y = df['rentals_z'].values

# Model
with pm.Model() as poly_model:
    # Priors for coefficients and intercept
    beta = pm.Normal('beta', mu=0, sigma=1, shape=X_poly.shape[1])
    intercept = pm.Normal('intercept', mu=0, sigma=1)
    sigma = pm.Exponential('sigma', 1)
    
    # Expected value
    mu = intercept + pm.math.dot(X_poly, beta)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    
    # Sample from posterior
    trace_poly = pm.sample(
        draws=1500,  # Between 1000-2000
        tune=1000,
        chains=3,    # Between 2-4
        target_accept=0.9,
        random_seed=42
    )

'''
3. Use pm.sample() with sensible settings (2-4 chains, 1000-2000 draws, target_accept=0.9) and summarize posteriors. 
Which variable do you think that influences the most the result (on exam paper).
'''
# Summarize linear model
print(az.summary(trace_linear, var_names=['beta', 'intercept'], round_to=2))

# Summarize polynomial model
print(az.summary(trace_poly, var_names=['beta', 'intercept'], round_to=2))

'''
On the same dataset, create a binary target for high demand and fit a Bayesian logistic regression model to estimate the probability of high demand.
5. Construct the binary target: define is_high_demand=1 if rentals >= Q, 
where Q is 75th percentile of rentals (computed on the original scale, pre-standardization).
'''
Q = np.percentile(df['rentals'], 75)
df['is_high_demand'] = (df['rentals'] >= Q).astype(int)
print(f"Q (75th percentile of rentals): {Q}")
print(df['is_high_demand'].value_counts())

'''
6. Build a Bayesian logistic regression model for rentals using PYMC and the same standardized festures. 
You may keep or drop temp_c^2 - justify your choice.
'''
# Prepare standardized features
predictor_cols = ['temp_c_z', 'temp_c_z2', 'humidity_z', 'wind_kph_z', 'is_holiday'] + [
    col for col in df.columns if col.startswith('season_')
]
X_logit = df[predictor_cols].values.astype(float)
y_logit = df['is_high_demand'].values

with pm.Model() as logit_model:
    beta = pm.Normal('beta', mu=0, sigma=1, shape=X_logit.shape[1])
    intercept = pm.Normal('intercept', mu=0, sigma=1)
    logits = intercept + pm.math.dot(X_logit, beta)
    y_obs = pm.Bernoulli('y_obs', logit_p=logits, observed=y_logit)
    trace_logit = pm.sample(
        draws=1500,
        tune=1000,
        chains=3,
        target_accept=0.9,
        random_seed=42
    )

summary = az.summary(trace_logit, var_names=['beta', 'intercept'], round_to=2)
print(summary)

'''
I chose to include the quadratic term temp_c² in the logistic regression because temperature is likely to have a 
nonlinear effect on rental demand: extremely high or low values may reduce demand, while moderate temperatures increase it. 
Including temp_c² allows the model to capture this possible nonlinearity.
'''

'''
7. Obtain 95% HDI estimates for the parameters. Which variable do you think influences the outcome the most? (on exam paper).
'''
summary = az.summary(trace_logit, var_names=['beta', 'intercept'], hdi_prob=0.95, round_to=2)
print(summary)