import os
os.environ['PYTENSOR_FLAGS'] = 'cxx='

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

if __name__ == '__main__':
    # Load data from CSV
    df = pd.read_csv('date_promovare_examen.csv')
    
    study_hours = df.iloc[:, 0].values
    sleep_hours = df.iloc[:, 1].values
    passed = df.iloc[:, 2].values.astype(int)
    
    # Standardize predictors for better sampling
    study_hours_std = (study_hours - study_hours.mean()) / study_hours.std()
    sleep_hours_std = (sleep_hours - sleep_hours.mean()) / sleep_hours.std()
    
    # a) Build logistic regression model in PyMC
    with pm.Model() as model:
        # Weakly informative priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_study = pm.Normal("beta_study", mu=0, sigma=10)
        beta_sleep = pm.Normal("beta_sleep", mu=0, sigma=10)
        
        # Logistic regression model
        logit_p = alpha + beta_study * study_hours_std + beta_sleep * sleep_hours_std
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        
        # Bernoulli likelihood
        pm.Bernoulli("passed", p=p, observed=passed)
        
        # Sample from posterior
        idata = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42)
    
    # Get posterior means
    alpha_mean = idata.posterior["alpha"].mean().values
    beta_study_mean = idata.posterior["beta_study"].mean().values
    beta_sleep_mean = idata.posterior["beta_sleep"].mean().values
    
    print(f"\nalpha: {alpha_mean:.4f}, beta1: {beta_study_mean:.4f}, beta2: {beta_sleep_mean:.4f}")
    
    # b) Check if data is balanced (pass rate close to 50%)
    pass_rate = passed.mean()
    print(f"\nb) Pass rate: {100*pass_rate:.1f}% - {'Balanced' if 0.4 <= pass_rate <= 0.6 else 'Imbalanced'}")
    
    # c) Determine which variable increases pass probability more
    beta_hdi = az.hdi(idata, var_names=["beta_study", "beta_sleep"], hdi_prob=0.95)
    print(f"\nc) beta1: {beta_study_mean:.4f} [{beta_hdi['beta_study'].values[0]:.4f}, {beta_hdi['beta_study'].values[1]:.4f}]")
    print(f"   beta2: {beta_sleep_mean:.4f} [{beta_hdi['beta_sleep'].values[0]:.4f}, {beta_hdi['beta_sleep'].values[1]:.4f}]")
    
    if abs(beta_study_mean) > abs(beta_sleep_mean):
        print(f"   → Study hours has stronger effect")
    else:
        print(f"   → Sleep hours has stronger effect")
