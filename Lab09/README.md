# Lab 9 - Exercise 1: Bayesian Inference for Store Customers

## Part b: Effect of Y and theta on the Posterior Distribution

### Effect of Y (Observed Buyers)

When Y increases (0 -> 5 -> 10), the posterior distribution for n shifts to higher values. 

Reason: More observed buyers implies more total customers must have visited the store, since we assume theta is fixed.

### Effect of theta (Purchase Probability)

When theta increases (0.2 -> 0.5), the posterior distribution for n shifts to lower values for the same Y.

Reason: A higher purchase probability means fewer customers are needed to explain the same number of buyers. For example, Y = 5 buyers could come from ~25 customers at theta = 0.2, but only ~10 customers at θ = 0.5.

## Part d: Posterior vs. Predictive Posterior

### Posterior Distribution p(n|Y, theta)

Represents our belief about how many customers n were present on the day we observed Y buyers. 

### Predictive Posterior Distribution p(Y*|Y, theta)

Represents the distribution of future buyers Y* on a new day. This is computed by: for each posterior sample of n, drawing Y* ~ Binomial(n, theta). 

### Key Difference

The posterior infers the past parameter n from observed data (more concentrated). 

The predictive posterior predicts future data Y* and includes two sources of uncertainty: uncertainty about n and binomial sampling variability. This makes it more spread out than the posterior. 