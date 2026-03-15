## Inverse Transform Sampling
This folder contains python implementation of inverse transform sampling for common distributions. It is used to generate random samples from the target distribution by applying the inverse of the Cumulative Distribution Function or CDF to samples from uniform distribution.
## Implemented Distributions
- cauchy
- exponential
- laplace
- logistic
- uniform
## How it Works
Inverse Transform sampling works in two steps:
1. Draw random samples with values between 0 and 1 from a uniform distribution (U).
2. Apply the inverse CDF of the target distribution to the values U.