import numpy as np
import matplotlib.pyplot as plt

def exponential_sampler(lamb, size = 10000, rng = None) -> np.ndarray:
    """
    Sampling from exponential distribution using inverse transform method

    parameters
    ----------
    lamb : float
        rate parameter in exponential distribution
    size : int
        number of the samples
    rng : numpy Object
        random number generator

    returns
    -------
    a numpy array of:
        inverse of exponential distribution CDF
    """
    # Create a default rng if nothing is passed to the function
    if rng is None:
        rng = np.random.default_rng()

    U = rng.uniform(0, 1, size)
    # Clip the values of U to avoid log(0) and log(1)
    U = np.clip(U, 1e-12, 1 - 1e-12)

    return -np.log(U) / lamb


def plot_exp_dist(samples, lamb):
    """
    Plot the samples of the exponential distribution with inverse transform

    parameters
    ----------
    samples : np.ndarray
        samples of the exponential distribution
    lamb : int
        rate parameter in exponential distribution
    """
    x = np.linspace(0, 5, 10000)
    plt.hist(samples, bins = 100, density = True, alpha = 0.5, label = 'exp sampled')
    plt.plot(x, np.sort(samples)[::-1], "r-", label = "exp sampled")
    plt.title("exp distribution with Inverse Transform sampling")
    plt.xlim(left = 0)
    plt.legend()
    plt.show()

lamb = 2
samples = exponential_sampler(lamb, rng = np.random.default_rng(10))
plot_exp_dist(samples, lamb)