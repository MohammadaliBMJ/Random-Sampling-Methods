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
    a numpy array
        inverse of exponential distribution CDF
    """
    # Create a default rng if nothing is passed to the function
    if rng is None:
        rng = np.random.default_rng()

    U = rng.uniform(0, 1, size)
    # Clip the values of U to avoid log(0) and log(1)
    U = np.clip(U, 1e-12, 1 - 1e-12)

    return -np.log(U) / lamb


def plot_exp_dist(samples):
    """
    Plot the samples of the exponential distribution with inverse transform

    parameters
    ----------
    samples : np.ndarray
        samples of the exponential distribution
    """
    x = np.linspace(0, 5, 10000)
    plt.hist(samples, bins = 100, density = True, alpha = 0.5, label = 'exp sampled')
    counts, edges = np.histogram(samples, bins = 100, density = True)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(centers, counts, "r-", label = "exp sampled")
    plt.title("exp Distribution With Inverse Transform Sampling")
    plt.xlim(left = 0)
    plt.legend()
    plt.show()


samples = exponential_sampler(lamb = 2, rng = np.random.default_rng(10))
plot_exp_dist(samples)