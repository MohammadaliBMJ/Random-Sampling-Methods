import numpy as np
import matplotlib.pyplot as plt

def cauchy_sampler(x0: float, gamma: float, size: int = 1000, rng = None)\
      -> np.ndarray:
    """
    Sampling from cauchy distribution using inverse transform method

    parameters
    ----------
    x0 : float
        the mean and the median of the distribution
    gamma : float
        the width or spread of the peak
    size : int
        number of samples
    rng : numpy object
        random number generator

    returns
    -------
    a numpy array
        inverse of cauchy distribution CDF
    """
    if rng is None:
        rng = np.random.default_rng()
    U = rng.uniform(0, 1, size)
    return x0 + gamma * (np.tan(np.pi * (U - 0.5)))


def plot_cauchy_dist(samples):
    """
    Plot the samples of the cauchy distribution with inverse transform

    parameters
    ----------
    samples : np.ndarray
        samples of the cauchy distribution
    """
    sorted_samples = np.sort(samples)
    plt.hist(sorted_samples, bins = 5000, density = True, alpha = 0.5, label = "histogram")
    # Calculate the counts in each bin and the edges
    counts, bin_edge = np.histogram(sorted_samples, bins = 5000, density = True)
    # Find the centers of each bin
    centers = (bin_edge[:-1] + bin_edge[1:]) / 2
    plt.plot(centers, counts, alpha = 0.5)
    plt.xlim(-100, 100)
    plt.legend()
    plt.title("Cauchy Distribution with Inverse Transform Method")
    plt.show()

samples = cauchy_sampler(x0 = 0.0, gamma = 6.0, rng = np.random.default_rng(10))
plot_cauchy_dist(samples)

    