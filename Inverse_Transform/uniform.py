import numpy as np
import matplotlib.pyplot as plt


def uniform_sampler(a: float, b: float, size: int = 1000, rng = None) -> np.ndarray:
    """
    Sampling from uniform distribution using inverse transform method

    parameters
    ----------
    a : float
        minimum value of the distribution
    b : float
        maximum value of the distribution
    size : int
        number of the samples
    rng : np.ndarray
        random number generator

    returns
    -------
    np.ndarray
        inverse of the uniform distribution CDF
    """
    if rng is None:
        rng = np.random.default_rng()
    U = rng.uniform(0, 1, size)
    return a + (b - a) * U


def plot_uniform_dist(a: float, b: float, samples: np.ndarray):
    """"""
    plt.hist(samples, bins = 1000, density = True, alpha = 0.5, color = "blue", label = "histogram")
    # Theoretical Uniform Distribution PDF
    x = np.linspace(a, b, 100)
    y = np.ones_like(x) * (1 / (b - a))
    plt.plot(x, y, label = "uniform dist", color = "red", lw = 2)

    # Vertical Boundaries
    plt.vlines(x = [a, b], ymin = 0, ymax = 1 / (b - a), colors = "black", linestyles = "--", linewidth = 2)

    # Horizontal Boundaries
    plt.hlines(y = 0, xmin = a - 10, xmax = a, colors = "black", linestyles = "-", linewidth = 5)
    plt.hlines(y = 0, xmin = b, xmax = b + 10, colors = "black", linestyles = "-", linewidth = 5)
    
    plt.legend()
    plt.show()

a = -2.0
b = 5.0
samples = uniform_sampler(a, b, size = 10000, rng = np.random.default_rng(10))
plot_uniform_dist(a, b, samples)