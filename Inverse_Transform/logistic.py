import numpy as np
import matplotlib.pyplot as plt

def logistic_sampler(mu: float, s: float, size: int = 1000, rng = None) -> np.ndarray:
    """
    Samples from logistic distribution using Inverse Transform Method

    parameters
    ----------
    mu : float
        location parameters. determines the mean of the distribution
    s : float
        scale parameters. determines the spread of the distribution
    size : int
        number of the samples
    rng : numpy object
        random number generator

    returns
    -------
    np.ndarray
        inverse of the logistic distribution CDF
    """
    if rng is None:
        rng = np.random.default_rng()
    U = rng.uniform(0, 1, size)
    return mu + s * np.log(U / (1 - U))


def plot_logistic_dist(samples: np.ndarray, mu: float, s: float):
    """
    Plot the logistic distribution using samples.

    parameters
    ----------
    samples : np.ndarray
        samples of the logistic distribution
    mu : float
        location parameters. determines the mean of the distribution
    s : float
        scale parameters. determines the spread of the distribution
    """
    plt.hist(samples, bins = 500, density = True, color = "blue", alpha = 0.5, range = (-10, 10))
    x = np.linspace(-10, 10, 1000)
    logistic_exp = np.exp(-(x - mu) / s)
    logistic_distribution = logistic_exp / (s * ((1 + logistic_exp) ** 2))
    plt.plot(x, logistic_distribution, color = "red", lw = 2, label = "logistic distribution")
    plt.legend()
    plt.show()


mu = 0.0
s = 1.0
samples = logistic_sampler(mu, s, 5000, np.random.default_rng(10))
plot_logistic_dist(samples, mu, s)