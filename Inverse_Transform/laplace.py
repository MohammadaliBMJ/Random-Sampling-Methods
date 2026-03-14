import numpy as np
import matplotlib.pyplot as plt


def laplace_sampler(mu: float, b: float, size: int = 1000, rng = None) -> np.ndarray:
    """
    Sampling from laplace distribution using inverse transform method

    parameters
    ----------
    mu : float
        mean of the distribution and location parameter
    b : float
        scale parameter. controls the spread of the distribution
    size : int
        number of samples
    rng : numpy object
        random number generator
    """
    if rng is None:
        rng = np.random.default_rng()
    U = np.random.uniform(0, 1, size)
    return np.where(U < 0.5, mu + b * np.log(2 * U), mu - b * np.log(2*(1 - U)))


def plot_laplace_dist(samples: np.ndarray, mu: float, b: float):
    sorted_samples = np.sort(samples)
    plt.hist(sorted_samples, bins = 50,color = "blue", alpha = 0.5, label = "histogram")
    # Plot the laplace distribution
    x = np.linspace(-1, 1, 1000)
    laplace = (1 / (2 * b)) * np.exp(-(np.abs(x - mu) / b))
    plt.plot(x, laplace, label = "laplace", color = "darkred", lw = 2)
    plt.legend()
    plt.show()

mu = 0.0
b = 0.05
samples = laplace_sampler(mu, b, size = 500, rng = np.random.default_rng(10))
plot_laplace_dist(samples, mu, b)