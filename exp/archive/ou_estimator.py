from typing import List, Tuple, Optional
import logging
import numpy as np
from nptyping import NDArray
import scipy.optimize as opt

class OrnsteinUhlenbeckEstimator:
    """
    Convenience class to evaluate the Ornstein-Uhlenbeck parameters of multiple
    sets of series.
    """

    def __init__(self, data, **kwargs):
        """
        Initialize the class.
        data: List of tuples of (t, x) arrays
        kwargs:
          n_it: number of iterations to take (default 1)
          init_mu: starting parameter for mu (default is mean of data)
          init_eta: starting parameter for eta (default is a very rough guess)
        """
        self.data = data
        
        n_iter = kwargs.pop("n_it", 1)
        self.n_iter = n_iter
        if n_iter <= 0:
            logging.warning(
                "Parameter estimates will not be accurate without at least one iteration."
            )
        # for t, x in data:
        #     if t.size != x.shape[0]:
        #         raise RuntimeError("Time and signal data must have the same length.")
        #     if t.size < 1:
        #         raise RuntimeError("Not enough data points in a set.")
        
        # The effective number is one less than the length because it is the
        # adjacent differences that are used.
        self.ns = [ts.size - 1 for ts, _ in data]
        self.mu = kwargs.pop(
            "init_mu", sum([np.sum(x) for (_, x) in data]) / sum(self.ns)
        )
        self.eta = kwargs.pop(
            "init_eta", np.average([eta_start(*tx) for tx in data], weights=self.ns),
        )
        self.variance = np.average(
            [variance(t, x, self.eta, self.mu) for t, x in self.data], weights=self.ns
        )
        if kwargs:
            logging.error("Unrecognized keyword arguments: {kwargs.keys()}")
            raise RuntimeError("Unrecognized keyword arguments")
    


    def iterate(self, data) -> None:
        """
        Do an iteration on eta then mu, using MLE for eta (at constant mu) then
        updating mu exactly.
        """
        n = self.n_iter
        
        while n > 0:
            def func(eta: float) -> float:
                return -2.0 * np.sum([likelihood(t, x, eta, self.mu) for t, x in data])

            def grad(eta: float) -> float:
                return np.array(
                    np.sum(
                        [-2.0 * deta_likelihood(t, x, eta, self.mu) for t, x in data],
                    )
                )

            # The bounds should be configurable; in principle eta can be > 1.0
            fit = opt.minimize(func, self.eta, jac=grad, bounds=[(0.0, 1.0)])
            if not fit.success:
                print("Error: fit was not successful.")
                raise RuntimeError(fit.message)
            self.eta = fit.x
            # eta_err = np.sqrt(2.0 * fit.hess_inv * np.eye(1))
            # print(f"eta err = {eta_err}")
            
            self.mu = mu_list(data, self.eta)

            n -= 1
        # The *average* variance (not the point-by-point expected variance)
        self.variance = np.average(
            [variance(t, x, self.eta, self.mu) for t, x in data], weights=self.ns
        )
        
        
        
        
    def sigma_sq(self) -> float:
        """
        Return the sigma-squared parameter (the variance per small change in
        time) given the current estimates.
        """
        return 2.0 * self.eta * self.variance

    def deviations(
        self, data=None
    ):
        """
        Returns the weighted deviations at each point.
        """
        the_data = data or self.data
        return np.concatenate(
            [deviations(t, x, self.eta, self.mu) for t, x in the_data]
        )


def eta_start(t, x) -> float:
    """
    Returns a guesstimate of eta (totally unverified). This probably only holds
    for equally-spaced steps. But it provides a reasonable start value around an
    order of magnitude from the real one.
    """
    dt = t[1:] - t[:-1]
    dx = x[1:] - x[:-1]
    xdx = (x[:-1] * dx).sum()
    x2dt = (np.square(x[:-1]) * dt).sum()
    eta = -xdx / (x2dt+1e-6) + 1e-6
    assert eta > 0.0, "eta must be positive"
    return eta


def likelihood(
    t, x, eta: float, mu: float = 0.0
) -> float:
    """
    Returns the likelihood where sigma^2 has been replace by its MLE estimator
    as a function of eta and mu. It diverges if eta == 0.
    - (n/2) * (1 + log D) - (1/2) * sum log [1 - exp(-2*eta*dt)]
    """
    dt = t[1:] - t[:-1]
    nm1 = dt.size
    dev = variance(t, x, eta, mu)

    return (
        -0.5 * nm1 * (1.0 + np.log(2.0 * np.pi * dev))
        - 0.5 * np.log(-np.expm1(-2.0 * eta * dt)).sum()
    )


def deta_likelihood(
    t, x, eta: float, mu: float = 0.0
) -> float:
    """
    The gradient of the total likelihood with respect to eta.
    """
    dt = t[1:] - t[:-1]
    nm1 = dt.size
    dev = variance(t, x, eta, mu)
    deta_dev = deta_variance(t, x, eta, mu)
    exp_2etadt = np.exp(-2 * eta * dt)
    expm1_2etadt = np.expm1(-2 * eta * dt)
    return -0.5 * nm1 * deta_dev / dev + np.sum(dt * exp_2etadt / expm1_2etadt)


def opt_eta(t, x, mu: float = 0.0) -> float:
    """
    Return the eta parameter estimated by maximum likelihood.
    """
    eta_init = np.array([eta_start(t, x)])

    def func(eta: float) -> float:
        return -2.0 * likelihood(t, x, eta, mu)

    def grad(eta: float) -> float:
        return np.array([-2.0 * deta_likelihood(t, x, eta, mu)])

    fit = opt.minimize(func, eta_init, jac=grad, bounds=[(0.0, 1.0)])
    if not fit.success:
        print("Error: fit was not successful. {fit.message}")
    # eta_err = np.sqrt(2.0 * fit.hess_inv * np.eye(1))
    # print(f"eta err = {eta_err}")
    return fit.x


def deviations(
    t, x, eta: float, mu: float = 0.0
):
    """
    Returns the appropriately-weighted unsigned deviations i.e. the difference
    from predicted divided by the standard deviation.
    """
    # the time (or position) steps
    dt = t[1:] - t[:-1]
    # the next and previous values offset by the mean
    xn = x[1:] - mu
    xp = x[:-1] - mu

    # right now returning the signed version
    return xn - xp * np.exp(-eta * dt) / np.sqrt(-np.expm1(-2.0 * eta * dt))


def variance(
    t, x, eta: float, mu: float = 0.0
) -> float:
    """
    Returns an analogue of the weighted average of squares.
    D = (1/(n-1)) * sum { [(x_t - mu) - (x_{t-1} - mu)*exp(-eta*dt)] / (1 - exp(-2*eta*dt)) }
    This is the variance of the stable distribution.
    """
    # the time (or position) steps
    dt = t[1:] - t[:-1]
    # the next and previous values offset by the mean
    xn = x[1:] - mu
    xp = x[:-1] - mu

    return np.mean(
        np.square(xn - xp * np.exp(-eta * dt)) / (-np.expm1(-2.0 * eta * dt))
    )


def deta_variance(
    t, x, eta: float, mu: float = 0.0
) -> float:
    """
    Returns the derivative of the `variance` function with respect to eta.
    WARNING: there seem to be numerical issues at small eta, and eta will be small.
    """
    dt = t[1:] - t[:-1]
    xn = x[1:] - mu
    xp = x[:-1] - mu
    exp_etadt = np.exp(-eta * dt)
    expm1_2etadt = np.expm1(-2.0 * eta * dt)

    terms = (
        2.0
        * dt
        * exp_etadt
        * (xn - xp * exp_etadt)
        * (xp - xn * exp_etadt)
        / np.square(expm1_2etadt)
    )
    return terms.mean()


def mu(t, x, eta: float) -> float:
    """
    Returns the appropriately weighted mu given eta.
    """
    num, den = _mu_one(t, x, eta)
    return num / den


def mu_list(data, eta) -> float:
    """
    Returns the mu for a collection of samplings, properly weighting all together.
    """
    nums, dens = zip(*[_mu_one(t, x, eta) for t, x in data])
    return np.sum(nums) / np.sum(dens) + 1


def _mu_one(t, x, eta: float) -> float:
    """
    Returns the numerator and denominator of an appropriately weighted mu given
    eta for a single t, x set
    """
    dt = t[1:] - t[:-1]
    xn, xp = x[1:], x[:-1]
    exp_etadt = np.exp(-eta * dt)
    expm1_etadt = np.expm1(-eta * dt)

    num = (xn - xp * exp_etadt) / (1.0 + exp_etadt)
    den = -expm1_etadt / (1.0 + exp_etadt)
    return num.sum(), den.sum()
