import numpy as np
import sklearn.gaussian_process as GP
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimization:

    def __init__(self, epsilon=0.0, alpha=0.001, bounds=[], min_tries=250):

        self.epsilon = epsilon
        self.gp = GP.GaussianProcessRegressor(alpha=alpha)
        self.train_x = None
        self.train_y = None

        self.bounds = np.array(bounds)

        self.f_max = 0
        self.minimize_tries = min_tries

    def next_iter(self, X, y):
        self._add_data(X, y)
        self._fit_gp()
        self._update_f_max()
        self._update_acq()
        return self._get_best()

    def _add_data(self, X, y):
        if self.train_x is not None and self.train_y is not None:
            self.train_x = np.concatenate([self.train_x, X])
            self.train_y = np.concatenate([self.train_y, y])
        else:
            self.train_x = X
            self.train_y = y

    def _fit_gp(self):
        x = self.train_x
        if x.ndim == 1:
            x = x[:, np.newaxis]
        self.gp.fit(x, self.train_y)

    def _update_f_max(self):
        self.f_max = np.max(self.train_y)

    def _update_acq(self):

        initial_guesses = np.random.uniform(self.bounds[:, 0],
                                            self.bounds[:, 1],
                                            size=(self.minimize_tries,
                                                  self.bounds.shape[0]))
        best_max = 0
        best_x = 0
        for x0 in initial_guesses:
            res = minimize(lambda x: -self._ei(x), x0, bounds=self.bounds)
            acq_max = self._ei(res.x)
            if acq_max > best_max:
                best_max = acq_max
                best_x = res.x
        self.best_x = best_x

    def _get_best(self):
        return self.best_x

    def _ei(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        mu, sigma = self.gp.predict(x, return_std=True)
        z = (mu - self.f_max - self.epsilon) / sigma
        return (mu - self.f_max - self.epsilon) * norm.cdf(z) + sigma * norm.pdf(z)
