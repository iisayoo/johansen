import numpy as np
from statsmodels.tsa.tsatools import lagmat

import critical_values


class Johansen(object):
    """Implementation of the Johansen test for cointegration.

    References:
        - Hamilton, J. D. (1994) 'Time Series Analysis', Princeton Univ. Press.
        - MacKinnon, Haug, Michelis (1996) 'Numerical distribution functions of
        likelihood ratio tests for cointegration', Queen's University Institute
        for Economic Research Discussion paper.

    """

    def __init__(self, x, model, k=1, trace=True,  significance_level=1):
        """
        :param x: (nobs, m) array of time series. nobs is the number of
        observations, or time stamps, and m is the number of series.

        :param k: The number of lags to use when regressing on the first
        difference of x.

        :param trace: Whether to use the trace or max eigenvalue statistic for
        the hypothesis testing. If False the latter is used.

        :param model: Which of the five cases in Osterwald-Lenum 1992 (or
        MacKinnon 1996) to use.
            - If set to 0, case 0 will be used. This case should be used if
            the input time series have no deterministic terms and all the
            cointegrating relations are expected to have 0 mean.
            - If set to 1, case 1* will be used. This case should be used if
            the input time series has neither a quadratic nor linear trend,
            but may have a constant term, and additionally if the cointegrating
            relations may have nonzero means.
            - If set to 2, case 1 will be used. This case should be used if
            the input time series have linear trends but the cointegrating
            relations are not expected to have linear trends.
            - If set to 3, case 2* will be used. This case should be used if
            the input time series do not have quadratic trends, but they and
            the cointegrating relations may have linear trends.
            - If set to 4, case 2 will be used. This case should be used if
            the input time series have quadratic trends, but the cointegrating
            relations are expected to only have linear trends.

        :param significance_level: Which significance level to use. If set to
        0, 90% significance will be used. If set to 1, 95% will be used. If set
        to 2, 99% will be used.

        """

        self.x = x
        self.k = k
        self.trace = trace
        self.model = model
        self.significance_level = significance_level

        if trace:
            key = "TRACE_{}".format(model)
        else:
            key = "MAX_EVAL_{}".format(model)

        critical_values_str = critical_values.mapping[key]

        select_critical_values = np.array(
            critical_values_str.split(),
            float).reshape(-1, 3)

        self.critical_values = select_critical_values[:, significance_level]

    def mle(self):
        """Obtain the cointegrating vectors and corresponding eigenvalues.

        Maximum likelihood estimation and reduced rank regression are used to
        obtain the cointegrating vectors and corresponding eigenvalues, as
        outlined in Hamilton 1994.

        :return: The possible cointegrating vectors, i.e. the eigenvectors
        resulting from maximum likelihood estimation and reduced rank
        regression, and the corresponding eigenvalues.

        """

        # Regressions on diffs and levels of x. Get regression residuals.

        # First differences of x.
        x_diff = np.diff(self.x, axis=0)

        # Lags of x_diff.
        x_diff_lags = lagmat(x_diff, self.k, trim='both')

        # First lag of x.
        x_lag = lagmat(self.x, 1, trim='both')

        # Trim x_diff and x_lag so they line up with x_diff_lags.
        x_diff = x_diff[self.k:]
        x_lag = x_lag[self.k:]

        # Include intercept in the regressions if self.model != 0.
        if self.model != 0:
            ones = np.ones((x_diff_lags.shape[0], 1))
            x_diff_lags = np.append(x_diff_lags, ones, axis=1)

        # Include time trend in the regression if self.model = 3 or 4.
        if self.model in (3, 4):
            times = np.asarray(range(x_diff_lags.shape[0])).reshape((-1, 1))
            x_diff_lags = np.append(x_diff_lags, times, axis=1)

        # Residuals of the regressions of x_diff and x_lag on x_diff_lags.
        try:
            inverse = np.linalg.pinv(x_diff_lags)
        except:
            print("Unable to take inverse of x_diff_lags.")
            return None

        u = x_diff - np.dot(x_diff_lags, np.dot(inverse, x_diff))
        v = x_lag - np.dot(x_diff_lags, np.dot(inverse, x_lag))

        # Covariance matrices of the residuals.
        t = x_diff_lags.shape[0]
        Svv = np.dot(v.T, v) / t
        Suu = np.dot(u.T, u) / t
        Suv = np.dot(u.T, v) / t
        Svu = Suv.T

        try:
            Svv_inv = np.linalg.inv(Svv)
        except:
            print("Unable to take inverse of Svv.")
            return None
        try:
            Suu_inv = np.linalg.inv(Suu)
        except:
            print("Unable to take inverse of Suu.")
            return None

        # Eigenvalues and eigenvectors of the product of covariances.
        cov_prod = np.dot(Svv_inv, np.dot(Svu, np.dot(Suu_inv, Suv)))
        eigenvalues, eigenvectors = np.linalg.eig(cov_prod)

        # Normalize the eigenvectors using Cholesky decomposition.
        evec_Svv_evec = np.dot(eigenvectors.T, np.dot(Svv, eigenvectors))
        cholesky_factor = np.linalg.cholesky(evec_Svv_evec)
        try:
            eigenvectors = np.dot(eigenvectors,
                                  np.linalg.inv(cholesky_factor.T))
        except:
            print("Unable to take the inverse of the Cholesky factor.")
            return None

        # Ordering the eigenvalues and eigenvectors from largest to smallest.
        indices_ordered = np.argsort(eigenvalues)
        indices_ordered = np.flipud(indices_ordered)
        eigenvalues = eigenvalues[indices_ordered]
        eigenvectors = eigenvectors[:, indices_ordered]

        return eigenvectors, eigenvalues

    def h_test(self, eigenvalues, r):
        """Carry out hypothesis test.

        The null hypothesis is that there are at most r cointegrating vectors.
        The alternative hypothesis is that there are at most m cointegrating
        vectors, where m is the number of input time series.

        :param eigenvalues: The list of eigenvalues returned from the mle
        function.

        :param r: The number of cointegrating vectors to use in the null
        hypothesis.

        :return: True if the null hypothesis is rejected, False otherwise.

        """

        nobs, m = self.x.shape
        t = nobs - self.k - 1

        if self.trace:
            m = len(eigenvalues)
            statistic = -t * np.sum(np.log(np.ones(m) - eigenvalues)[r:])
        else:
            statistic = -t * np.sum(np.log(1 - eigenvalues[r]))

        critical_value = self.critical_values[m - r - 1]

        if statistic > critical_value:
            return True
        else:
            return False

    def johansen(self):
        """Obtain the possible cointegrating relations and numbers of them.

        See the documentation for methods mle and h_test.

        :return: The possible cointegrating relations, i.e. the eigenvectors
        obtained from maximum likelihood estimation, and the numbers of
        cointegrating relations for which the null hypothesis is rejected.

        """

        nobs, m = self.x.shape

        try:
            eigenvectors, eigenvalues = self.mle()
        except:
            print("Unable to obtain possible cointegrating relations.")
            return None

        rejected_r_values = []
        for r in range(m):
            if self.h_test(eigenvalues, r):
                rejected_r_values.append(r)

        return eigenvectors, rejected_r_values
