"""
Poisson Regression
"""

# Author: Brian Keng <brian.keng@gmail.com>

import numbers
import warnings

import numpy as np
from scipy import optimize, sparse

from .base import LinearModel, RegressorMixin
#from .sag import sag_solver
#from ..feature_selection.from_model import _LearntSelectorMixin
#from ..preprocessing import LabelEncoder, LabelBinarizer
#from ..svm.base import _fit_liblinear
#from ..utils import check_array, check_consistent_length, compute_class_weight
from ..utils import check_array, check_consistent_length
#from ..utils import check_random_state
#from ..utils.extmath import (logsumexp, log_logistic, safe_sparse_dot,
#                             softmax, squared_norm)
#from ..utils.extmath import row_norms
#from ..utils.optimize import newton_cg
from ..utils.validation import check_X_y
#from ..exceptions import DataConversionWarning
#from ..exceptions import NotFittedError
#from ..utils.fixes import expit
#from ..externals.joblib import Parallel, delayed
#from ..model_selection import check_cv
#from ..externals import six
#from ..metrics import SCORERS

def _poisson_loss(w, X, y, exposure, alpha):
    """Computes the Poisson loss.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of outcome variables.

    exposure : ndarray, shape (n_samples,)
        Array of exposure variables.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    Returns
    -------
    out : float
        Poisson loss.
    """
    # Poisson loss:
    # -\Sum_{i=1}^{n} y_i * w * x_i - exp(w * x_i + ln(exposure))
    Xw = np.dot(X, w)

    if exposure is None:
        exposure = np.zeros(len(y))
    z = np.exp(Xw + exposure)

    out = -np.sum(y * Xw - z) + .5 * alpha * np.dot(w, w)
    return out


def _poisson_grad(w, X, y, exposure, alpha):
    """Computes the Poisson loss.

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of outcome variables.

    exposure : ndarray, shape (n_samples,)
        Array of exposure variables.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    Returns
    -------
    out : float
        Poisson loss.
    """
    n_samples, n_features = X.shape

    # Grad of Poisson loss:
    # -\Sum_{i=1}^{n} (y_i - exp(w*x_i + ln(exposure)))*x_i
    Xw = np.dot(X, w)

    if exposure is None:
        exposure = np.zeros(len(y))
    z = np.exp(Xw + exposure)

    grad = -np.dot(y - z, X)
    return grad


def poisson_regression(X, y, exposure, alpha, solver, max_iter, 
                       tol, verbose, fit_intercept):
    # SAG needs X and y columns to be C-contiguous and np.float64
    X = check_array(X, dtype=np.float64)
    y = check_array(y, dtype='numeric', ensure_2d=False)
    check_consistent_length(X, y)

    n_samples, n_features = X.shape

    if y.ndim > 2:
        raise ValueError("Target y has the wrong shape %s" % str(y.shape))

    n_samples_y = len(y)

    if n_samples != n_samples_y:
        raise ValueError("Number of samples in X and y does not correspond:"
                         " %d != %d" % (n_samples, n_samples_y))

    if exposure is not None:
        if exposure.ndim > 2:
            raise ValueError("Target exposure has the wrong shape %s" % str(exposure.shape))

        n_samples_exposure = len(exposure)

        if n_samples != n_samples_exposure:
            raise ValueError("Number of samples in X and exposure do not correspond:"
                             " %d != %d" % (n_samples, n_samples_exposure))

        exposure = np.log(exposure)

    w0 = np.zeros(n_features + int(fit_intercept))

    print "================================ cg"
    result = optimize.fmin_cg(
        _poisson_loss, w0,
        args=(X, y, exposure, alpha),
        full_output=(verbose > 0) - 1, gtol=tol, maxiter=max_iter)

    for r in result:
        print r

    print "================================ l_bfgs"
    result = optimize.fmin_l_bfgs_b(
        _poisson_loss, w0, _poisson_grad,
        args=(X, y, exposure, alpha),
        iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)

    for r in result:
        print r

    print "================================ powell"
    result = optimize.fmin_powell(
        _poisson_loss, w0, args=(X, y, exposure, alpha),
        full_output=(verbose > 0) - 1, ftol=tol, maxiter=max_iter)

    for r in result:
        print r

    return w0


class PoissonRegression(LinearModel, RegressorMixin):
    """Poisson Regression.

        alpha : float
            Small positive values of alpha improve the conditioning of the problem
            and reduce the variance of the estimates.  Alpha corresponds to
            ``C^-1`` in other linear models such as LogisticRegression or
            LinearSVC.

        solver : {'lbfgs', 'powell'}
            Solver to use in the computational routines

        tol : float
            Precision of the solution.

        max_iter : int, optional
            Maximum number of iterations for conjugate gradient solver.
            The default value is determined by scipy.sparse.linalg.
 """

    def __init__(self, tol=1e-4, alpha=0., fit_intercept=False,
                 solver='sag', max_iter=1000, verbose=0):
        self.tol = tol
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y, exposure=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : returns an instance of self.
        """
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)

        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                         order="C")

        n_samples, n_features = X.shape

        coeff = poisson_regression(X, y, exposure, alpha=self.alpha,
                    solver=self.solver, max_iter=self.max_iter, tol=self.tol,
                    verbose=self.verbose, fit_intercept=self.fit_intercept)

        self.coef_ = coeff

        #if self.fit_intercept:
        #    self.intercept_ = self.coef_[:, -1]
        #    self.coef_ = self.coef_[:, :-1]

        return self

    def predict(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")

        assert False

