import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import raises
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_raise_message
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import compute_class_weight
from sklearn.utils.fixes import sp_version

from sklearn.linear_model.poisson import (
    _poisson_loss,
    _poisson_grad,
    PoissonRegression
)
#from sklearn.model_selection import StratifiedKFold
#from sklearn.datasets import load_iris, make_classification
#from sklearn.metrics import log_loss

#X = [[-1, 0], [0, 1], [1, 1]]
#X_sp = sp.csr_matrix(X)
#Y1 = [0, 1, 1]
#Y2 = [2, 1, 0]
#iris = load_iris()


def test_poisson_loss():
    w = np.array([1., -1.])
    X = np.array([[10, 5.4], [20, 13.1], [12, 4]])
    y = np.array([100, 1000, 3000])
    exposure = None
    alpha = 0.0
    result1 = _poisson_loss(w, X, y, exposure, alpha)
    result2 = _poisson_grad(w, X, y, exposure, alpha)

    assert False, (result1, result2)


def test_poisson_regression():
    model = PoissonRegression(tol=1e-10)
    X = np.array([[10.00517, 5.4], [20.077, 13.1], [12.00637, 4]])
    y = np.array([100, 1000, 3000])

    #import statsmodels
    from statsmodels.discrete.discrete_model import Poisson
    
    m = Poisson(y, X)
    res = m.fit(method='lbfgs')
    print res.summary()

    model.fit(X, y, )
    assert False, model.coef_

