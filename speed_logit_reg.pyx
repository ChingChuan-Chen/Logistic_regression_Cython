import numpy as np
cimport numpy as np
from libc.math cimport exp, log
# from libcpp.limits cimport numeric_limits
import cython

# cdef double double_eps = numeric_limits[double].epsilon()
cdef double MTHRESH = -30.0
cdef double THRESH = 30.0

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline np.ndarray logit_linkinv(np.ndarray eta):
    cdef np.ndarray etaClamped = np.clip(eta, MTHRESH, THRESH)
    return 1.0 / (1.0 + np.exp(-etaClamped))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def LogitLinkInv(np.ndarray[np.float64_t, ndim=2] x, np.ndarray[np.float64_t] beta):
    cdef np.ndarray eta = np.matmul(x, beta)
    return logit_linkinv(eta)

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline np.ndarray y_log_y(np.ndarray y, np.ndarray mu):
    cdef np.ndarray tmp = np.divide(y, mu, where=(mu != 0.0), out=np.zeros_like(mu))
    return y * np.log(tmp, where=(tmp != 0.0), out=np.zeros_like(tmp))

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline np.ndarray binomial_dev_resids(np.ndarray y, np.ndarray mu, np.ndarray weights):
    cdef np.ndarray y_log_y1 = y_log_y(y, mu)
    cdef np.ndarray y_log_y2 = y_log_y(1.0 - y, 1.0 - mu)
    return 2 * weights * (y_log_y1 + y_log_y2)

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline np.ndarray logit_mu_eta(np.ndarray eta):
    cdef np.ndarray etaClampedExp = np.exp(np.clip(eta, MTHRESH, THRESH))
    return np.divide(etaClampedExp, (1 + etaClampedExp)**2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def speedLogisticRegression(
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] weights,
    np.ndarray[np.float64_t] init_beta,
    double acc=1e-4, int max_iter=25
):
    cdef int n = x.shape[0], p = x.shape[1]
    cdef np.ndarray start = np.copy(init_beta)
    cdef np.ndarray eta = np.matmul(x, start)
    cdef np.ndarray mu = logit_linkinv(eta)

    cdef int iter = 0
    cdef double tol = 1.0, dev = np.sum(binomial_dev_resids(y, mu, weights)), dev0 = 0.0
    cdef np.ndarray beta, varmu, mu_eta_val, z, W
    cdef np.ndarray xw, XTX, XTz

    while tol > acc and iter < max_iter:
        iter += 1
        beta = start
        dev0 = dev
        xw = np.copy(x)
        varmu = mu * (1.0 - mu)
        mu_eta_val = logit_mu_eta(eta)

        z = eta + np.divide((y - mu), mu_eta_val)
        W = np.divide(weights * mu_eta_val * mu_eta_val, varmu)

        for i in range(p):
            xw[:, i] = x[:, i] * W

        XTWX = np.matmul(x.T, xw)
        XTWz = np.matmul(xw.T, z)

        Q, R = np.linalg.qr(XTWX, mode='reduced')
        Qb = np.matmul(Q.T, XTWz)
        start = np.linalg.solve(R, Qb)

        eta = np.matmul(x, start)
        mu = logit_linkinv(eta)
        dev = np.sum(binomial_dev_resids(y, mu, weights))
        tol = abs(dev0 - dev) / (abs(dev) + 0.1)

    return start
