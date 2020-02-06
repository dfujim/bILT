#!/usr/bin/python3

# Inverse Laplace Transform (ILT) of β-NMR spin-lattice relaxation (SLR) data

# https://stackoverflow.com/a/36112536
# https://stackoverflow.com/a/35423745


# https://gist.github.com/diogojc/1519756

import numpy as np
import matplotlib.pyplot as plt
import bdata as bd
from bfit.fitting import functions
from scipy.optimize import nnls, curve_fit, root_scalar
from scipy.linalg import norm

# ILT function
def ilt(t, y, y_error, T1, alpha, maxiter=None):
    """
        Get ILT for a kernel matrix defined by a set of exponentials, using a
        non-negative least squares solver for the constrained ridge resgression
        problem.
        
        t:  array of time steps in data to fit
        y:  array of data points f(t) needing to fit
        T1: logspaced array of T1 values corresponding to the probabilities in 
            the output (ex: np.logspace(-5,5,500))
        alpha: Tikhonov regularization parameter
        
        output: (p,f)
        
            p:  probability density function corresponding to the weights of 
                T1 in the solution. Fit funtion is 
            f:  array corresponding to fit function
    """
    # pulsed exponential
    pexp = functions.pulsed_exp(lifetime=1.2096, pulse_len=4.0)

    sigma = 1 / y_error
    Sigma = np.diag(sigma)

    # make kernal matrix
    # K = np.array([np.exp(-i / T1) for i in t])
    K = np.array([pexp(time=t, lambda_s=1 / i, amp=1) for i in T1])
    K = K.T

    # prep input matrices for nnls - no weights
    # Q = np.dot(K.T, K) + alpha * np.eye(K.shape[1])
    # c = np.dot(K.T, y)

    # first crack at adding weights
    # https://stackoverflow.com/a/36112536
    # Q = np.matmul(K.T, np.matmul(Sigma, K)) + alpha * np.eye(K.shape[1])
    # c = np.dot(K.T, y * sigma)
    # c = np.matmul(K.T, np.matmul(Sigma, y))

    # adding regularization
    # https://stackoverflow.com/a/35423745
    Q = np.matmul(K.T, np.matmul(Sigma, K))
    Q_reg = np.concatenate([Q, np.sqrt(alpha) * np.eye(K.shape[1])])

    c = np.matmul(K.T, np.matmul(Sigma, y))
    c_reg = np.concatenate((c, np.zeros(K.shape[1])))

    # solve
    if maxiter is None:
        P, r = nnls(Q_reg, c_reg)
    else:
        P, r = nnls(Q_reg, c_reg, maxiter)

    # define functional output
    f = np.dot(K, P)

    # normalize <- don't do this here - leave it for after the fact.
    # P /= norm(P)

    # returns chi (not chi^2)
    # chi = norm((y - np.dot(K, P)) * sigma)
    chi = norm(np.matmul(Sigma, (np.matmul(K, P) - y)))

    # calculate the generalize cross-validation (GCV) parameter tau
    # tau = np.trace(np.eye(K.shape[1]) - K ( np.matmul(K.T, K) + alpha * alpha * np.matmul(L.T, L) ) K.T )

    return (P, f, chi)


# multiexponential relaxation convenience function
def multiexp(times, fractions, T1s):
    # numpyfy input
    times = np.asarray(times)
    fractions = np.asarray(fractions)
    T1s = np.asarray(T1s)

    # normalize input
    # fractions = fractions / np.sum(fractions)

    # calculation

    # first one
    """
    y = fractions[0] * np.exp(-times / T1s[0])
    # every other one
    for f, T in zip(fractions[1:], T1s[1:]):
        y += f * np.exp(-times / T)
    """
    pexp = functions.pulsed_exp(lifetime=1.2096, pulse_len=4.0)

    y = pexp(time=times, lambda_s=1 / T1s[0], amp=fractions[0])
    for f, T in zip(fractions[1:], T1s[1:]):
        y += pexp(time=times, lambda_s=1 / T, amp=f)
    return y


# read the run and

# Pt foil
run = bd.bdata(40214, year=2009)

# PEO
# run = bd.bdata(40374, year=2013)

# rutile
# run = bd.bdata(40343, year=2014)

# bts
# run = bd.bdata(45886, year=2011)

# LAO
# run = bd.bdata(41109, year=2015)

# LNO/LAO 2||2
# run = bd.bdata(40978, year=2015)

# Ag nanoparticles
# run = bd.bdata(41944, year=2014)


# find the beam on time/bins
beam_on_bin = int(run.ppg.beam_on.mean)
beam_on_time = run.ppg.dwelltime.mean * 0.001 * beam_on_bin

# calculate the combined asymmetry
asy = run.asym(option="c", omit="", rebin=1, hist_select="", nbm=False)

# check the correct bins are selected
# print(asy[0][beam_on_bin:] - beam_on_time)

# sample points when calculating p(T1)
T1 = np.logspace(np.log(0.01 * 1.2096), np.log(100.0 * 1.2096), 1000)


# find the optimum alpha

# smoothing parameters to check
alpha = np.logspace(2, 8, 50)

# norm of residuals and result
chi = np.zeros(alpha.size)
p_norm = np.zeros(alpha.size)

# how does the residual change with alpha?
for i, a in enumerate(alpha):
    p, f, c = ilt(
        # asy[0][beam_on_bin:] - beam_on_time, asy[1][beam_on_bin:], T1, a, maxiter=10000
        asy[0],
        asy[1],
        asy[2],
        T1,
        a,
        maxiter=10000,
    )
    chi[i] = c
    p_norm[i] = norm(p)


ln_alpha = np.log(alpha)
ln_chi = np.log(chi)

# https://stackoverflow.com/a/19459160
dlnchi_dlnalpha = np.gradient(ln_alpha) / np.gradient(ln_chi)


# pick an (arbitrary) alpha as the optimum
# estimate from the chi(alpha) minumum
chi_opt = np.min(chi)
alpha_opt = alpha[np.argmin(chi)]
p_norm_opt = p_norm[np.argmin(chi)]
# alpha_opt = 6e7


# g_ln_chi = np.gradient(np.log(chi))
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, sharey=False, constrained_layout=True,
)
ax1.plot(
    alpha, np.array(chi) ** 2, "o", zorder=1, label=r"$\chi ( \alpha )^{2}$",
)

ax1.plot(
    alpha_opt,
    chi_opt ** 2,
    "o",
    zorder=2,
    label=r"$\chi ( \alpha_{\mathrm{opt}} )^{2}$",
)


# ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$\chi ( \alpha )^{2}$")
ax1.set_yscale("log")


ax2.plot(
    alpha,
    dlnchi_dlnalpha,
    ".-",
    label=r"$\mathrm{d} \ln \chi ( \alpha ) / \mathrm{d} \ln \alpha$",
)
ax2.set_xlabel(r"$\alpha$")
ax2.set_ylabel(r"$\mathrm{d} \ln \chi ( \alpha ) / \mathrm{d} \ln \alpha$")
ax2.axhline(
    0.1,
    linestyle="--",
    color="black",
    zorder=0,
    label=r"$\mathrm{d} \ln \chi ( \alpha ) / \mathrm{d} \ln \alpha = 0.1$",
)
# ax2.set_title(r"$\chi ( \alpha )$ vs. $\alpha$")
ax2.legend()

ax2.set_xscale("log")
# print(asy["time_s"])
# print(asy["c"])


# plot the L-curve
figp, axp = plt.subplots(1, 1, constrained_layout=True)

axp.plot(
    chi, p_norm, "o-", zorder=1,
)
# for x, y, a in zip(chi, p_norm, alpha):
#    axp.text(x, y, r"$\alpha = %.2e$" % a)


axp.plot(
    chi_opt, p_norm_opt, "o", zorder=2,
)
axp.text(
    0.95,
    0.95,
    r"$\chi^{2} = %.1f$" % chi_opt ** 2
    + "\n"
    + "$n_{\mathrm{data}} = %d$" % asy[0].size
    + "\n"
    + r"$\alpha_{\mathrm{opt}} = %.3e$" % alpha_opt,
    ha="right",
    va="top",
    transform=axp.transAxes,
)

axp.set_xlabel("$|| \Sigma ( K \mathbf{P} - \mathbf{y} ) ||$")
axp.set_ylabel("$|| \mathbf{P} ||$")
axp.set_title("L-curve")

axp.set_xscale("log")
axp.set_yscale("log")


# ax1.axvline(alpha_opt, linestyle="--", color="black", zorder=0)
# ax2.axvline(alpha_opt, linestyle="--", color="black", zorder=0)


# do the ILT using the "optimal" alpha
"""
p, f, c = ilt(
    asy[0][beam_on_bin:] - beam_on_time,
    asy[1][beam_on_bin:],
    T1,
    alpha_opt,
    maxiter=10000,
)
"""
p, f, c = ilt(asy[0], asy[1], asy[2], T1, alpha_opt, maxiter=10000,)

# normalize the amplitudes to give probabilities
p_norm = p / np.sum(p)

fig2, ax2 = plt.subplots(1, 1, constrained_layout=True,)

# fig2.suptitle(r"$\alpha = %.3f$" % alpha_opt)

"""
ax21.errorbar(
    asy[0][beam_on_bin:] - beam_on_time,
    asy[1][beam_on_bin:],
    yerr=asy[2][beam_on_bin:],
    fmt="o",
    zorder=1,
    label="Year: %s\nRun: %s" % (run.year, run.run),
)

ax21.plot(
    asy[0][beam_on_bin:] - beam_on_time,
    multiexp(asy[0][beam_on_bin:] - beam_on_time, p, T1),
    "-",
    zorder=2,
    label="ILT",
)
"""

ax2.errorbar(
    asy[0],
    asy[1],
    yerr=asy[2],
    fmt="o",
    zorder=1,
    label="Year: %s\nRun: %s" % (run.year, run.run),
)

ax2.plot(
    asy[0], multiexp(asy[0], p, T1), "-", zorder=2, label="ILT",
)

ax2.set_xlabel("$t$ (s)")
ax2.set_ylabel("$A(t)$")

ax2.legend()


fig3, ax3 = plt.subplots(1, 1, constrained_layout=True,)

ax3.plot(
    T1, p_norm, "-", zorder=1, label="ILT",
)

# mean T1
T1_mean = np.sum(p * T1) / np.sum(p)
beta = 1 / np.sqrt(np.sum(p * (T1_mean - T1) ** 2)) / np.sum(p)

ax3.axvline(T1_mean, linestyle="--", color="black", zorder=0)

print("T1_mean = %.3f s" % T1_mean)
print("   beta = %.3f" % beta)

ax3.set_xlabel("$T_{1}$ (s)")
ax3.set_ylabel("$p(T_{1})$")

ax3.set_xscale("log")

ax3.legend()

plt.show()
