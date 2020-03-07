import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from glob import glob
from datetime import date


filenames = glob("benchmark/*.dat")

data = [
    np.genfromtxt(fn, delimiter="\t", names=["n", "t", "t_error"],) for fn in filenames
]

fig, ax = plt.subplots(1, 1, constrained_layout=True)

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    bottom=True,
    top=True,
    left=True,
    right=True,
)


def linear(x, slope, intercept):
    return slope * x + intercept


for d, fn in zip(data, filenames):
    cpu = (
        fn.split(".dat")[0]
        .split("/")[-1]
        .replace("_tm_", " (tm) ")
        .replace("_TM_", " TM ")
        .replace("_R_", " R ")
        .replace("_", " ")
    )
    ax.errorbar(d["n"], d["t"], yerr=d["t_error"], fmt="o", label=cpu, zorder=1)

    popt, pcov = curve_fit(linear, d["n"], d["t"], sigma=d["t_error"])

    n = np.logspace(2.5, 9.5, 100)
    ax.plot(n, linear(n, *popt), "-", color="lightgrey", zorder=0)


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Number of β-NMR probes $n$")
ax.set_ylabel("Simulation time $t$ (s)")

ax.legend(frameon=False)

ax.set_title(date.today().strftime("[%Y/%m/%d]"), loc="left")
ax.set_title("data_generator_pp", loc="center")
ax.set_title("CPU Benchmarks", loc="right")

plt.show()
