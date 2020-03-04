import numpy as np
import matplotlib.pyplot as plt


def W(theta, A=1.0, P=1.0, v_over_c=1.0):
    return 1.0 + v_over_c * A * P * np.cos(theta)


fig, ax = plt.subplots(1, 1)

for n in [300, 200, 100, 50]:
    theta = np.linspace(0, 2 * np.pi, n)
    w = W(theta, -1.0 / 3.0, 0.7, 1.0)

    ax.plot(
        theta, w, ".-", label="$n = %d$" % n,
    )

ax.axvspan(0.0 * np.pi, 0.5 * np.pi, color="lightgrey", zorder=0)
ax.axvspan(1.5 * np.pi, 2.0 * np.pi, color="lightgrey", zorder=0)

ax.legend(title=r"$W(\theta; A=1, P=1)$")

ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$W(\theta)$")

ax.set_xlim(0.0 * np.pi, 2.0 * np.pi)

plt.show()
