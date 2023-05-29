import numpy as np
from matplotlib import pyplot as plt


def eta_f(theta):
    return np.cos(theta) + np.sin(theta) * 1j


def hole(e):
    return e + 1 / 3 / e ** 2


fig2, ax = plt.subplots(1, 1, figsize=(16, 9))
theta = np.linspace(0, 2 * np.pi, 200)
w = hole(eta_f(theta))
"""
x-COORDINATE = W.REAL
y-COORDINATE = W.IMAG
Plotting the triangle
"""

ax.plot(w.real, w.imag)
ax.set_aspect("equal")
plt.show()

