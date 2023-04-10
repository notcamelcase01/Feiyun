import numpy as np
from matplotlib import pyplot as plt
plt.style.use('dark_background')


def eta(theta):
    return np.cos(theta) + np.sin(theta) * 1j


def circle(e):
    return e


def phiZ(e, sigma):
    return -sigma / 4 / e ** 2


def siZ(e, sigma):
    return 3 * sigma / 4



fig, (ax, ay) = plt.subplots(1, 2, figsize=(12, 6))
t = np.linspace(0, np.pi, 100)
Z1 = circle(eta(t))
sigy = (1 + 2 * (phiZ(eta(t), 1) + siZ(eta(t), 1)).real)
sigx = (- 2 * (phiZ(eta(t), 1) + siZ(eta(t), 1)).real)
sigxy = (- 2 * (1j * phiZ(eta(t), 1) - 1j * siZ(eta(t), 1)).real)
sigP = np.cos(t) ** 2 * sigx + np.sin(t) ** 2 * sigy + 2 * np.sin(t) * np.cos(t) * sigxy
sigTheta = np.sin(t) ** 2 * sigx + np.cos(t) ** 2 * sigy - 2 * np.sin(t) * np.cos(t) * sigxy
sigthetaP = -np.cos(t) * np.sin(t) * sigx + np.sin(t) * np.cos(t) * sigy + (np.cos(t) ** 2 - np.sin(t) ** 2)*sigxy


ax.plot(t, sigP)

ay.plot(Z1.real, Z1.imag)
ay.set_aspect('equal')
plt.show()
