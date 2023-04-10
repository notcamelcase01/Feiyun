import numpy as np
from matplotlib import pyplot as plt
plt.style.use('dark_background')


def eta(theta):
    return np.cos(theta) + np.sin(theta) * 1j


def triangle(e, n=1):
    return e + n/3/e**2#+ 1/45/e**5 + 1/162/e**8 + 7/2673/e**11


def phiZ(e, sigma):
    return sigma * (0.5 / e ** 2 +  1/6/e ** 3) / (1 - 2/3/e ** 3)


def siZ(e, sigma):
    return sigma * (0.25 / e ** 2 +  1/3/e ** 3) / (-1/e ** 2 - 2/3/e ** 3)



fig, (ax, ay) = plt.subplots(1, 2, figsize=(12, 6))
t = np.linspace(0, 2*np.pi, 100)
Z1 = triangle(eta(t))
sigy = (1 + 2 * (phiZ(eta(t), 1) + siZ(eta(t), 1)).real)
ax.plot(t, sigy)
ay.plot(Z1.real, Z1.imag)
ay.set_aspect('equal')
plt.show()
