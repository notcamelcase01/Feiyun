import numpy as np
from matplotlib import pyplot as plt
plt.style.use('dark_background')


def eta_f(theta):
    return np.cos(theta) + np.sin(theta) * 1j


R = 1
eta = eta_f(np.linspace(0, np.pi, 50))
eps = 1/3

s = np.array([2.3992 * 1j,  0.6757 * 1j, 0, 0])
a = 1 + 1j * s
b = 1 - 1j * s
alpha = s.real
beta = s.imag

sigma_infty_x = 1
sigma_infty_y = 0
sigma_infty_xy = 0

C1 = ((alpha[1] - alpha[0]) ** 2 + beta[1] ** 2 - beta[0] ** 2) * 2
C2 = alpha[1] * (alpha[0] ** 2 - beta[0] ** 2) - alpha[0] * (alpha[1] ** 2 - beta[1] ** 2)
C3 = (alpha[0] ** 2 - beta[0] ** 2) - (alpha[1] ** 2 - beta[1] ** 2)

B = (sigma_infty_x + (alpha[1] ** 2 + beta[1] ** 2) * sigma_infty_y + 2 * alpha[1] * sigma_infty_xy)/C1
B_d = ((alpha[0] ** 2 - beta[0] ** 2 - 2 * alpha[0] * alpha[1]) * sigma_infty_y - sigma_infty_x - 2 * alpha[1] * sigma_infty_xy)/C1
C = ((alpha[0] - alpha[1]) * sigma_infty_x + C2 * sigma_infty_y + C3 * sigma_infty_xy) / (beta[1 * C1])

K1 = R/2 * (B * a[0] + (B_d + 1j * C) * a[1])
K2 = R/2 * (B * b[0] + (B_d + 1j * C) * b[1])
K3 = R/2 * (s[0] * B * a[0] + s[1] * a[1] * (B_d + 1j * C))
K4 = R/2 * (s[0] * B * b[0] + s[1] * b[1] * (B_d + 1j * C))

a[2] = 1/(s[2] - s[1]) * (s[1] * (K1 + K2.conjugate()) - (K3 + K4.conjugate()))
b[2] = 1/(s[2] - s[1]) * (s[1] * (K2 + K1.conjugate()) - (K4 + K3.conjugate()))
a[3] = 1/(s[2] - s[1]) * (s[0] * (K1 + K2.conjugate()) - (K3 + K4.conjugate()))
b[3] = 1/(s[2] - s[1]) * (s[0] * (K2 + K1.conjugate()) - (K4 + K3.conjugate()))

# z = R/2 * (a/eta + a * eta ** 2 / 3 + eps * b * eta + b * eps / 3 / eta ** 2)
dz = R/2 * (- a/eta ** 2 + 2 / 3 * eta * a + eps * b * (1 - 2 / 3 / eta ** 3))
dphiZ = (-a[2]/eta ** 2 - 2/3 * eps * beta[2]/eta**3) / dz[0]
dsiZ = (a[3] / eta ** 2 + 2/3 * b[3] * eps / eta ** 3) / dz[1]



