import numpy as np


def hole(e, eps=1):
    return e + eps / 3 / e ** 2


def get_signora(s, sigma_infty_x, sigma_infty_y, sigma_infty_xy, R, eta, eps):
    a = 1 + 1j * s
    b = 1 - 1j * s
    alpha = s.real
    beta = s.imag

    C1 = ((alpha[1] - alpha[0]) ** 2 + beta[1] ** 2 - beta[0] ** 2) * 2
    C2 = alpha[1] * (alpha[0] ** 2 - beta[0] ** 2) - alpha[0] * (alpha[1] ** 2 - beta[1] ** 2)
    C3 = (alpha[0] ** 2 - beta[0] ** 2) - (alpha[1] ** 2 - beta[1] ** 2)
    B = (sigma_infty_x + (alpha[1] ** 2 + beta[1] ** 2) * sigma_infty_y + 2 * alpha[1] * sigma_infty_xy) / C1
    B_d = ((alpha[0] ** 2 - beta[0] ** 2 - 2 * alpha[0] * alpha[1]) * sigma_infty_y - sigma_infty_x - 2 * alpha[1] * sigma_infty_xy) / C1
    C = ((alpha[0] - alpha[1]) * sigma_infty_x + C2 * sigma_infty_y + C3 * sigma_infty_xy) / (beta[1] * C1)

    K1 = R / 2 * (B * a[0] + (B_d + 1j * C) * a[1])
    K2 = R / 2 * (B * b[0] + (B_d + 1j * C) * b[1])
    K3 = R / 2 * (s[0] * B * a[0] + s[1] * a[1] * (B_d + 1j * C))
    K4 = R / 2 * (s[0] * B * b[0] + s[1] * b[1] * (B_d + 1j * C))

    a[2] = 1 / (s[0] - s[1]) * (s[1] * (K1 + K2.conjugate()) - (K3 + K4.conjugate()))
    b[2] = 1 / (s[0] - s[1]) * (s[1] * (K2 + K1.conjugate()) - (K4 + K3.conjugate()))
    a[3] = 1 / (s[0] - s[1]) * (s[0] * (K1 + K2.conjugate()) - (K3 + K4.conjugate()))
    b[3] = 1 / (s[0] - s[1]) * (s[0] * (K2 + K1.conjugate()) - (K4 + K3.conjugate()))

    dz1 = R / 2 * (- a[0] / eta ** 2 + 2 * eps / 3 * eta * a[0] + b[0] * (1 - 2 * eps / 3 / eta ** 3))
    dz2 = R / 2 * (- a[1] / eta ** 2 + 2 * eps / 3 * eta * a[1] + b[1] * (1 - 2 * eps / 3 / eta ** 3))

    dphiZ = (-a[2] / eta ** 2 - 2 / 3 * eps * b[2] / eta ** 3) / dz1
    dsiZ = (a[3] / eta ** 2 + 2 / 3 * b[3] * eps / eta ** 3) / dz2

    sigx = sigma_infty_x + 2 * (s[0] ** 2 * dphiZ + s[1] ** 2 * dsiZ).real
    sigy = sigma_infty_y + 2 * (dphiZ + dsiZ).real
    sigxy = sigma_infty_xy - 2 * (s[0] * dphiZ + s[1] * dsiZ).real
    return sigx, sigy, sigxy


def transform(sigx, sigy, sigxy, t):
    sigP = np.cos(t) ** 2 * sigx + np.sin(t) ** 2 * sigy + 2 * np.sin(t) * np.cos(t) * sigxy
    sigTheta = np.sin(t) ** 2 * sigx + np.cos(t) ** 2 * sigy - 2 * np.sin(t) * np.cos(t) * sigxy
    sigthetaP = -np.cos(t) * np.sin(t) * sigx + np.sin(t) * np.cos(t) * sigy + (np.cos(t) ** 2 - np.sin(t) ** 2) * sigxy
    return sigP, sigTheta, sigthetaP


def get_la_signora(s, sigma_infty_x, sigma_infty_y, sigma_infty_xy, R, eta, eps, t):
    a, b, c = get_signora(s, sigma_infty_x, sigma_infty_y, sigma_infty_xy, R, eta, eps)
    a, b, c = transform(a, b, c, t)
    return b
