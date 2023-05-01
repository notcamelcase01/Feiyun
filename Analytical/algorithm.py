import numpy as np
from enum import IntEnum
from Composite import composite_stiffness as stiffness


class HoleType(IntEnum):
    ELLIPSE = 1
    TRIANGLE = 2
    SQUARE = 3


def hole(e, eps=1, hole_type=1):
    if hole_type == HoleType.ELLIPSE:
        eps = 1 - eps
        return e + (1 - eps) / (1 + eps) / e
    elif hole_type == HoleType.TRIANGLE:
        return e + eps / 3 / e ** 2
    elif hole_type == HoleType.SQUARE:
        return e - eps / 6 / e ** 3 + eps / 56 / e ** 7
    else:
        return e + (1 - eps) / (1 + eps) / e


def get_stress_fn(hole_type, a, b, R, eta, eps):
    if hole_type == HoleType.TRIANGLE:
        dz1 = R / 2 * (- a[0] / eta ** 2 + 2 * eps / 3 * eta * a[0] + b[0] * (1 - 2 * eps / 3 / eta ** 3))
        dz2 = R / 2 * (- a[1] / eta ** 2 + 2 * eps / 3 * eta * a[1] + b[1] * (1 - 2 * eps / 3 / eta ** 3))
        dphiZ = (-a[2] / eta ** 2 - 2 / 3 * eps * b[2] / eta ** 3) / dz1
        dsiZ = (a[3] / eta ** 2 + 2 / 3 * b[3] * eps / eta ** 3) / dz2

    elif hole_type == HoleType.SQUARE:
        dz1 = R / 2 * (- a[0] / eta ** 2 - eps / 2 * eta ** 2 * a[0] - 7 * eps / 56 * eta ** 6 * a[0] + b[0] * (1 + eps / 2 / eta ** 4 - 7 / 56 * eps / eta ** 8))
        dz2 = R / 2 * (- a[1] / eta ** 2 - eps / 2 * eta ** 2 * a[1] - 7 * eps / 56 * eta ** 6 * a[1] + b[1] * (1 + eps / 2 / eta ** 4 - 7 / 56 * eps / eta ** 8))
        dphiZ = (-a[2] / eta ** 2 + 1 / 2 * eps * b[2] / eta ** 4 - b[2] * eps * 7 / 56 / eta ** 8) / dz1
        dsiZ = (a[3] / eta ** 2 - 1 / 2 * b[3] * eps / eta ** 4 + b[2] * eps * 7 / 56 / eta ** 8) / dz2

    elif hole_type == HoleType.ELLIPSE:
        eps = 1 - eps
        if eps == 0:
            eps = 1e-4
        dz1 = R / 2 * (- a[0] / eta ** 2 + a[0] * (1 - eps) / (1 + eps) + b[0] * (1 - (1 - eps) / (1 + eps) / eta ** 2))
        dz2 = R / 2 * (- a[1] / eta ** 2 + a[1] * (1 - eps) / (1 + eps) + b[1] * (1 - (1 - eps) / (1 + eps) / eta ** 2))
        dphiZ = (-a[2] / eta ** 2 - (1 - eps) / (1 + eps) * b[2] / eta ** 2) / dz1
        dsiZ = (a[3] / eta ** 2 + (1 - eps) / (1 + eps) * b[3] / eta ** 2) / dz2

    else:
        eps = 1 - eps
        if eps == 0:
            eps = 1e-4
        dz1 = R / 2 * (- a[0] / eta ** 2 + a[0] * (1 - eps) / (1 + eps) + b[0] * (1 - (1 - eps) / (1 + eps) / eta ** 2))
        dz2 = R / 2 * (- a[1] / eta ** 2 + a[1] * (1 - eps) / (1 + eps) + b[1] * (1 - (1 - eps) / (1 + eps) / eta ** 2))
        dphiZ = (-a[2] / eta ** 2 - (1 - eps) / (1 + eps) * b[2] / eta ** 2) / dz1
        dsiZ = (a[3] / eta ** 2 + (1 - eps) / (1 + eps) * b[3] / eta ** 2) / dz2
    return dphiZ, dsiZ


def get_signora(s, sigma, lamda, loading_angle, R, eta, eps, hole_type=1):
    a = 1 + 1j * s
    b = 1 - 1j * s
    alpha = s.real
    beta = s.imag
    sigma_infty_x = sigma / 2 * (lamda + 1 + (lamda - 1) * np.cos(2 * loading_angle))
    sigma_infty_y = sigma / 2 * (lamda + 1 - (lamda - 1) * np.cos(2 * loading_angle))
    sigma_infty_xy = sigma / 2 * ((lamda - 1) * np.sin(2 * loading_angle))
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

    dphiZ, dsiZ = get_stress_fn(hole_type, a, b, R, eta, eps)

    sigx = sigma_infty_x + 2 * (s[0] ** 2 * dphiZ + s[1] ** 2 * dsiZ).real
    sigy = sigma_infty_y + 2 * (dphiZ + dsiZ).real
    sigxy = sigma_infty_xy - 2 * (s[0] * dphiZ + s[1] * dsiZ).real
    return sigx, sigy, sigxy


def transform(sigx, sigy, sigxy, t):
    sigP = np.cos(t) ** 2 * sigx + np.sin(t) ** 2 * sigy + 2 * np.sin(t) * np.cos(t) * sigxy
    sigTheta = np.sin(t) ** 2 * sigx + np.cos(t) ** 2 * sigy - 2 * np.sin(t) * np.cos(t) * sigxy
    sigthetaP = -np.cos(t) * np.sin(t) * sigx + np.sin(t) * np.cos(t) * sigy + (np.cos(t) ** 2 - np.sin(t) ** 2) * sigxy
    return sigP, sigTheta, sigthetaP


def get_la_signora(s, sigma, lamda, la, R, eta, eps, t, hole_type=1):
    a, b, c = get_signora(s, sigma, lamda, la, R, eta, eps, hole_type)
    a, b, c = transform(a, b, c, t)
    return a, b, c


def get_anisotrpic_coefficients(t, theta, E1, E2, V12, G, V21 = None):
    h = np.sum(t)
    b = np.zeros((3, 3))
    Q = stiffness.get_normal_stiffness(E1, E2, V12, G, V21)
    for i in range(len(t)):
        b += stiffness.transform(Q, theta[i]) * t[i]
    b /= h
    # B = b[0, 0] * b[1, 1] * b[2, 2] + b[0, 0] * b[1, 2] ** 2 + 2 * b[0, 1] * b[1, 2] * b[0, 2] - b[2, 2] * b[0, 1] ** 2 - b[1, 1] * b[0, 2] ** 2
    # a11 = (b[1, 1] * b[2, 2] - b[1, 2] * 2) / B
    # a12 = (b[0, 2] * b[1, 2] - b[0, 1]* b[2, 2]) / B
    # a16 = (b[0, 1] * b[1, 2] - b[0, 2] * b[1, 1]) / B
    # a22 = (b[0, 0] * b[2, 2] - b[0, 2] ** 2) / B
    # a26 = (b[0, 1] * b[0, 2] - b[0, 0] * b[1, 2]) / B
    # a66 = (b[0, 0] * b[1, 1] - b[0, 1] ** 2) / B
    a = np.linalg.inv(b)
    a11 = a[0, 0]
    a12 = a[0, 1]
    a16 = a[0, 2]
    a22 = a[1, 1]
    a26 = a[1, 2]
    a66 = a[2, 2]
    s = np.roots((a11, -2 * a16, (2 * a12 + a66), - 2 * a26, a22))
    return np.array([s[0], s[2], 0, 0])


if __name__ == '__main__':
    S11 = 12.7
    S22 = 2.2 * 10 ** (-2)
    S33 = 0
    S11, S22, S33 = transform(S11, S22, S33, np.pi / 2)
    print(S11, S22, S33)
    S11 = 5.32 * 10e-1
    S22 = 4.16 * 10e-3
    S33 = 0
    S11, S22, S33 = transform(S11, S22, S33, np.pi / 2)
    print(S11, S22, S33)