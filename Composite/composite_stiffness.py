import numpy as np


def get_normal_stiffness(E1, E2, V12, G12):
    """
    :param E1: E1
    :param E2: E2
    :param V12: V
    :param G12: G
    :return: normal stiffness
    """
    C = np.zeros((3, 3))
    C[0][0] = 1 / E1
    C[0][1] = -V12 / E1
    C[1][0] = -V12 / E1
    C[1][1] = 1 / E2
    C[2][2] = 1 / G12
    return np.linalg.inv(C)


def get_shear_stiffness(G):
    """
    :param G: G
    :return: Sheer stiffness
    """
    return np.array([[G, 0], [0, G]]) * 5/6


def transform(Q, theta):
    """
    :param Q: stiffness
    :param theta: transformation angle
    :return: transformed stiffness
    """
    if Q.shape[0] == 3:
        Ts = np.array([[np.cos(theta) ** 2, np.sin(theta) ** 2, 2 * np.sin(theta) * np.cos(theta)],
                   [np.sin(theta) ** 2, np.cos(theta) ** 2, -2 * np.sin(theta) * np.cos(theta)],
                   [-np.sin(theta) * np.cos(theta), np.sin(theta) * np.cos(theta), np.cos(theta) ** 2 - np.sin(theta) ** 2]])
        return np.linalg.inv(Ts) @ Q @ np.linalg.inv(Ts.T)
    else:
        Ts = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        return np.linalg.inv(Ts) @ Q @ Ts


def get_ABD(t, theta, E1, E2, V12, G12):
    """
    :param t: thickness of each laminate
    :param theta: direction
    :param E1: E1
    :param E2: E2
    :param V12: V (poison's ratio)
    :param G12: G
    :return: ABD
    """
    Q126 = get_normal_stiffness(E1, E2, V12, G12)
    Q45 = get_shear_stiffness(G12)
    if len(t) is not len(theta):
        raise Exception("theta and t have different amount of elements plis recheck")
    height_of_lamina = np.sum(t)
    ords = [-height_of_lamina / 2]
    for i in range(len(t)):
        ords.append((ords[i] + t[i]))
    ABD = np.zeros((8, 8))
    for k in range(0, len(theta)):
        Q = transform(Q126, theta[k])
        ABD[0:3, 0:3] += Q * (ords[k + 1] - ords[k])
        ABD[3:6, 0:3] += 1 / 2 * Q * (-ords[k + 1] ** 2 + ords[k] ** 2)
        ABD[0:3, 3:6] += 1 / 2 * Q * (-ords[k + 1] ** 2 + ords[k] ** 2)
        ABD[3:6, 3:6] += 1 / 3 * Q * (ords[k + 1] ** 3 - ords[k] ** 3)
        Q = transform(Q45, theta[k])
        ABD[6:8, 6:8] += Q * (ords[k + 1] - ords[k])
    return ABD


if __name__ == '__main__':
    E1_ = 150
    E2_ = 20
    G = 5
    V = 0.3
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    theta_ = [0, 90, 90, 55]
    t_ = [.15, .15, .55, .15]
    theta_ = np.array(theta_) * np.pi / 180
    ABD_ = get_ABD(t_, theta_, E1_, E2_, V, G)
    print(ABD_)
