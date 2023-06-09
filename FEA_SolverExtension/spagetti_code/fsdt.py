import numpy as np


def get_lagrange_shape_function(x, y, element_type=2):
    """
    :param x: x_gauss point
    :param y: y_gauss point
    :param element_type:  2 means quad element
    :return: lagrange function
    """
    xi = np.array((-1, 0, 1, 1, 1, 0, -1, -1, 0))
    yi = np.array((-1, -1, -1, 0, 1, 1, 1, 0, 0))
    seq = np.array(((-1, 1, 1, -1), (-1, -1, 1, 1)))
    if element_type == 3:
        N = ((1.5 * xi**2 - 1) * x**2 + 0.5 * xi * x + 1 - xi**2) * ((1.5 * yi**2 - 1) * y**2 + 0.5 * yi * y + 1 - yi**2)
        Nx = ((1.5 * xi**2 - 1) * x * 2 + 0.5 * xi) * ((1.5 * yi**2 - 1) * y**2 + 0.5 * yi * y + 1 - yi**2)
        Ny = ((1.5 * xi**2 - 1) * x**2 + 0.5 * xi * x + 1 - xi**2) * ((1.5 * yi**2 - 1) * y * 2 + 0.5 * yi)
    elif element_type == 2:
        N = 0.25 * (1 + seq[0] * x) * (1 + seq[1] * y)
        Nx = 0.25 * (seq[0] * (1 + seq[1] * y))
        Ny = 0.25 * (seq[1] * (1 + seq[0] * x))
    else:
        raise Exception("Uhm, This is wendy's, we don't, more than 3 nodes here")
    return N[:, None], Nx[:, None], Ny[:, None]


def get_b1_matrix(Nx, Ny):
    """
    :param Nx: Nx
    :param Ny: Ny
    :return: B1
    """
    B1 = np.zeros((3, 2 * len(Nx)))
    for i in range(len(Nx)):
        B1[0, 1 * i] = Nx[i][0]
        B1[1, 1 * i + 1] = Ny[i][0]
        B1[2, 1 * i] = Nx[i][0]
        B1[2, 1 * i + 1] = Ny[i][0]
    return B1


def get_n_matrix(N):
    N1 = np.zeros((2, 2 * len(N)))
    for i in range(len(N)):
        N1[0, 1 * i] = N[i][0]
        N1[1, 1 * i + 1] = N[i][0]
    return N1


def get_z1_matrix(z):
    return np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])

