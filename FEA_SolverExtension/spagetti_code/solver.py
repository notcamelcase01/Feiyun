import numpy as np


def assemble_2Dmat(kloc, iv, n, v=None):
    """
    :param v: nodes where stiffness/mat are to be places
    :param kloc: local stiffness matrix/2d mat
    :param iv: nodes where stiffness are to be placed
    :param n: DOF*number of nodes
    :return: stiffness matrix/ 2dmat to be added to global stiffness/ big 2d mat
    """
    # TODO: Remove if statement (make another function if this ain't resolved)
    # if v is None:
    #     v = iv
    K = np.zeros((n, n))
    iv = np.array(iv)
    K[iv[:, None], iv] += kloc
    return K


def assemble_force(floc, iv, n):
    """
    :param floc: local force vector
    :param iv: nodes where forces are to be placed
    :param n: DOF*number of nodes
    :return: force vector to be added to global force vector
    """
    K = np.zeros((n, 1))
    iv = np.array(iv)
    K[iv[:, None], 0] += floc
    return K


def init_gauss_points(n=3):
    """
    :param n: number of gauss points
    :return: (weights of gp,Gauss points)
    """
    if n == 1:
        wgp = np.array([2])
        egp = np.array([0])
    elif n == 2:
        wgp = np.array([1, 1])
        egp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    elif n == 3:
        wgp = np.array([5 / 9, 8 / 9, 5 / 9])
        egp = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    else:
        raise Exception("Uhm, This is wendy's, we don't, more than 3 gauss points here")
    return wgp, egp


def init_stiffness_force(nnod, DOF):
    """
    :param nnod: number of nodes
    :param DOF: Dof
    :return: zero stiffness n force
    """
    return np.zeros((nnod * DOF, nnod * DOF)), np.zeros((nnod * DOF, 1))


def impose_boundary_condition(K, f, ibc, bc):
    """
    :param K: Stiffness matrix
    :param f: force vector
    :param ibc: node at with BC is prescribed
    :param bc: boundary condition
    :return: stiffness matrix and force vector after imposed bc
    """
    f = f - (K[:, ibc] * bc)[:, None]
    f[ibc] = bc
    K[:, ibc] = 0
    K[ibc, :] = 0
    K[ibc, ibc] = 1
    return K, f


def get_displacement_vector(K, f):
    """
    :param K: Non-singular stiffness matrix
    :param f: force vector
    :return: nodal displacement
    """
    try:
        return np.linalg.solve(K, f)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("------------------")
            i = np.eye(K.shape[0])
            pin = np.linalg.lstsq(K, i, rcond=None)[0]
            return pin @ f
        else:
            raise


def get_assembly_vector(DOF, n):
    """
    :param DOF: dof
    :param n: nodes
    :return: assembly points
    """
    iv = []
    for i in n:
        for j in range(DOF):
            iv.append(DOF * i + j)
    return iv


def get_node_from_cord(icon, position, nodalArray, nelm, nodePerElement, element_type=2):
    for elm in range(nelm):
        n = icon[elm][1:]
        xloc = []
        yloc = []
        for i in range(nodePerElement):
            xloc.append(nodalArray[1][n[i]])
            yloc.append(nodalArray[2][n[i]])
        if xloc[0] <= position[0] <= xloc[2*(element_type - 1)] and yloc[0] <= position[1] <= yloc[2*(element_type - 1)]:
            eta = -1 + 2 * (position[1] - yloc[0]) / (yloc[2*(element_type - 1)] - yloc[0])
            zeta = -1 + 2 * (position[0] - xloc[0]) / (xloc[2*(element_type - 1)] - xloc[0])
            if np.isnan(np.sum(xloc)) or np.isnan(np.sum(yloc)):
                return None, None, None
            return n, zeta, eta
    return None, None, None

