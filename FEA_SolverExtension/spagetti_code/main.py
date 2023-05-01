import numpy as np
from matplotlib import pyplot as plt
from Composite import composite_stiffness as stiffness
import gencon as gencon
import solver as sol
import fsdt as fsdt
import time
from Analytical import algorithm as alm

plt.style.use('dark_background')

E1 = 181 * 10 ** 9
E2 = 10.30 * 10 ** 9
V12 = 0.28
# V21 = .02
G = E1 / (2 * (1 + V12))
# G = 7.17 * 10 ** 9
t = np.array([1, 1, 1, 1]) / 4
H = np.sum(t)
theta = np.array([0, 90, 90, 0]) * np.pi/180
ABD = stiffness.get_ABD(t, theta, E1, E2, V12, G) / H
print(ABD)
element_type = 2
OVERRIDE_REDUCED_INTEGRATION = False
DIMENSIONS = 2
DOF = 2
GAUSS_POINT_REQUIRED = 3

Hx = 8
Hy = 8

nx = 15
ny = 15
lx = 1
ly = 1
by_max = 0.6 * ly
by_min = 0.4 * ly
bx_max = 0.5 * lx + 0.1
bx_min = 0.5 * lx - 0.1
connectivityMatrix, nodalArray, (X, Y), nodalArray1 = gencon.get_2d_connectivity_hole(nx, ny, lx, ly, Hx, Hy, by_max, by_min, bx_max, bx_min)
numberOfElements = connectivityMatrix.shape[0]
numberOfNodes = nodalArray.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINT_REQUIRED)


KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
nodePerElement = element_type ** DIMENSIONS
hole_elements = []
tik = time.time()
for elm in range(numberOfElements):
    KK = 1
    n = connectivityMatrix[elm][1:]
    xloc = []
    yloc = []
    for i in range(nodePerElement):
        xloc.append(nodalArray1[1][n[i]])
        yloc.append(nodalArray1[2][n[i]])
    if np.isnan(np.sum(xloc)) or np.isnan(np.sum(yloc)):
        """
        CHECKING IF ELEMENT IS IN HOLE
        if THERE IS HOLE THEN STIFFNESS WILL BE 0 
        """
        KK = 0
        hole_elements.append(elm)
    xloc = []
    yloc = []
    for i in range(nodePerElement):
        xloc.append(nodalArray[1][n[i]])
        yloc.append(nodalArray[2][n[i]])
    xloc = np.array(xloc)[:, None]
    yloc = np.array(yloc)[:, None]
    Jy = 0.5 * (yloc[2][0] - yloc[0][0])
    if np.isclose(xloc[1][0], lx):
        q0 = 100000
        xeta = 1
    # elif yloc[0][0] < ly < yloc[1][0]:
    #     q0 = 1
    #     xeta = -1 + 2 * (ly/2 - yloc[0][0]) / (yloc[1][0] - yloc[0][0])
    else:
        xeta = 0
        q0 = 0
    kloc, floc = sol.init_stiffness_force(nodePerElement, DOF)
    for ygp in range(len(weightOfGaussPts)):
        for xgp in range(len(weightOfGaussPts)):
            N, Nx, Ny = fsdt.get_lagrange_shape_function(gaussPts[xgp], gaussPts[ygp], element_type)
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            Nx, Ny = Nx * Jinv[0, 0] + Ny * Jinv[0, 1], Nx * Jinv[1, 0] + Ny * Jinv[1, 1]
            B1 = fsdt.get_b1_matrix(Nx, Ny)
            kloc += B1.T @ ABD[:3, :3] @ B1 * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
        N, Nx, Ny = fsdt.get_lagrange_shape_function(xeta, gaussPts[ygp], element_type)
        floc += q0 * (fsdt.get_n_matrix(N).T @ np.array([[H * 1, 0]]).T) * weightOfGaussPts[ygp] * Jy

    iv = np.array(sol.get_assembly_vector(DOF, n))
    fg[iv[:, None], 0] += floc
    KG[iv[:, None], iv] += kloc * KK
tf = np.sum(fg)
encastrate = np.where((np.isclose(nodalArray[1], 0)))[0]
iv = sol.get_assembly_vector(DOF, encastrate)
for i in iv:
    KG, fg = sol.impose_boundary_condition(KG, fg, i, 0)
encastrate = np.where((np.isclose(nodalArray[1], lx)))[0]
# iv = sol.get_assembly_vector(DOF, encastrate)
# for i in iv:
#     KG, fg = sol.impose_boundary_condition(KG, fg, i, .001 * lx)
u = sol.get_displacement_vector(KG, fg)
tok = time.time()
print(tok - tik)
u0 = []
v0 = []
for i in range(numberOfNodes):
    x = nodalArray[1][i]
    u0.append(u[DOF * i][0])
    v0.append(u[DOF * i + 1][0])
reqN, zeta, eta = sol.get_node_from_cord(connectivityMatrix, (lx * 0.5, ly * 0.6 + .0000001), nodalArray1, numberOfElements, nodePerElement, element_type)
if reqN is None:
    raise Exception("Chose a position inside plate plis")
N, Nx, Ny = fsdt.get_lagrange_shape_function(zeta, eta, element_type)
b1 = fsdt.get_b1_matrix(Nx, Ny)
z1 = fsdt.get_z1_matrix(0)
iv = np.array(sol.get_assembly_vector(DOF, reqN))
u0 = u[iv[:, None], 0]
e = z1 @ b1 @ u0
q = stiffness.transform(stiffness.get_normal_stiffness(E1, E2, V12, G), 0)
sigma = ABD[:3, :3] @ e
print(sigma.T)
ssi = alm.transform(sigma[0][0], sigma[1][0], sigma[2][0], np.pi / 4)
print(np.array(ssi))
sigma = q @ e
print(sigma.T)
ssi = alm.transform(sigma[0][0], sigma[1][0], sigma[2][0], np.pi / 4)
print(np.array(ssi))

