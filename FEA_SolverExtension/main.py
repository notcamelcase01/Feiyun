import numpy as np
from matplotlib import pyplot as plt
from Composite import composite_stiffness as stiffness
import gencon as gencon
import solver as sol
import fsdt as fsdt
import time

plt.style.use('dark_background')
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

E1 = 30 * 10**6
E2 = 30 * 10**6
mu = 0.3
G = E1/(2 * (1 + mu))
t = np.array([1/100])
H = np.sum(t)
theta = np.array([0]) * np.pi/180
ABD = stiffness.get_ABD(t, theta, E1, E2, mu, G)

element_type = 2
OVERRIDE_REDUCED_INTEGRATION = False
DIMENSIONS = 2
DOF = 5
GAUSS_POINT_REQUIRED = 2

nx = 10
ny = 10
lx = 1
ly = 1
connectivityMatrix, nodalArray, (X, Y) = gencon.get_2D_connectivity_Hybrid(nx, ny, lx, ly, element_type)
numberOfElements = connectivityMatrix.shape[0]
numberOfNodes = nodalArray.shape[1]
weightOfGaussPts, gaussPts = sol.init_gauss_points(GAUSS_POINT_REQUIRED)
reduced_wts, reduced_gpts = sol.init_gauss_points(1 if (not OVERRIDE_REDUCED_INTEGRATION and
                                                        element_type == 2) else GAUSS_POINT_REQUIRED)

KG, fg = sol.init_stiffness_force(numberOfNodes, DOF)
nodePerElement = element_type ** DIMENSIONS

tik = time.time()
for elm in range(numberOfElements):
    n = connectivityMatrix[elm][1:]
    xloc = []
    yloc = []
    for i in range(nodePerElement):
        xloc.append(nodalArray[1][n[i]])
        yloc.append(nodalArray[2][n[i]])
    xloc = np.array(xloc)[:, None]
    yloc = np.array(yloc)[:, None]
    kloc, floc = sol.init_stiffness_force(nodePerElement, DOF)
    for xgp in range(len(weightOfGaussPts)):
        for ygp in range(len(weightOfGaussPts)):
            N, Nx, Ny = fsdt.get_lagrange_shape_function(gaussPts[xgp], gaussPts[ygp], element_type)
            xx = N.T @ xloc
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            Nx, Ny = Nx * Jinv[0, 0] + Ny * Jinv[0, 1], Nx * Jinv[1, 0] + Ny * Jinv[1, 1]
            B1 = fsdt.get_b1_matrix(Nx, Ny)
            kloc += B1.T @ ABD[:6, :6] @ B1 * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
            floc += (fsdt.get_n_matrix(N).T @ np.array([[0, 0, -1000000, 0, 0]]).T) * weightOfGaussPts[xgp] * weightOfGaussPts[ygp] * np.linalg.det(J)
    for xgp in range(len(reduced_wts)):
        for ygp in range(len(reduced_wts)):
            N, Nx, Ny = fsdt.get_lagrange_shape_function(reduced_gpts[xgp], reduced_gpts[ygp], element_type)
            J = np.zeros((2, 2))
            J[0, 0] = Nx.T @ xloc
            J[0, 1] = Nx.T @ yloc
            J[1, 0] = Ny.T @ xloc
            J[1, 1] = Ny.T @ yloc
            Jinv = np.linalg.inv(J)
            Nx, Ny = Nx * Jinv[0, 0] + Ny * Jinv[0, 1], Nx * Jinv[1, 0] + Ny * Jinv[1, 1]
            B2 = fsdt.get_b2_matrix(N, Nx, Ny)
            kloc += B2.T @ ABD[6:, 6:] @ B2 * reduced_wts[xgp] * reduced_wts[ygp] * np.linalg.det(J)
    iv = np.array(sol.get_assembly_vector(DOF, n))
    fg[iv[:, None], 0] += floc
    KG[iv[:, None], iv] += kloc

encastrate = np.where((np.isclose(nodalArray[1], 0)) | (np.isclose(nodalArray[1], lx)) | (np.isclose(nodalArray[2], 0)) | (np.isclose(nodalArray[2], ly)))[0]
iv = sol.get_assembly_vector(DOF, encastrate)
for i in iv:
    KG, fg = sol.impose_boundary_condition(KG, fg, i, 0)
u = sol.get_displacement_vector(KG, fg)
tok = time.time()
print(tok - tik)
w0 = []
for i in range(numberOfNodes):
    x = nodalArray[1][i]
    w0.append(u[DOF * i + 2][0])
reqN, zeta, eta = sol.get_node_from_cord(connectivityMatrix, (lx/2, ly/2), nodalArray, numberOfElements, nodePerElement, element_type)
if reqN is None:
    raise Exception("Chose a position inside plate plis")
N, Nx, Ny = fsdt.get_lagrange_shape_function(zeta, eta, element_type)
wt = np.array([w0[i] for i in reqN])[:, None]
xxx = N.T @ wt
print("Displacement at mid point (mm)", xxx[0][0] * 1000)
w0 = np.array(w0)
w0 = w0 / max(w0.max(), w0.min(), key=abs)
w0 = w0.reshape(((element_type - 1) * ny + 1, (element_type - 1) * nx + 1))
np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)
print(w0)
fig2 = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, w0)
ax.set_aspect('equal')
ax.set_title('w0 is scaled to make graph look prettier')
ax.set_axis_off()
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.contourf(X, Y, w0, 100, cmap='jet')
ax.set_title('Contour Plot')
ax.set_xlabel('_x')
ax.set_ylabel('_y')
ax.set_aspect('equal')
plt.show()

