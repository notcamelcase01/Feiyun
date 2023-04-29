import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.ticker import FuncFormatter, MultipleLocator
import algorithm as alm
plt.style.use('dark_background')
hole_type = alm.HoleType.ELLIPSE


def eta_f(theta):
    return np.cos(theta) + np.sin(theta) * 1j


R = 1
lamda = 0
t = np.linspace(0, np.pi, 180)
eta = eta_f(t)
eps = 0.5
t_laminate = np.array([1] * 4) / 4
theta_laminate = np.array([0, 90, 90, 0]) * np.pi / 180
E1 = 181 * 10 ** 9
E2 = 10.30 * 10 ** 9
V12 = 0.28
V21 = .02
G = E1 / (2 * (1 + V12))
G = 7.17 * 10 ** 9
s = alm.get_anisotrpic_coefficients(t_laminate, theta_laminate, E1, E2, V12, G, V21)
sigma = 1
beta = np.pi / 4
z = alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps, hole_type)
fig, (wa, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 2.5, 1]})
wa.axis('off')
ax2.set_xlim((-3, 3))
ax2.set_ylim((-3, 3))
x_ax3 = np.linspace(-3, 3, 2)
line, = ax1.plot(t, alm.get_la_signora(s, sigma, lamda, beta, R, eta, eps, t, hole_type)[1], label=r"$\sigma_{\theta}$")
line2, = ax2.plot(alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps, hole_type).real,
                  alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps, hole_type).imag)
line3, = ax2.plot(x_ax3, np.tan(np.pi / 2 - beta) * x_ax3)

eps_slider = fig.add_axes([0.04, 0.20, 0.012, 0.63])
load_slider = fig.add_axes([0.08, 0.20, 0.012, 0.63])
hole_radio = fig.add_axes([0.10, 0.65, 0.15, 0.15])
# bi_asis = fig.add_axes([0.11, 0.45, 0.13, 0.015])


shape_radio = RadioButtons(
    hole_radio, ['ELLIPSE', 'TRIANGLE', 'SQUARE'],
    radio_props=dict(edgecolor=['white', 'white', 'white']))

#
# bi_slider = Slider(
#     ax=bi_asis,
#     label=r"$\lambda$",
#     valmin=0.0,
#     valmax=1,
#     valinit=lamda,
#     orientation="horizontal"
# )


amp_slider = Slider(
    ax=eps_slider,
    label=r"$\epsilon$",
    valmin=0.0,
    valmax=1,
    valinit=eps,
    orientation="vertical"
)

loading_slider = Slider(
    ax=load_slider,
    label=r"$\beta$",
    valmin=-np.pi / 2,
    valmax=np.pi / 2,
    valinit=beta,
    orientation="vertical"
)


def update(val):
    global eps
    global beta
    eps = amp_slider.val
    beta = loading_slider.val
    w = alm.get_la_signora(s, sigma, lamda, beta, R, eta, eps, t, hole_type)[1]
    line.set_ydata(w)
    line2.set_ydata(alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps, hole_type).imag)
    line2.set_xdata(alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps, hole_type).real)
    line3.set_ydata(x_ax3 * np.tan(np.pi / 2 - beta))
    ax1.set_ylim((w.min() - 1, 1 + w.max()))
    fig.canvas.draw_idle()


def change_hole(val):
    d = {'ELLIPSE': alm.HoleType.ELLIPSE, 'TRIANGLE': alm.HoleType.TRIANGLE, 'SQUARE': alm.HoleType.SQUARE}
    f = d[val]
    global hole_type
    hole_type = d[val]
    w = alm.get_la_signora(s, sigma, lamda, loading_slider.val, R, eta, eps, t, hole_type)[1]
    line.set_ydata(w)
    line2.set_ydata(alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps, hole_type).imag)
    line2.set_xdata(alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps, hole_type).real)
    line3.set_ydata(x_ax3 * np.tan(np.pi / 2 - beta))

    ax1.set_ylim((w.min() - 1, 1 + w.max()))
    fig.canvas.draw_idle()


amp_slider.on_changed(update)
loading_slider.on_changed(update)
shape_radio.on_clicked(change_hole)

ax1.set_ylabel("Normalized Stress")
ax1.set_xlabel("theta curvilinear coordinate")
plt.suptitle("Behaviour of stress under uni-axial loading on plate weakened by hole")
ax1.legend()
ax2.set_aspect('equal')
ax2.set_title('Plate with Hole and stress direction', fontsize=8)
ax1.xaxis.set_major_formatter(FuncFormatter(
    lambda val, pos: r'{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
))
ax1.xaxis.set_major_locator(MultipleLocator(base=np.pi / 4))
plt.show()
