import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

import algorithm as alm

plt.style.use('dark_background')


def eta_f(theta):
    return np.cos(theta) + np.sin(theta) * 1j


R = 1
t = np.linspace(0,  np.pi, 180)
eta = eta_f(t)
eps = 0

s = np.array([1.0005 * 1j,  0.9995 * 1j, 0, 0])
sigma_infty_x = 1
sigma_infty_y = 0
sigma_infty_xy = 0
z = alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps)
sigx, sigy, sigxy = alm.get_signora(s, sigma_infty_x, sigma_infty_y, sigma_infty_xy, R, eta, eps)
sigx, sigy, sigxy = alm.transform(sigx, sigy, sigxy, t)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
ax2.set_xlim((-3, 3))
ax2.set_ylim((-3, 3))
#ax1.plot(t, sigx, label=r"$\sigma_p$", marker="v")
line, = ax1.plot(t, alm.get_la_signora(s, sigma_infty_x, sigma_infty_y, sigma_infty_xy, R, eta, eps, t)[1], label=r"$\sigma_{\theta}$")
line2, = ax2.plot(alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps).real, alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), eps).imag)
axamp = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label=r"$\epsilon$",
    valmin=-1,
    valmax=1,
    valinit=eps,
    orientation="vertical"
)
#ax1.plot(t, sigxy, label=r"$\sigma_{p\theta}$", marker="o")
def update(val):
    w = alm.get_la_signora(s, sigma_infty_x, sigma_infty_y, sigma_infty_xy, R, eta, amp_slider.val, t)[1]
    line.set_ydata(w)
    line2.set_ydata(alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), amp_slider.val).imag)
    line2.set_xdata(alm.hole(eta_f(np.linspace(0, 2 * np.pi, 100)), amp_slider.val).real)
    ax1.set_ylim((w.min() - 1, 1 + w.max()))
    fig.canvas.draw_idle()

amp_slider.on_changed(update)
ax1.legend()
plt.show()
