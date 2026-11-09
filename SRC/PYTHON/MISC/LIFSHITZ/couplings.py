import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph, format_value
from tenet.utils.sty import COLORS
from tenet.utils.math import PI

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():

	x = np.linspace(-1, 0.5, 10001)

	J = np.sqrt(2) * np.cos(x)
	g = np.sqrt(2) * np.sin(x)

	e_0 = - 2 * J / PI + 32 * g / (9 * PI ** 3)

	A_phi = J / 2 - 200 * g / (9 * PI ** 2)
	A_theta = 2 * J - 32 * g / (9 * PI ** 2)
	kappa_phi = - J / 24 + 1130 * g / (27 * PI ** 2)
	# kappa_theta = - J / 6 + 8 * g / (27 * PI ** 2)
	u_phi = - PI * J / 48 + 385 * g / (27 * PI)
	# u_theta = - PI * J / 3 + 16 * g / (27 * PI)
	u_phi_theta = - PI * J / 2 + 200 * g / (9 * PI)
	# g_u = - 160 * g / (3 * PI ** 3)

	v = 2 * np.sqrt(A_phi * A_theta)
	eta = np.sqrt(A_phi / A_theta)

	fig, ax = plt.subplots(1, 2, gridspec_kw={"wspace": 0.3}, figsize=(3, 3))

	neg = np.where((x <= 0) & (eta > 0) & (eta <= 1))[0]
	pos = np.where((x >= 0) & (eta > 0) & (eta <= 1))[0]

	p, = ax[0].plot(x[neg], A_phi[neg], marker='none', color=COLORS[2], linewidth=2, label=r"$\mathfrak{a}^{-1}\,\rho^{\,}_{\phi}$", zorder=101)
	p, = ax[0].plot(x[neg], A_theta[neg], marker='none', color=COLORS[2], linestyle=(0, (0, 1.5)), linewidth=2, label=r"$\mathfrak{a}^{-1}\,\rho^{\,}_{\theta}$", zorder=101)

	p, = ax[1].plot(x[pos], A_phi[pos], marker='none', color=COLORS[2], linewidth=2, label=r"$\mathfrak{a}^{-1}\,\rho^{\,}_{\phi}$", zorder=101)
	p, = ax[1].plot(x[pos], A_theta[pos], marker='none', color=COLORS[2], linestyle=(0, (0, 1.5)), linewidth=2, label=r"$\mathfrak{a}^{-1}\,\rho^{\,}_{\theta}$", zorder=101)
	p, = ax[1].plot(x[pos], kappa_phi[pos], marker='none', color=COLORS[0], linewidth=2, label=r"$\mathfrak{a}^{-3}\,\kappa^{\,}_{\phi}$", zorder=101)
	p, = ax[1].plot(x[pos], u_phi[pos], marker='none', color=COLORS[1], linestyle=(0, (0, 1.5)), linewidth=2, label=r"$\mathfrak{a}^{-3}\,u^{\,}_{\phi}$", zorder=102)
	
	p, = ax[0].plot(x[neg], eta[neg], marker='none', color=COLORS[4], linewidth=2, label=r"$\eta$", zorder=101)
	p, = ax[0].plot(x[neg], v[neg], marker='none', color=COLORS[4], linestyle=(0, (0, 1.5)), linewidth=2, label=r"$\mathfrak{a}^{-1}\,v$", zorder=101)
	p, = ax[1].plot(x[pos], eta[pos], marker='none', color=COLORS[4], linewidth=2, label=r"$\eta$", zorder=101)
	p, = ax[1].plot(x[pos], v[pos], marker='none', color=COLORS[4], linestyle=(0, (0, 1.5)), linewidth=2, label=r"$\mathfrak{a}^{-1}\,v$", zorder=101)

	ax[0].axhline(0, color="k", linewidth=0.5, zorder=99)
	ax[1].axhline(0, color="k", linewidth=0.5, zorder=99)
	ax[0].axhline(1/2, color="k", linewidth=0.5, zorder=99)
	ax[1].axhline(1/2, color="k", linewidth=0.5, zorder=99)
	ax[0].axhline(1, color="k", linewidth=0.5, zorder=99)
	ax[1].axhline(1, color="k", linewidth=0.5, zorder=99)

	ax[0].axvline(0, color="k", linewidth=0.5, zorder=99)
	ax[1].axvline(0, color="k", linewidth=0.5, zorder=99)

	ax[0].set_title(r"$\lambda\leq0$", y=0, pad=-12, fontsize=12)
	ax[1].set_title(r"$\lambda\geq0$", y=0, pad=-12, fontsize=12)

	ax[0].set_xticks([0])
	ax[1].set_xticks([0])

	print(v[pos][0])

	ax[0].set_yticks([0, 0.5, 1])
	ax[1].set_yticks([0, 0.5, 1])
	ax[0].set_yticklabels(["$0$", r"$\frac{1}{2}$", "$1$"])
	ax[1].set_yticklabels([])

	ax[0].tick_params(
		axis="y",
		which="major",
		left=False,
		labelleft=False,
		right=True,
		labelright=True
	)

	ax[0].tick_params(axis='y', which='minor', left=False)
	ax[1].tick_params(axis='y', which='minor', left=False)

	ax[0].set_xlim(-0.75, 0)
	ax[1].set_xlim(0, 0.25)

	ax[0].set_ylim(0, 3.5)
	ax[1].set_ylim(0, 3.5)

	ax[1].legend(
		loc="lower right",
		ncol=2,
		handlelength=1.5,
		handletextpad=0.8,
		labelspacing=0.4,
		bbox_to_anchor=(0.7, 0.95)).set_zorder(1000)

	for spine in ax[0].spines.values():
		spine.set_zorder(999)
	for spine in ax[1].spines.values():
		spine.set_zorder(999)

	plt.draw()

	if args.save:
		save_graph(fig, "./", "couplings")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()
