import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import scipy.optimize as sco
from collections import defaultdict, Counter

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph, format_value
from tenet.utils.sty import COLORS
from tenet.utils.helpers import energies_aggregated, energies_aggregated_with_keys, energies_degeneracies, specific_heat_from_energies

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "PE+TEL/LEFT"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	n_levels = 4
	num = 20
	eps = 1e-10
	delta_x = 0.4
	delta_y = 0

	Ns_fit = [14, 8, 18, 8]

	Ns = []
	gaps_Ns = []

	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		if (N == N_one) or N_all:

			lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
			theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

			for i in range(len(data_filenames)):

				with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
					engine = pickle.load(file)

				if N not in [28, 30]:
					EK = engine["energies"]
					E = np.sort(np.concatenate(list(EK.values())))[:num]
				else:
					EM, EK, E = energies_aggregated(engine["energies"], num)
					E = np.array(E)

				gaps = E[1:] - E[:-1]

				# if N == 26:

				# 	T = np.linspace(0.1, 0.15, 100)

				# 	C = np.array([specific_heat_from_energies(E, t, N) for t in T])

				# 	print("T", T)
				# 	print("C", C)

				# 	fig, ax = plt.subplots(figsize=(8, 4))

				# 	def f(x, a, b):
				# 		return a * x + b

				# 	popt, pcov = sco.curve_fit(f, 1/T, np.log(C * T ** 2))
				# 	perr = np.sqrt(np.diag(pcov))

				# 	print(popt, perr)

				# 	plt.plot(T, C)
				# 	plt.show()

			Ns.append(N)
			gaps_Ns.append(gaps)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	gaps_Ns = np.array(gaps_Ns)[ind, :]

	colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[4]]

	fig, ax = plt.subplots(4, 2, gridspec_kw={"wspace": 0.02, "width_ratios": [2, 1]}, figsize=(5, 4))

	for i, e_ in enumerate([0, 1, 2, 3]):
		
		ind = np.where((Ns >= Ns_fit[i]))[0]

		if e_ in [0, 2, 3]:

			def f(N, a, b, c, d, e, f):
				return (a * 1/(N * np.log(N)) ** b) * (1 + c * np.cos(2 * np.pi * N / d + e)) + f

			def g(N, a, b, c, d):
				return (a * 1/(N * np.log(N)) ** b) * (1 + c) + d

			popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, e_], [1, 1, 1, 3, 1, 0])
			perr = np.sqrt(np.diag(pcov))			

		elif e_ in [1]:
			
			def f(N, a):
				return a * np.ones(len(N))

			def g(N, a):
				return a * np.ones(len(N))
			
			popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, e_], [0])
			perr = np.sqrt(np.diag(pcov))

		print(f"E{e_+1}-E{e_}", popt, perr)

		p, = ax[i, 0].plot(Ns, gaps_Ns[:, e_], marker="o", markerfacecolor=colors[i], lw=0, markersize=7, zorder=100-i)

		x = np.linspace(np.min(Ns[ind]), np.max(Ns)+10, 120)
		p, = ax[i, 0].plot(x, f(x, *popt), marker="none", color=colors[i], lw=0.5, alpha=1, zorder=99-i)

		x = np.linspace(np.min(Ns[ind]), np.max(Ns)+10, 120)
	
		if e_ in [0, 2, 3]:
	
			env = g(x, popt[0], popt[1], popt[2], popt[-1])
			env_ = g(x, popt[0], popt[1], -popt[2], popt[-1])

			ax[i, 0].fill_between(
				x,
				env_,
				env,
				facecolor=colors[i],
				alpha=0.2,
				linewidth=0,
				zorder=90
			)

		label = fr"$\mathrm{{fit}}^{{\,}}_{{E^{{\,}}_{{{e_+1}}}-E^{{\,}}_{{{e_}}}}}$"

		x = np.logspace(np.log10(np.max(Ns)), np.log10(1000), 1000)
		p, = ax[i, 1].plot(x, f(x, *popt), marker="none", color=colors[i], lw=0.5, alpha=1, label=label, zorder=99-i)

		x = np.logspace(np.log10(np.max(Ns)), np.log10(1000), 100)

		if e_ in [0, 2, 3]:

			env = g(x, popt[0], popt[1], popt[2], popt[-1])
			env_ = g(x, popt[0], popt[1], -popt[2], popt[-1])

			ax[i, 1].fill_between(
				x,
				env_,
				env,
				facecolor=colors[i],
				alpha=0.2,
				linewidth=0,
				zorder=90
			)

		ax[i, 1].set_ylabel(fr"$E^{{\,}}_{{{e_+1}}}-E^{{\,}}_{{{e_}}}$", rotation=-90, labelpad=16)
		ax[i, 1].yaxis.set_label_position("right")

		ax[i, 1].set_xscale("log")

		ax[i, 0].set_xlim(6, 32)
		ax[i, 1].set_xlim(32, np.max(x))

		ax[i, 0].set_yticks([])
		ax[i, 1].yaxis.tick_right()

		# ========== BROKEN X AXIS
		ax[i, 0].spines.right.set_visible(False)
		ax[i, 1].spines.left.set_visible(False)

		d = 0.05
		ax[i, 0].plot([1-d/4, 1+d/4], [-d, d], transform=ax[i, 0].transAxes, lw=0.5, color="k", marker="none", clip_on=False, zorder=1000)
		ax[i, 0].plot([1-d/4, 1+d/4], [1-d, 1+d], transform=ax[i, 0].transAxes, lw=0.5, color="k", marker="none", clip_on=False, zorder=1000)
		ax[i, 1].plot([-d/2, d/2], [-d, d], transform=ax[i, 1].transAxes, lw=0.5, color="k", marker="none", clip_on=False, zorder=1000)
		ax[i, 1].plot([-d/2, d/2], [1-d, 1+d], transform=ax[i, 1].transAxes, lw=0.5, color="k", marker="none", clip_on=False, zorder=1000)
		# ==========

		ax[i, 1].legend(
			loc="upper right",
			ncol=1,
			bbox_to_anchor=(0.9, 1.1)).set_zorder(1000)

		for spine in ax[i, 0].spines.values():
			spine.set_zorder(999)
		for spine in ax[i, 1].spines.values():
			spine.set_zorder(999)

	ax[3, 0].set_xlabel(r"$2N$", x=0.75)

	ax[0, 0].set_xticks([])
	ax[0, 1].set_xticks([])
	ax[1, 0].set_xticks([])
	ax[1, 1].set_xticks([])
	ax[2, 0].set_xticks([])
	ax[2, 1].set_xticks([])

	ax[0, 1].set_yticks([0, 0.5])
	ax[1, 1].set_yticks([0, 1])
	ax[2, 1].set_yticks([0, 0.5])
	ax[3, 1].set_yticks([0, 0.5, 1])
	ax[0, 1].set_yticklabels(["$0$", r"$\frac{1}{2}$"])
	ax[1, 1].set_yticklabels(["$0$", "$1$"])
	ax[2, 1].set_yticklabels(["$0$", r"$\frac{1}{2}$"])
	ax[3, 1].set_yticklabels(["$0$", r"$\frac{1}{2}$", "1"])

	ax[3, 0].xaxis.set_minor_locator(mpl.ticker.FixedLocator(Ns))
	ax[3, 0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

	ax[0, 1].xaxis.set_minor_locator(mpl.ticker.FixedLocator([]))
	ax[1, 1].xaxis.set_minor_locator(mpl.ticker.FixedLocator([]))
	ax[2, 1].xaxis.set_minor_locator(mpl.ticker.FixedLocator([]))

	ax[0, 0].set_ylim(-0.05, 0.55)
	ax[0, 1].set_ylim(-0.05, 0.55)
	ax[1, 0].set_ylim(-0.5, 1.1)
	ax[1, 1].set_ylim(-0.5, 1.1)
	ax[2, 0].set_ylim(-0.05, 0.55)
	ax[2, 1].set_ylim(-0.05, 0.55)

	ax[3, 0].set_ylim(-0.1, 1.1)
	ax[3, 1].set_ylim(-0.1, 1.1)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"gaps")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()
