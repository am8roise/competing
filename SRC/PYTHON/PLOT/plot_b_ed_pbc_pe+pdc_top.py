import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from collections import defaultdict
import scipy.optimize as sco

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph, format_value
from tenet.utils.sty import COLORS
from tenet.utils.helpers import energies_aggregated, energies_aggregated_with_keys
from tenet.utils.math import r2

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "PE+PDC/TOP"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	num = 30
	eps = 1e-10
	delta_x = 0.4
	delta_y = 0

	Ns = []
	gaps_Ns = []

	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			E = engine["energies"]

			EM, EK, E = energies_aggregated_with_keys(E, num)
			Ms = sorted(set(M for tup in EK.values() for (_, M) in tup))

			E = np.array(E)
			gaps = E[1:] - E[:-1]

			if N == 28:

				print(gaps)

			Ns.append(N)
			gaps_Ns.append(gaps)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	gaps_Ns = np.array(gaps_Ns)[ind, :]

	fig, ax = plt.subplots(1, 2, figsize=(4, 2))

	axin = ax[0].inset_axes([0.44, 0.41, 0.48, 0.41])
	axin.set_zorder(110)

	for spine in ax[0].spines.values():
		spine.set_zorder(1000)
	for spine in ax[1].spines.values():
		spine.set_zorder(1000)
	for spine in axin.spines.values():
		spine.set_zorder(1000)

	handles = []

	ind = np.where(Ns % 4 == 0)[0]

	p, = ax[0].plot(Ns[ind], gaps_Ns[ind, 0], marker="o", markerfacecolor=COLORS[2], linewidth=0, markersize=7, zorder=101, label=r"$\Delta E^{{(0)}}_{{1,0}}$")
	handles.append(p)

	p, = axin.plot(Ns[ind], gaps_Ns[ind, 0], marker="o", markerfacecolor=COLORS[2], linewidth=0, markersize=5, zorder=111, label=r"$\Delta E^{{(0)}}_{{1,0}}$")

	ind = np.where(Ns % 4 == 2)[0]

	p, = ax[0].plot(Ns[ind], gaps_Ns[ind, 0], marker="o", markerfacecolor=COLORS[3], linewidth=0, markersize=7, zorder=102, label=r"$\Delta E^{{(2)}}_{{1,0}}$")
	handles.append(p)

	p, = ax[1].plot(Ns, gaps_Ns[:, 1], marker="o", markerfacecolor=COLORS[0], linewidth=0, markersize=7, zorder=101, label=r"$\Delta E^{{\,}}_{{2,1}}$")
	handles.append(p)

	def f(N, a, b):
		return a * N + b

	def g(N, a, b):
		return a / N * (-1) ** ((N % 4) // 2) + b

	ind = np.where((Ns % 4 == 0) & (Ns >= 16))[0]

	popt, pcov = sco.curve_fit(f, Ns[ind], np.log(gaps_Ns[ind, 0]))
	perr = np.sqrt(np.diag(pcov))

	a = 1 / popt[0]
	da = perr[0] * a ** 2

	print("1-0", a, da)

	ind = np.where((Ns >= 18))[0]

	popt, pcov = sco.curve_fit(g, Ns[ind], gaps_Ns[ind, 1])
	perr = np.sqrt(np.diag(pcov))

	b = popt[1]
	db = perr[1]

	print("2-1", b, db)

	x = np.linspace(np.min(Ns), np.max(Ns), 100)

	_env = -popt[0] / x + popt[1]
	env = popt[0] / x + popt[1]

	ax[1].fill_between(
		x,
		_env,
		env,
		facecolor=COLORS[4],
		alpha=0.2,
		linewidth=0,
		zorder=91)

	p, = ax[1].plot(x, _env, linestyle='-', color=COLORS[4], linewidth=1, zorder=92, marker="none", label=r"$\Delta^{{\,}}_{{2,1}}\pm A^{{\,}}_{{2,1}}(2N)^{-1}$")
	handles.append(p)
	p, = ax[1].plot(x, env, linestyle='-', color=COLORS[4], linewidth=1, zorder=92, marker="none")

	line = np.ones(len(x)) * popt[1]

	p, = ax[1].plot(x, line, linestyle=':', color=COLORS[4], linewidth=1, zorder=92, marker="none", label=fr"$\Delta^{{\,}}_{{2,1}}$")
	handles.append(p)

	ax[0].set_xlabel(r"$2N$", fontsize=10)
	ax[1].set_xlabel(r"$2N$", fontsize=10)

	ax[0].set_xlim([4, 32])
	ax[1].set_xlim([4, 32])

	ax[0].set_yticks([0, 1])
	ax[1].set_yticks([1, 2])

	axin.set_yscale("log")
	
	axin.tick_params(labelsize=10)

	ax[0].text(
		0.97, 0.97,
		"(a)",
		transform=ax[0].transAxes,
		ha="right",
		va="top",
		fontsize=12)

	ax[1].text(
		0.97, 0.97,
		"(b)",
		transform=ax[1].transAxes,
		ha="right",
		va="top",
		fontsize=12)

	ax[1].legend(
		handles=handles,
		loc="lower left",
		ncol=2,
		handletextpad=0.4,
		borderpad=0.3,
		labelspacing=0.6,
		bbox_to_anchor=(0.3, 0.05)).set_zorder(1000)	

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"gap")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()


