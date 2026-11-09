import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import scipy.optimize as sco

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph
from tenet.utils.sty import COLORS

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "PE+TML/TOP"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	n_E = 6

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

				E = engine["energies"]

				fig, ax = plt.subplots(figsize=(8, 6.4))

				for k in E.keys():
				
					momentum = k * 2 * np.pi / N
					
					p = ax.plot([momentum for _ in range(len(E[k]))], E[k], alpha=1, marker=".", color=COLORS[0], markeredgewidth=1.4, markersize=6, linewidth=0, zorder=100)
					[p[s].set_marker(mpl.markers.MarkerStyle(p[s].get_marker(), capstyle="round")) for s in range(len(p))]

				ax.minorticks_on()

				ax.set_xticks([0, np.pi, 2*np.pi])
				ax.set_xticklabels([r"0", r"$\pi$", r"$2\pi$"])

				plt.xlabel(r"$k$")
				plt.ylabel(r"$E^{\,}_{k}$")

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

				E_flat = []
				for E_sector in E.values():
					E_flat = np.append(E_flat, E_sector)
				E_flat = np.sort(E_flat)[:n_E]

				gaps = E_flat[1:] - E_flat[:-1]

			Ns.append(N)
			gaps_Ns.append(gaps)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	gaps_Ns = np.array(gaps_Ns)[ind, :]

	for e in [3]:

		fig, ax = plt.subplots(figsize=(8, 3))
		
		ind = np.where(Ns % 4 == 0)[0]

		p, = ax.plot(1 / Ns[ind], gaps_Ns[ind, e], alpha=1, marker="x", color=COLORS[0], linewidth=0, markeredgewidth=1.5, markersize=10, zorder=100, label=r"$2N\equiv 0\ \mathrm{mod}\ 4$")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		def f(N, a, b):
			return a*1/N + b

		ind = np.where((Ns % 4 == 0) & (Ns > 4))[0]

		popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, e])
		perr = np.sqrt(np.diag(pcov))

		p, = ax.plot(1 / Ns[ind], f(Ns[ind], *popt), linestyle='--', color=COLORS[2], linewidth=2, zorder=99, marker="none", label=fr"$v^{{(0\ \mathrm{{mod}}\ 4)}}_{{{e+1},{e}}} = {popt[0]:.1f} \pm {perr[0]:.1f}$")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		
		ind = np.where(Ns % 4 == 2)[0]

		p, = ax.plot(1 / Ns[ind], gaps_Ns[ind, e], alpha=1, marker="+", color=COLORS[3], linewidth=0, markeredgewidth=1.5, markersize=12, zorder=100, label=r"$2N\equiv 2\ \mathrm{mod}\ 4$")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		
		ind = np.where((Ns % 4 == 2) & (Ns > 6))[0]

		popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, e])
		perr = np.sqrt(np.diag(pcov))

		p, = ax.plot(1 / Ns[ind], f(Ns[ind], *popt), linestyle=':', color=COLORS[5], linewidth=2, zorder=99, marker="none", label=fr"$v^{{(2\ \mathrm{{mod}}\ 4)}}_{{{e+1},{e}}} = {popt[0]:.1f} \pm {perr[0]:.1f}$")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		ax.minorticks_on()

		plt.xlabel(r"$(2N)^{-1}$")
		plt.ylabel(fr"$E^{{\,}}_{{{e+1}}}-E^{{\,}}_{{{e}}}$")

		ax.legend(loc="center right", ncol=1, labelspacing=0.8, columnspacing=2, handletextpad=0.8, handlelength=1.6).set_zorder(200)

		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}_E{e+1}-E{e}")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

if __name__ == "__main__":
	main()
