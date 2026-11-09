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
	model_name = "B"
	simul_name = "PE+TEL/BOTTOM"

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

	Ns_fit = [8, 14, 16, 16]

	Ns = []
	gaps_Ns = []
	E0s = []

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

				if N == 28:

					print(gaps)

			Ns.append(N)
			gaps_Ns.append(gaps)
			E0s.append(E[0])

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	gaps_Ns = np.array(gaps_Ns)[ind, :]
	E0s = np.array(E0s)[ind]

	print(Ns)
	print(E0s)

	fig, ax = plt.subplots(figsize=(4.8, 2))

	p = ax.plot(Ns, E0s/Ns, marker="o", color=COLORS[0], markersize=8, linestyle=(0, (0, 2)), linewidth=1, zorder=100)

	N = np.linspace(6, 30, 100)
	
	ax.plot(N, -4*np.sqrt(2)/np.sin(2*np.pi/N) / N, marker="none", color=COLORS[2], linestyle=(0, (3, 3)), linewidth=1.5, zorder=99, label=r"$\displaystyle -\frac{4\sqrt{2}}{2N}\left(\sin\frac{2\pi}{2N}\right)^{-1}$")
	ax.plot(N, -2*np.sqrt(2)/np.sin(np.pi/N) / N, marker="none", color=COLORS[3], linestyle="-", linewidth=1.5, zorder=99, label=r"$\displaystyle -\frac{2\sqrt{2}}{2N}\left(\sin\frac{\pi}{2N}\right)^{-1}$")

	plt.xlabel(r"$2N$")
	plt.ylabel(r"$(2N)^{-1}\times E^{(0)}_{+}$")

	ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([i for i in range(6, 31, 4)]))
	ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([i for i in range(8, 31, 4)]))

	ax.text(
		0.01, 0.97,
		"(a)",
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=12)

	ax.legend(
		loc="lower right",
		ncol=2,
		handlelength=1.5,
		handletextpad=1,
		bbox_to_anchor=(1.05, 0.1)).set_zorder(1000)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"E0_lam={np.round(lam[0], 3)}_theta={np.round(theta[0], 3)}")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()
if __name__ == "__main__":
	main()
