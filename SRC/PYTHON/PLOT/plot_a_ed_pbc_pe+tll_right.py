import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

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
	model_name = "A"
	simul_name = "PE+TLL/RIGHT"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True
					
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


if __name__ == "__main__":
	main()
