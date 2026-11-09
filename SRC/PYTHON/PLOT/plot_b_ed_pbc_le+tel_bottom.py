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
	model_name = "B"
	simul_name = "LE+TEL/BOTTOM"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	k = 5

	Ns = []
	diffs_Ns = []
			
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		if (N == N_one) or N_all:

			lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
			theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

			Ns.append(N)

			diffs = np.zeros((len(lam), len(theta), k-1))

			for i in range(len(data_filenames)):

				with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
					engine = pickle.load(file)

				E = engine["energies"]

				E_flat = []
				for E_sector in E.values():
					E_flat = np.append(E_flat, E_sector)
				E_flat = np.sort(E_flat)[:k]

				gaps = E_flat[1:] - E_flat[:-1]

				diffs[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = gaps
					
			diffs_Ns.append(diffs)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

	cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[7],COLORS[0],COLORS[1],COLORS[2],COLORS[3],COLORS[5]], N=256)

	plt.rcParams.update({'axes.prop_cycle': cycler(marker=([".", "|", "+"]*len(Ns))[:len(Ns)], color=cmap(np.linspace(0, 1, len(Ns))))})

	for t in range(len(theta)):

		fig, ax = plt.subplots(1, k-1, figsize=(4*(k-1)-1, 4))

		for i in range(k-1):

			for n in range(len(Ns)):
				p = ax[i].plot(lam, Ns[n]*diffs_Ns[n, :, t, i], alpha=1, linestyle=":", markeredgewidth=1.4, markersize=6, linewidth=0.6, zorder=100, label=fr"$2N = {Ns[n]}$")
				[p[s].set_marker(mpl.markers.MarkerStyle(p[s].get_marker(), capstyle="round")) for s in range(len(p))]

			ax[i].minorticks_on()
			ax[i].set_xlabel(r"$\theta$")
			if i==2 or i==3:
				ax[i].set_title(fr"$2N(E^{{\,}}_{{{i+1}}} - E^{{\,}}_{{{i}}})$", loc="right", pad=10, fontsize=18)
			else:
				ax[i].set_title(fr"$2N(E^{{\,}}_{{{i+1}}} - E^{{\,}}_{{{i}}})$", pad=10, fontsize=18)

			ax[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto", integer=True))
			
			ax[i].axvline(np.pi/4, color="k", linestyle="--", linewidth=1)

		ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, ncol=1, labelspacing=0.8).set_zorder(200)

		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"theta={np.round(theta[t], 3)}")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

if __name__ == "__main__":
	main()
