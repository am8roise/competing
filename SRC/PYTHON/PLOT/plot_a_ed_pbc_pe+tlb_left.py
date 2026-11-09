import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from collections import defaultdict

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph
from tenet.utils.sty import COLORS
from tenet.utils.helpers import energies_aggregated, energies_aggregated_with_keys

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "PE+TLB/LEFT"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	num = 30
	eps = 1e-10
	delta_x = 0.4
	delta_y = 0
					
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

			EM, EK = energies_aggregated_with_keys(E, num)
			Ms = sorted(set(M for tup in EK.values() for (_, M) in tup))

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=max(Ms)-min(Ms)+1)
			markers = ["|", "."]

			norm = mpl.colors.Normalize(vmin=min(Ms)-0.5, vmax=max(Ms)+0.5)

			fig, ax = plt.subplots(figsize=(8, 4))

			for K in sorted(EK.keys()):

				# K = 2 * np.pi * K / N
				
				grouped = defaultdict(list)
				for energy, M in EK[K]:
					key = round(energy / eps)
					grouped[key].append((energy, M))

				for group in grouped.values():
					n = len(group)

					for j, (energy, M) in enumerate(group):

						x_shift = delta_x * (j - (n - 1) / 2) / n
						y_shift = delta_y * (j - (n - 1) / 2) / n
						
						p, = ax.plot(K + x_shift, energy / N + y_shift, marker=markers[M % len(markers)], color=cmap(norm(M)), markersize=5)
						p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
			sm.set_array([])

			cb = plt.colorbar(sm, ax=ax, ticks=[], location='top', aspect=30)
			cb.set_label("$m$", labelpad=10)

			cb.ax.tick_params(size=0)
			cb.set_ticks(np.arange(min(Ms), max(Ms) + 1))
			cb.ax.set_xticklabels([fr"{int(i)}" for i in range(min(Ms), max(Ms)+1)], fontsize=14)

			for M in Ms:
				pos = cb.ax.get_xlim()[0] + norm(M) * (cb.ax.get_xlim()[1] - cb.ax.get_xlim()[0])

				p, = cb.ax.plot([pos], [0.5], marker=markers[M % len(markers)], color="k", transform=cb.ax.transData, clip_on=False)
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			ax.set_xticks([0, N//4, N//2, 3*N//4, N])
			ax.set_xticklabels([r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

			ticks = np.arange(N+2)
			for j in range(len(ticks)):
				ax.axvspan(ticks[j] - 0.5, ticks[j] + 0.5, facecolor=COLORS[1], edgecolor=None, alpha=0.08 if j % 2 == 0 else 0)

			ax.set_xlabel(r"$k$")
			ax.set_ylabel(r"$(2N)^{-1}\,E^{\,}_{m, k}$")

			ax.set_xlim(-0.5, N + 0.5)

			ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))	

			ax.minorticks_on()
			ax.tick_params(axis='x', which='minor', bottom=False)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

if __name__ == "__main__":
	main()
