import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from collections import defaultdict, Counter
import scipy.optimize as sco

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.sty import COLORS
from tenet.utils.io import folder_names, load_file, save_graph, format_value, format_sign
from tenet.utils.helpers import energies_aggregated, energies_aggregated_with_keys, energies_degeneracies

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "PE+TET/BOTTOM"

	PBCs = False

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	num = 30
	n_levels = 1
	eps = 1e-10
	delta_x = 0.4
	delta_y = 0

	Ns = []
	diffs_Ns = []
				
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
		diffs = np.zeros((len(lam), len(theta), num - 1))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			E = engine["energies"]

			EM, EK, E = energies_aggregated(E, num)
			E = np.array(E)

			diffs[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = E[1:] - E[:-1]

			if N == 30:
				
				fig, ax = plt.subplots(1, 2,
					gridspec_kw=dict(width_ratios=[1, 2]),
					figsize=(6, 2.5))
				plt.subplots_adjust(wspace=0.33)

				handles = []

				Ms = []
				grounds = []
				for M in sorted(EM.keys()):

					Ms.append(M - N // 2)
					grounds.append(EM[M][0] / N)

					grouped = defaultdict(list)
					for energy in EM[M]:
						key = round(energy / eps)
						grouped[key].append(energy)

					for group in grouped.values():
						n = len(group)

						for j, energy in enumerate(group):

							x_shift = delta_x * (j - (n - 1) / 2) / n
							y_shift = delta_y * (j - (n - 1) / 2) / n
							
							p, = ax[1].plot(M - N // 2 + x_shift, energy / N + y_shift, marker="o", markerfacecolor=COLORS[0], lw=0, markersize=4, label=r"$(2N)^{-1}\,E^{\,}_{m,n}$", zorder=110)

				handles.append(p)

				degeneracy_info = energies_degeneracies(EM, n=n_levels, tol=eps)

				x_offsets = sorted([j for j in range(-(n_levels - 1), n_levels, 2)], key=lambda x: -abs(x))

				energy_level = 0
				for j, (energy, deg, sectors) in enumerate(degeneracy_info):
					
					sector_counts = Counter(sectors)
					sector_str = ", ".join([fr"{m - N // 2}^{{({count})}}" for m, count in sorted(sector_counts.items())])

					label = fr"$E^{{\,}}_{{{energy_level}}}$,\quad$m\in\bigl\{{{sector_str}\bigr\}}$"
					energy_level += deg

					y = energy / N + 0.05
					y_offset = 0.58

					for (m, _) in sorted(sector_counts.items()):
						ax[1].annotate(label,
							xy=(m - N // 2, y),
							xytext=(x_offsets[j], y + y_offset),
							textcoords='data',
							arrowprops=dict(
								arrowstyle="-",
								color="0",
								lw=0.5),
							fontsize=10,
							color="0",
							bbox=dict(
								boxstyle="square, pad=0.4",
								mutation_aspect=1.8,
								ec="0",
								fc=COLORS[-1],
								lw=0.5,
								alpha=1),
							ha="center",
							va="bottom",
							zorder=1000 - j)
					
				def f(M, a, b, c):
					return a * np.abs(M) ** b + c

				Ms = np.array(Ms)
				grounds = np.array(grounds)

				if N > 10:
					ind = np.where(np.abs(Ms) < N // 2)[0]

					Ms_ = Ms[ind]
					grounds_ = grounds[ind]

				popt, pcov = sco.curve_fit(f, Ms_, grounds_, [1, 2, -1])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = ax[1].plot(Ms, f(Ms, *popt), linestyle='-', lw=1.5, color=COLORS[4], zorder=100, label=r"$\mathrm{fit}^{\,}_{m}$")
				handles.append(p)

				ticks = np.arange(-N // 2, N // 2 + 1)
				for j in range(len(ticks)):
					ax[1].axvspan(ticks[j] - 0.5, ticks[j] + 0.5, facecolor=COLORS[0], edgecolor=None, alpha=0.08 if j % 2 == 0 else 0)

				ax[1].set_xlabel(r"$m$")

				ax[1].set_xlim(-N // 2 - 0.5, N // 2 + 0.5)
				ax[1].set_ylim(-0.9, 0.2)

				ax[1].set_xticks([-N/2, -N/4, 0, N/4, N/2])
				ax[1].set_xticklabels([r"$-\frac{2N}{2}$", r"$-\frac{2N}{4}$", r"0", r"$\frac{2N}{4}$", r"$\frac{2N}{2}$"])
				ax[1].tick_params(axis='x', which='minor', bottom=False)

				ax[1].set_yticks([0, -0.4, -0.8])
				ax[1].set_yticklabels(["$0$", "$-0.4$", "$-0.8$"])

				ax[1].legend(
					handles=handles,
					loc="upper center",
					ncol=1,
					bbox_to_anchor=(0.5, 1.05)).set_zorder(1001)

				ax[1].text(
					0.97, 0.97,
					"(b)",
					transform=ax[1].transAxes,
					ha="right",
					va="top",
					fontsize=12)
				
			Ns.append(N)

		diffs_Ns.append(diffs)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

	markersizes = [12, 8, 4, 12]
	colors = [COLORS[0],COLORS[2],COLORS[3],COLORS[4]]
	linestyles = ["-", (1, (2, 2)), (0, (0, 2)), "-"]

	handles = []

	for e in [0, 1, 2, 3]:
		
		p, = ax[0].plot(Ns, Ns * diffs_Ns[:, 0, 0, e], marker="o", markerfacecolor=colors[e], lw=0, markersize=markersizes[e], label=fr"$2N\,(E^{{\,}}_{{{e+1}}}-E^{{\,}}_{{{e}}})$", zorder=100+e)
		handles.append(p)

		def f(N, a, b):
			return a * 1 / N + b

		ind = np.where((Ns >= 16))[0]

		popt, pcov = sco.curve_fit(f, Ns[ind], diffs_Ns[ind, 0, 0, e])
		perr = np.sqrt(np.diag(pcov))

		print(e, popt, perr)
	
	ax[0].set_xlabel(r"$2N$")

	ax[0].dataLim.y0 = 0
	ax[0].dataLim.y1 = 5
	ax[0].autoscale_view()

	ax[0].set_xticks([10, 20, 30])
	ax[0].set_yticks([0, 2, 4, 6])

	ax[0].text(
		0.97, 0.97,
		"(a)",
		transform=ax[0].transAxes,
		ha="right",
		va="top",
		fontsize=12)

	ax[0].legend(
		handles=handles,
		loc="center right",
		ncol=1,
		bbox_to_anchor=(1.12, 0.44)).set_zorder(1000)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"gap")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()
	
if __name__ == "__main__":
	main()
