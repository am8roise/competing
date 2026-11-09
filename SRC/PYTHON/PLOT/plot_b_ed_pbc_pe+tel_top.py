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
	simul_name = "PE+TEL/TOP"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	n_levels = 3
	num = 10
	eps = 1e-10
	delta_x = 0.4
	delta_y = 0

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

				fig, ax = plt.subplots(figsize=(8, 4))

				for K in sorted(EK.keys()):

					grouped = defaultdict(list)
					for energy in EK[K]:
						key = round(energy / eps)
						grouped[key].append(energy)

					for group in grouped.values():
						n = len(group)

						for j, energy in enumerate(group):

							x_shift = delta_x * (j - (n - 1) / 2) / n
							y_shift = delta_y * (j - (n - 1) / 2) / n
							
							p, = ax.plot(K + x_shift, energy / N + y_shift, marker=".", color=COLORS[0], markersize=5, zorder=110)
							p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				degeneracy_info = energies_degeneracies(EK, n=n_levels, tol=eps)

				x_offsets = sorted([j for j in range(-(n_levels - 1), n_levels, 2)], key=lambda x: -abs(x))

				energy_level = 0
				for j, (energy, deg, sectors) in enumerate(degeneracy_info):
					
					sector_counts = Counter(sectors)
					sector_str = ", ".join([fr"{k}^{{({count})}}" for k, count in sorted(sector_counts.items())])

					label = fr"$E^{{\,}}_{{{energy_level}}}$, $k\in\{{{sector_str}\}}$, $\mathrm{{deg.}}={deg}$"
					energy_level += deg

					y = energy / N 
					y_offset = 0.05 + 0.01 * j

					for (k, _) in sorted(sector_counts.items()):
						ax.annotate(label,
									xy=(k, y),
									xytext=(N // 2 + x_offsets[j], y + y_offset),
									textcoords='data',
									arrowprops=dict(arrowstyle="-", color=COLORS[(j + 4) % 8], lw=0.5),
									fontsize=9,
									bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS[(j + 4) % 8], alpha=0.8),
									ha="center",
									va="bottom",
									zorder=150 - j)
					
				ax.set_xticks([0, N//4, N//2, 3*N//4, N])
				ax.set_xticklabels([r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

				ticks = np.arange(N+2)
				for j in range(len(ticks)):
					ax.axvspan(ticks[j] - 0.5, ticks[j] + 0.5, facecolor=COLORS[1], edgecolor=None, alpha=0.08 if j % 2 == 0 else 0)


				ax.set_xlabel(r"$k$")
				ax.set_ylabel(r"$(2M)^{-1}\,E^{\,}_{k}$")

				ax.set_xlim(-0.5, N + 0.5)

				ax.minorticks_on()
				ax.tick_params(axis='x', which='minor', bottom=False)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			Ns.append(N)
			gaps_Ns.append(gaps)
			E0s.append(np.min(E))

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
	ax.plot(N, -2*np.sqrt(2)/np.tan(np.pi/N) / N, marker="none", color=COLORS[3], linestyle="-", linewidth=1.5, zorder=99, label=r"$\displaystyle -\frac{2\sqrt{2}}{2N}\left(\tan\frac{\pi}{2N}\right)^{-1}$")

	plt.xlabel(r"$2N$")
	plt.ylabel(r"$(2N)^{-1}\times E^{(0)}_{-}$")

	ax.set_ylim(-1.12, -0.68)

	ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([i for i in range(6, 31, 4)]))
	ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([i for i in range(8, 31, 4)]))

	ax.text(
		0.01, 0.97,
		"(b)",
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=12)

	ax.legend(
		loc="upper right",
		ncol=2,
		handlelength=1.5,
		handletextpad=1,
		bbox_to_anchor=(1.05, 1.05)).set_zorder(1000)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"E0_lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()
