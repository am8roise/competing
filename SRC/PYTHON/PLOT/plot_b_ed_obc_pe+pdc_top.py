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

	PBCs = False

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	num = 30
	n_levels = 3
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

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			E = engine["energies"]

			EM, EK, E = energies_aggregated(E, num)

			E = np.array(E)
			gaps = E[1:] - E[:-1]

			fig, ax = plt.subplots(figsize=(8, 4))

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
						
						p, = ax.plot(M - N // 2 + x_shift, energy / N + y_shift, marker=".", color=COLORS[0], markersize=5, zorder=110)
						p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			degeneracy_info = energies_degeneracies(EM, n=n_levels, tol=eps)

			x_offsets = sorted([j for j in range(-(n_levels - 1), n_levels, 2)], key=lambda x: -abs(x))

			energy_level = 0
			for j, (energy, deg, sectors) in enumerate(degeneracy_info):
				
				sector_counts = Counter(sectors)
				sector_str = ", ".join([fr"{m - N // 2}^{{({count})}}" for m, count in sorted(sector_counts.items())])

				label = fr"$E^{{\,}}_{{{energy_level}}}$, $m\in\{{{sector_str}\}}$, $\mathrm{{deg.}}={deg}$"
				energy_level += deg

				y = energy / N + 0.02
				y_offset = 0.3 + 0.1 * j

				for (m, _) in sorted(sector_counts.items()):
					ax.annotate(label,
								xy=(m - N // 2, y),
								xytext=(x_offsets[j], y + y_offset),
								textcoords='data',
								arrowprops=dict(arrowstyle="-", color=COLORS[7], lw=0.5),
								fontsize=9,
								bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS[7], alpha=0.8),
								ha="center",
								va="bottom",
								zorder=150 - j)
				
			def f(M, a, b, c):
				return a * np.abs(M) ** b + c

			Ms = np.array(Ms)
			grounds = np.array(grounds)

			if N > 10:
				ind = np.where(np.abs(Ms) < N // 2 - 2)[0]

				Ms = Ms[ind]
				grounds = grounds[ind]

			popt, pcov = sco.curve_fit(f, Ms, grounds, [1, 2, -1])
			perr = np.sqrt(np.diag(pcov))

			print(popt, perr)

			ax.plot(Ms, f(Ms, *popt), linestyle='--', color=COLORS[3], zorder=99, label=fr"$({format_value(popt[0], perr[0])})\cdot |m|^{{{format_value(popt[1], perr[1])}}}{format_sign(popt[2])}({format_value(np.abs(popt[2]), perr[2])})$")

			ax.set_xticks([-N//2, -N//4, 0, N//4, N//2])
			ax.set_xticklabels([r"$-\frac{1}{2}\,2N$", r"$-\frac{1}{4}\,2N$", r"0", r"$\frac{1}{4}\,2N$", r"$\frac{1}{2}\,2N$"])

			ticks = np.arange(-N // 2, N // 2 + 1)
			for j in range(len(ticks)):
				ax.axvspan(ticks[j] - 0.5, ticks[j] + 0.5, facecolor=COLORS[1], edgecolor=None, alpha=0.08 if j % 2 == 0 else 0)

			ax.set_xlabel(r"$m$")
			ax.set_ylabel(r"$(2N)^{-1}\,E^{\,}_{m}$")

			ax.set_xlim(-N // 2 - 0.5, N // 2 + 0.5)

			ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))	

			ax.minorticks_on()
			ax.tick_params(axis='x', which='minor', bottom=False)

			ax.legend(loc="upper center", ncol=1, labelspacing=0.8).set_zorder(200)

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
	E0s = np.array(E0s)[ind]
	gaps_Ns = np.array(gaps_Ns)[ind, :]

	for e in [0]:

		fig, ax = plt.subplots(figsize=(8, 3))
		
		p, = ax.plot(Ns, np.log(gaps_Ns[:, e]), alpha=1, marker="x", color=COLORS[0], linewidth=0, markeredgewidth=1.5, markersize=10, zorder=100)
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		
		ax.set_xticks(Ns)

		def f(N, a, b):
			return a * N + b

		ind = np.where(Ns > 6)[0]

		popt, pcov = sco.curve_fit(f, Ns[ind], np.log(gaps_Ns[ind, e]))
		perr = np.sqrt(np.diag(pcov))

		a = np.exp(popt[1])
		da = perr[1] * np.exp(popt[1])

		b = 1 / popt[0]
		db = perr[0] * b ** 2
		
		R2 = r2(np.log(gaps_Ns[ind, e]), f(Ns[ind], *popt))
 
		p, = ax.plot(Ns[ind], f(Ns[ind], *popt), linestyle='--', color=COLORS[2], linewidth=2, zorder=99, marker="none", label=fr"$E^{{\,}}_{{{e+1},{e}}} = ({format_value(a, db)}) \cdot e^{{-\frac{{2N}}{{{format_value(np.abs(b), db)}}}}}$")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		ax.minorticks_on()

		plt.xlabel(r"$2N$")
		plt.ylabel(fr"$\ln\left(E^{{\,}}_{{{e+1}}}-E^{{\,}}_{{{e}}}\right)$")

		ax.legend(loc="upper right", ncol=1, labelspacing=0.8, columnspacing=2, handletextpad=0.8, handlelength=1.6).set_zorder(200)

		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}_E{e+1}-E{e}")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

	fig, ax = plt.subplots(figsize=(8, 3))

	p = ax.plot(Ns, E0s/Ns, alpha=1, marker=".", color=COLORS[0], markersize=10, linestyle=":", linewidth=1, zorder=100)

	ax.minorticks_on()

	plt.xlabel(r"$2N$")
	plt.ylabel(r"$E^{\,}_{0}$")

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"E0_lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()
