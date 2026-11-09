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
from tenet.utils.sty import COLORS
from tenet.utils.math import r2
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
	simul_name = "PE+TEB/BOTTOM"

	PBCs = True

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
	E0s = []
	diffs_Ns = []

	ns = [0, 2, 8]
					
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

			EM, EK, E = energies_aggregated_with_keys(E, num)
			Ms = sorted(set(M for tup in EK.values() for (_, M) in tup))
			
			E = np.array(E)
			diffs[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = E[1:] - E[:-1]

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

			E = engine["energies"]

			EM, EK, E = energies_aggregated(E, num)
			
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
				y_offset = 0.3 + 0.2 * j

				for (m, _) in sorted(sector_counts.items()):
					ax.annotate(label,
								xy=(m - N // 2, y),
								xytext=(x_offsets[j], y + y_offset),
								textcoords='data',
								arrowprops=dict(arrowstyle="-", color=COLORS[(j + 4) % 8], lw=0.5),
								fontsize=9,
								bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS[(j + 4) % 8], alpha=0.8),
								ha="center",
								va="bottom",
								zorder=150 - j)
				
			def f(M, a, b, c):
				return a * np.abs(M) ** 2 + b * np.abs(M) ** 4 + c

			Ms = np.array(Ms)
			grounds = np.array(grounds)

			if N > 20:
				ind = np.where(np.abs(Ms) < N // 2 - 4)[0]

				Ms = Ms[ind]
				grounds = grounds[ind]

				popt, pcov = sco.curve_fit(f, Ms, grounds, [-1, 1, -1])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				ax.plot(Ms, f(Ms, *popt), linestyle='--', color=COLORS[3], zorder=99, label=fr"${format_sign(popt[0])}{format_value(np.abs(popt[0]), perr[0])}\cdot |m|^{{2}}{format_sign(popt[1])}{format_value(np.abs(popt[1]), perr[1])}\cdot |m|^{{4}}{format_sign(popt[2])}{format_value(np.abs(popt[2]), perr[2])}$")

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
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"mag_lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()
			
			Ns.append(N)
			E0s.append(np.min(E))

		diffs_Ns.append(diffs)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	E0s = np.array(E0s)[ind]
	diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

	print(Ns[ind], diffs_Ns[ind, 0, 0, :12])

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

	linestyles = ["-", "--", ":", ":", "-"]
	markers = ["x", ".", "+", "$/$", "."]
	markersizes = [5, 10, 7, 8]
	colors = [COLORS[0],COLORS[2],COLORS[3],COLORS[5],COLORS[6]]

	fig, ax = plt.subplots(figsize=(7, 2))

	handles = []
	s = 0

	for n in ns:
		
		def f(N, a, b):
			return a * np.log(N) + b

		ind = np.where(Ns >= 8)[0]

		p, = ax.plot(Ns[ind], diffs_Ns[ind, 0, 0, n], color=colors[s], marker=markers[s], markersize=markersizes[s], linewidth=0, markeredgewidth=2, zorder=99, label=fr"$E^{{\,}}_{{{n+1},{n}}}$")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		handles.insert(s, p)

		ind = np.where(Ns >= 22)[0]

		popt, pcov = sco.curve_fit(f, Ns[ind], np.log(diffs_Ns[ind, 0, 0, n]))
		perr = np.sqrt(np.diag(pcov))

		a = np.exp(popt[1])
		da = perr[1] * np.exp(popt[1])

		R2 = r2(np.log(diffs_Ns[ind, 0, 0, n]), f(Ns[ind], *popt))
		print("R2 power", R2)

		o, = ax.plot(Ns[ind], np.exp(f(Ns[ind], *popt)), color=p.get_color(), linestyle=linestyles[s], linewidth=1, markersize=0, zorder=98, label=fr"$({format_value(a, da)})\times (2N)^{{{format_value(popt[0], perr[0])}}}$")
		o.set_marker(mpl.markers.MarkerStyle(o.get_marker(), capstyle="round"))
		handles.append(o)
		s += 1

	ax.minorticks_on()

	ax.set_xscale("log")
	ax.set_yscale("log")

	plt.xlabel(r"$2N$")

	ax.legend(handles=handles, loc="lower center", ncol=2, labelspacing=0.6, columnspacing=1, handletextpad=0.5, handlelength=1.4, borderaxespad=0, bbox_to_anchor=(0.5, 1.03)).set_zorder(200)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"alg")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()
