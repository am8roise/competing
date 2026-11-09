import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as sco
from cycler import cycler
from collections import defaultdict, Counter

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph
from tenet.utils.helpers import energies_aggregated, energies_aggregated_with_keys, energies_degeneracies, specific_heat_from_energies
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
	simul_name = "PE+TLL/BOTTOM"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	num = 20

	n_levels = 4
	eps = 1e-10
	delta_x = 0.4
	delta_y = 0

	Ns_fit = [16, 12, 14, 10]

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

				if N not in [28, 30]:
					EK = engine["energies"]
					E = np.sort(np.concatenate(list(EK.values())))[:num]
				else:
					EM, EK, E = energies_aggregated(engine["energies"], num)
					E = np.array(E)

				gaps = E[1:] - E[:-1]

				# fig, ax = plt.subplots(figsize=(8, 4))

				# for K in sorted(EK.keys()):

				# 	grouped = defaultdict(list)
				# 	for energy in EK[K]:
				# 		key = round(energy / eps)
				# 		grouped[key].append(energy)

				# 	for group in grouped.values():
				# 		n = len(group)

				# 		for j, energy in enumerate(group):

				# 			x_shift = delta_x * (j - (n - 1) / 2) / n
				# 			y_shift = delta_y * (j - (n - 1) / 2) / n
							
				# 			p, = ax.plot(K + x_shift, energy / N + y_shift, marker=".", color=COLORS[0], markersize=5, zorder=110)
				# 			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				# degeneracy_info = energies_degeneracies(EK, n=n_levels, tol=eps)

				# x_offsets = sorted([j for j in range(-(n_levels - 1), n_levels, 2)], key=lambda x: -abs(x))

				# energy_level = 0
				# for j, (energy, deg, sectors) in enumerate(degeneracy_info):
					
				# 	sector_counts = Counter(sectors)
				# 	sector_str = ", ".join([fr"{k}^{{({count})}}" for k, count in sorted(sector_counts.items())])

				# 	label = fr"$E^{{\,}}_{{{energy_level}}}$, $k\in\{{{sector_str}\}}$, $\mathrm{{deg.}}={deg}$"
				# 	energy_level += deg

				# 	y = energy / N 
				# 	y_offset = 0.05 + 0.01 * j

				# 	for (k, _) in sorted(sector_counts.items()):
				# 		ax.annotate(label,
				# 					xy=(k, y),
				# 					xytext=(N // 2 + x_offsets[j], y + y_offset),
				# 					textcoords='data',
				# 					arrowprops=dict(arrowstyle="-", color=COLORS[(j + 4) % 8], lw=0.5),
				# 					fontsize=9,
				# 					bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS[(j + 4) % 8], alpha=0.8),
				# 					ha="center",
				# 					va="bottom",
				# 					zorder=150 - j)
					
				# ax.set_xticks([0, N//4, N//2, 3*N//4, N])
				# ax.set_xticklabels([r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

				# ticks = np.arange(N+2)
				# for j in range(len(ticks)):
				# 	ax.axvspan(ticks[j] - 0.5, ticks[j] + 0.5, facecolor=COLORS[1], edgecolor=None, alpha=0.08 if j % 2 == 0 else 0)

				# ax.set_xlabel(r"$k$")
				# ax.set_ylabel(r"$(2M)^{-1}\,E^{\,}_{k}$")

				# ax.set_xlim(-0.5, N + 0.5)

				# ax.minorticks_on()
				# ax.tick_params(axis='x', which='minor', bottom=False)

				# plt.draw()

				# if args.save:
				# 	save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}")
				# 	fig.clf()

				# if args.show:
				# 	plt.show()

				# plt.close()
			
			Ns.append(N)
			gaps_Ns.append(gaps)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	gaps_Ns = np.array(gaps_Ns)[ind, :]

	colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[4]]

	fig, ax = plt.subplots(4, 2, gridspec_kw={"wspace": 0.02, "width_ratios": [2, 1]}, figsize=(5, 4))

	for i, e_ in enumerate([0, 1,]):
		
		ind = np.where((Ns >= Ns_fit[i]))[0]

		if e_ in [0]:
		
			def f(N, a, b, c, d, e, f):
				return (a * 1 / N ** b) * (1 + c * np.cos((np.pi / 2 + d) * N + e)) + f
	
			def g(N, a, b, c, d):
				return (a * 1 / N ** b) * (1 + c) + d

			popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, e_], [10, 1, 1, 0.06, -1, 0])
			perr = np.sqrt(np.diag(pcov))

		elif e_ in [1]:
	
			def f(N, a, b, c, d, e, f, g):
				return (a * 1 / N ** b) + c * np.cos((np.pi / 2 + d) * N + e) / N ** f + g

			def g(N, a, b, c, d, e):
				return (a * 1 / N ** b) + (c * 1 / N ** d) + e

			popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, e_], [5, 1, -10, 0.1, 1, 1, 0])
			perr = np.sqrt(np.diag(pcov))
			
		elif e_ in [2]:
	
			def f(N, a, b, c, d, e, f, g):
				return (a * 1 / N ** b) + c * np.cos((np.pi / 2 + d) * N + e) / N ** f + g

			def g(N, a, b, c, d, e):
				return (a * 1 / N ** b) + (c * 1 / N ** d) + e

			popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, e_], [1, 1, -1, 0.06, 1, 1, 0])
			perr = np.sqrt(np.diag(pcov))

		elif e_ in [4]:
	
			def f(N, a, b, c, d, e, f, g):
				return (a * 1 / N ** b) + c * np.cos((np.pi / 2 + d) * N + e) / N ** f + g

			def g(N, a, b, c, d, e):
				return (a * 1 / N ** b) + (c * 1 / N ** d) + e

			popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, e_], [5, 1, -10, 0.1, -10, 1, -0.05])
			perr = np.sqrt(np.diag(pcov))

		print(f"E{e_+1}-E{e_}", popt, perr)

		p, = ax[i, 0].plot(Ns, gaps_Ns[:, e_], marker="o", markerfacecolor=colors[i], lw=0, markersize=7, zorder=100-i)

		x = np.linspace(np.min(Ns[ind]), np.max(Ns)+10, 120)
		p, = ax[i, 0].plot(x, f(x, *popt), marker="none", color=colors[i], lw=0.5, alpha=1, zorder=99-i)

		x = np.linspace(np.min(Ns[ind]), np.max(Ns)+10, 120)
		
		if e_ in [0]:
	
			env = g(x, popt[0], popt[1], popt[2], popt[-1])
			env_ = g(x, popt[0], popt[1], -popt[2], popt[-1])

		else:

			env = g(x, popt[0], popt[1], popt[2], popt[5], popt[-1])
			env_ = g(x, popt[0], popt[1], -popt[2], popt[5], popt[-1])

		ax[i, 0].fill_between(
			x,
			env_,
			env,
			facecolor=colors[i],
			alpha=0.2,
			linewidth=0,
			zorder=90
		)

		label = fr"$\mathrm{{fit}}^{{\,}}_{{E^{{\,}}_{{{e_+1}}}-E^{{\,}}_{{{e_}}}}}$"

		x = np.logspace(np.log10(np.max(Ns)), np.log10(1000), 1000)
		p, = ax[i, 1].plot(x, f(x, *popt), marker="none", color=colors[i], lw=0.5, alpha=1, label=label, zorder=99-i)

		x = np.logspace(np.log10(np.max(Ns)), np.log10(1000), 100)
	
		if e_ in [0]:
	
			env = g(x, popt[0], popt[1], popt[2], popt[-1])
			env_ = g(x, popt[0], popt[1], -popt[2], popt[-1])

		else:

			env = g(x, popt[0], popt[1], popt[2], popt[5], popt[-1])
			env_ = g(x, popt[0], popt[1], -popt[2], popt[5], popt[-1])

		ax[i, 1].fill_between(
			x,
			env_,
			env,
			facecolor=colors[i],
			alpha=0.2,
			linewidth=0,
			zorder=90
		)

		ax[i, 1].set_ylabel(fr"$E^{{\,}}_{{{e_+1}}}-E^{{\,}}_{{{e_}}}$", rotation=-90, labelpad=16)
		ax[i, 1].yaxis.set_label_position("right")

		ax[i, 1].set_xscale("log")

		ax[i, 0].set_xlim(6, 32)
		ax[i, 1].set_xlim(32, np.max(x))

		ax[i, 0].set_yticks([])
		ax[i, 1].yaxis.tick_right()

		# ========== BROKEN X AXIS
		ax[i, 0].spines.right.set_visible(False)
		ax[i, 1].spines.left.set_visible(False)

		d = 0.05
		ax[i, 0].plot([1-d/4, 1+d/4], [-d, d], transform=ax[i, 0].transAxes, lw=0.5, color="k", marker="none", clip_on=False, zorder=1000)
		ax[i, 0].plot([1-d/4, 1+d/4], [1-d, 1+d], transform=ax[i, 0].transAxes, lw=0.5, color="k", marker="none", clip_on=False, zorder=1000)
		ax[i, 1].plot([-d/2, d/2], [-d, d], transform=ax[i, 1].transAxes, lw=0.5, color="k", marker="none", clip_on=False, zorder=1000)
		ax[i, 1].plot([-d/2, d/2], [1-d, 1+d], transform=ax[i, 1].transAxes, lw=0.5, color="k", marker="none", clip_on=False, zorder=1000)
		# ==========

		ax[i, 1].legend(
			loc="upper right",
			ncol=1,
			bbox_to_anchor=(0.9, 1.1)).set_zorder(1000)

		for spine in ax[i, 0].spines.values():
			spine.set_zorder(999)
		for spine in ax[i, 1].spines.values():
			spine.set_zorder(999)

	ax[3, 0].set_xlabel(r"$2N$", x=0.75)

	ax[0, 0].set_xticks([])
	ax[0, 1].set_xticks([])
	ax[1, 0].set_xticks([])
	ax[1, 1].set_xticks([])
	ax[2, 0].set_xticks([])
	ax[2, 1].set_xticks([])

	ax[0, 1].set_yticks([0, 1, 2])
	ax[1, 1].set_yticks([0, 1, 2])
	ax[2, 1].set_yticks([0, 1/3])
	ax[3, 1].set_yticks([0, 0.2])
	ax[0, 1].set_yticklabels(["$0$", "$1$", "$2$"])
	ax[1, 1].set_yticklabels(["$0$", "$1$", "$2$"])
	ax[2, 1].set_yticklabels(["$0$", r"$\frac{1}{3}$"])
	ax[3, 1].set_yticklabels(["$0$", r"$\frac{1}{5}$"])

	ax[3, 0].xaxis.set_minor_locator(mpl.ticker.FixedLocator(Ns))
	ax[3, 0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

	ax[0, 1].xaxis.set_minor_locator(mpl.ticker.FixedLocator([]))
	ax[1, 1].xaxis.set_minor_locator(mpl.ticker.FixedLocator([]))
	ax[2, 1].xaxis.set_minor_locator(mpl.ticker.FixedLocator([]))

	ax[0, 0].set_ylim(-0.5, 2.5)
	ax[0, 1].set_ylim(-0.5, 2.5)
	ax[1, 0].set_ylim(-0.9, 2.5)
	ax[1, 1].set_ylim(-0.9, 2.5)
	ax[2, 0].set_ylim(-0.1, 0.4)
	ax[2, 1].set_ylim(-0.1, 0.4)

	ax[3, 0].set_ylim(-0.1, 0.25)
	ax[3, 1].set_ylim(-0.1, 0.25)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"gaps")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

	fig, ax = plt.subplots(figsize=(5, 4))

	ind = np.where((Ns >= 8))[0]

	def f(N, a, b, c, d, e, f):
		return (a * 1 / N ** b) * (1 + c * np.cos((np.pi / 2 + d) * N + e)) + f
	def g(N, a, b, c, d, e, f):
		return (a + b * np.cos((np.pi / 2 + c) + d)) * np.exp(-N / e) + f
	
	popt, pcov = sco.curve_fit(f, Ns[ind], gaps_Ns[ind, 0], [10, 1, 1, 0.12, -1, 0])
	perr = np.sqrt(np.diag(pcov))

	print(f"alg", popt, perr)

	# x = np.linspace(np.min(Ns[ind]), np.max(Ns)+10, 120)
	# p, = ax.plot(x, f(x, *popt), marker="none", color=colors[i], lw=0.5, alpha=1, zorder=99-i)

	popt, pcov = sco.curve_fit(g, Ns[ind], np.log(gaps_Ns[ind, 0]), [1, 1, 0.12, -1, 1, 0])
	perr = np.sqrt(np.diag(pcov))

	print(f"exp", popt, perr)

	x = np.linspace(np.min(Ns[ind]), np.max(Ns)+10, 120)
	p, = ax.plot(x, np.exp(g(x, *popt)), marker="none", color=colors[i], lw=0.5, alpha=1, zorder=99-i)

	p, = ax.plot(Ns, gaps_Ns[:, 0], marker="o", markerfacecolor=colors[i], lw=0, markersize=7, zorder=100-i)

	ax.set_yscale("log")

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"E1-E0")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()
