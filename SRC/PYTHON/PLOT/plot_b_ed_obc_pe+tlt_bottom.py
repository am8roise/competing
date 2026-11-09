import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from collections import defaultdict
import scipy.optimize as sco

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.sty import COLORS
from tenet.utils.io import folder_names, load_file, save_graph, format_value, format_sign
from tenet.utils.helpers import energies_aggregated
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
	simul_name = "PE+TLT/BOTTOM"

	PBCs = False

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	num = 8
	
	ns = [0, 1, 3]
	ns_fit = [20, 20, 20]

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
					
		Ns.append(N)
		diffs_Ns.append(diffs)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

	fig, ax = plt.subplots(figsize=(2, 3))

	handles = []
	handles_ = []

	linestyles = ["-", "-", "-"] 
	colors = [COLORS[0],COLORS[2],COLORS[4]]
	colors_ = [COLORS[1],COLORS[3],COLORS[5]]

	for (i, n) in enumerate(ns):
		for l in range(len(lam)):
			for t in range(len(theta)):
		
				def f(N, a, b):
					return a * np.log(N - 1) - 3 * np.log(N - 1) + b
		
				def g(N, a, b):
					return a * np.log(N - 1) - 4 * np.log(N - 1) + b

				def h(N, a, b):
					return a * np.log(N) + b

				ind = np.where(Ns >= 8)[0]

				p, = ax.plot(Ns[ind], diffs_Ns[ind, l, t, n], marker="o", markerfacecolor=colors[i], lw=0, markersize=10, label=fr"$E^{{\,}}_{{{n+1}}}-E^{{\,}}_{{{n}}}$", zorder=99-n)
				handles.append(p)

				x = np.linspace(np.min(Ns[ind]), np.max(Ns[ind]), 100)

				ind = np.where(Ns >= ns_fit[i])[0]

				if n == 0:
					popt, pcov = sco.curve_fit(f, Ns[ind], np.log(diffs_Ns[ind, l, t, n]))

					p, = ax.plot(x, np.exp(f(x, *popt)), color=colors_[i], linestyle=linestyles[i], linewidth=2, markersize=0, label=fr"$\mathrm{{fit}}^{{\,}}_{{E^{{\,}}_{{{n+1}}}-E^{{\,}}_{{{n}}}}}$", zorder=100-n)
				elif n == 1:
					popt, pcov = sco.curve_fit(g, Ns[ind], np.log(diffs_Ns[ind, l, t, n]))

					p, = ax.plot(x, np.exp(g(x, *popt)), color=colors_[i], linestyle=linestyles[i], linewidth=2, markersize=0, label=fr"$\mathrm{{fit}}^{{\,}}_{{E^{{\,}}_{{{n+1}}}-E^{{\,}}_{{{n}}}}}$", zorder=100-n)
				else:
					popt, pcov = sco.curve_fit(h, Ns[ind], np.log(diffs_Ns[ind, l, t, n]))

					p, = ax.plot(x, np.exp(h(x, *popt)), color=colors_[i], linestyle=linestyles[i], linewidth=2, markersize=0, label=fr"$\mathrm{{fit}}^{{\,}}_{{E^{{\,}}_{{{n+1}}}-E^{{\,}}_{{{n}}}}}$", zorder=100-n)
				perr = np.sqrt(np.diag(pcov))

				handles_.append(p)

				a = popt[0]
				da = perr[0]

				b = np.exp(popt[1])
				db = perr[1] * np.exp(popt[1])

				print(n, np.array([a, b]), np.array([da, db]))

				if n ==  0:
					R2 = r2(np.log(diffs_Ns[ind, l, t, n]), f(Ns[ind], *popt))
					print("R2 (algebraic)", R2)
					
					popt, pcov = sco.curve_fit(h, np.exp(Ns[ind]), np.log(diffs_Ns[ind, l, t, n]))
					perr = np.sqrt(np.diag(pcov))

					a = 1 / popt[0]
					da = a ** 2 * perr[0]

					b = np.exp(popt[1])
					db = perr[1] * np.exp(popt[1])

					print(n, np.array([a, b]), np.array([da, db]))

					R2 = r2(np.log(diffs_Ns[ind, l, t, n]), h(np.exp(Ns[ind]), *popt))
					print("R2 (exponential)", R2)

	ax.set_xscale("log")
	ax.set_yscale("log")

	ax.set_xlabel(r"$2N$")

	ax.set_xlim([6.5, 100])

	ax.set_xticks([10, 30, 100])
	ax.set_xticklabels(["$10^{1}$", 30, "$10^{2}$"])

	ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([2 * k for k in range(3, 15)]))
	ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

	ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([10 ** k for k in range(-10,1)]))
	ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

	ax.text(
		0.97, 0.97,
		"(a)",
		transform=ax.transAxes,
		ha="right",
		va="top",
		fontsize=12)
	
	ax.legend(
		handles=[*handles, *handles_],
		loc="center right",
		ncol=1,
		borderpad=0.3,
		labelspacing=0.8,
		bbox_to_anchor=(1.13, 0.45)).set_zorder(1001)

	for spine in ax.spines.values():
		spine.set_zorder(999)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"gaps")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()
	
if __name__ == "__main__":
	main()
