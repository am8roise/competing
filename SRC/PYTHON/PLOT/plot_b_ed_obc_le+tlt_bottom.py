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
	simul_name = "LE+TLT/BOTTOM"

	PBCs = False

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	num = 8
	
	ns = [0, 1, 3]
	n_ns = [20, 20, 20]
	n_lam = [10, 10, 10]

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

	As = np.zeros((len(lam), len(theta), num - 1, 2))
	Bs = np.zeros((len(lam), len(theta), num - 1, 2))

	for t in range(len(theta)):
		for l in range(len(lam)):

			for (i, n) in enumerate(ns):

				def f(N, a, b):
					return a * np.log(N - 1) - 3 * np.log(N - 1) + b

				def g(N, a, b):
					return a * np.log(N - 1) - 4 * np.log(N - 1) + b

				def h(N, a, b):
					return a / N + b

				ind = np.where(Ns >= n_ns[i])[0]

				if n == 0:
					popt, pcov = sco.curve_fit(f, Ns[ind], np.log(diffs_Ns[ind, l, t, n]))
		
					perr = np.sqrt(np.diag(pcov))

					a = np.exp(popt[1])
					da = perr[1] * np.exp(popt[1])
				elif n == 1:
					popt, pcov = sco.curve_fit(g, Ns[ind], np.log(diffs_Ns[ind, l, t, n]))
		
					perr = np.sqrt(np.diag(pcov))

					a = np.exp(popt[1])
					da = perr[1] * np.exp(popt[1])
				else:
					popt, pcov = sco.curve_fit(h, Ns[ind], diffs_Ns[ind, l, t, n])
						
					perr = np.sqrt(np.diag(pcov))

					a = popt[0]
					da = perr[0]

				As[lam[l]==lam, theta[t]==theta, n, :] = [a, da]

	fig, ax = plt.subplots(2, 1, height_ratios=[1, 2], figsize=(2, 3))
	plt.subplots_adjust(hspace=0.15)

	handles = []

	linestyles = [(0.5, (0, 1.5)), "-", "-"] 
	colors = [COLORS[0],COLORS[2],COLORS[4]]
	colors_ = [COLORS[1],COLORS[3],COLORS[5]]

	handles = []
	handles_ = []

	for (i, n) in enumerate(ns):
		for t in range(len(theta)):

			def g(x, a, b, c):
				return a * x ** b + c

			if n == 0:
				label = fr"$v^{{\prime}}_{{{n+1},{n}}}$"
				label_ = fr"$\mathrm{{fit}}^{{\,}}_{{v^{{\prime}}_{{{n+1},{n}}}}}$"
			else:
				label = fr"$v^{{\,}}_{{{n+1},{n}}}$"
				label_ = fr"$\mathrm{{fit}}^{{\,}}_{{v^{{\,}}_{{{n+1},{n}}}}}$"

			if n == 1:
				p, = ax[0].plot(np.pi/2 - lam, As[:, t, n, 0], marker="o", markerfacecolor=colors[i], lw=0, markersize=7, label=label, zorder=99-n)
				handles_.append(p)
			else:
				p, = ax[1].plot(np.pi/2 - lam, As[:, t, n, 0], marker="o", markerfacecolor=colors[i], lw=0, markersize=7, label=label, zorder=99-n)
				handles.append(p)

			popt, pcov = sco.curve_fit(g, np.pi/2 - lam[-n_lam[i]:], As[-n_lam[i]:, t, n, 0], [10, 2, 0])
			perr = np.sqrt(np.diag(pcov))

			print(n, popt, perr)
			
			x = np.linspace(0, np.max(np.pi/2 - lam), 100)
			if n == 1:
				p, = ax[0].plot(x,  g(x, *popt), color=colors_[i], linestyle=linestyles[i], linewidth=2, markersize=0, zorder=100-n, label=label_)
				handles_.append(p)
			else:
				p, = ax[1].plot(x,  g(x, *popt), color=colors_[i], linestyle=linestyles[i], linewidth=2, markersize=0, zorder=100-n, label=label_)
				handles.append(p)

	ax[0].set_xlim(0, 0.05)
	ax[1].set_xlim(0, 0.05)
	ax[0].set_ylim(-0.01, 0.08)
	ax[1].set_ylim(-0.5, 8.5)

	ax[0].set_xticks([])
	ax[1].set_xticks([0, 0.02, 0.04])
	ax[1].set_xticklabels(["$0$", "$0.02$", "$0.04$"])

	ax[0].set_yticks([0, 0.05])
	ax[1].set_yticks([0, 2, 4, 6, 8])
	ax[0].set_yticklabels(["$0$", "$0.05$"])

	ax[1].set_xlabel(r"$\frac{\pi}{2}-\lambda$")

	ax[1].set_xlabel(r"$\frac{\pi}{2}-\lambda$")

	ax[0].text(
		0.99, 0.95,
		"(b)",
		transform=ax[0].transAxes,
		ha="right",
		va="top",
		fontsize=12)

	ax[0].legend(
		handles=handles_,
		loc="upper center",
		ncol=2,
		labelspacing=0.5,
		borderpad=0.3,
		bbox_to_anchor=(0.42, 1.1)).set_zorder(1001)

	ax[1].legend(
		handles=[handles[0], handles[2], handles[1], handles[3]],
		loc="upper center",
		ncol=2,
		labelspacing=0.5,
		borderpad=0.3,
		bbox_to_anchor=(0.56, 1.07)).set_zorder(1001)

	for spine in ax[0].spines.values():
		spine.set_zorder(999)
	for spine in ax[1].spines.values():
		spine.set_zorder(999)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"velocities")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()
	
if __name__ == "__main__":
	main()
