import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import seaborn as sns
import scipy.optimize as sco

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph, format_value
from tenet.utils.sty import COLORS
from tenet.utils.math import find_nearest
from tenet.linalg.sun import SUIrrep

su2 = SUIrrep(2, 2)
X, Y, Z = [2 * su2.generators[i] for i in range(3)]

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

plot_entropy = True

eps_norm = 1e-5
num_eigen = 10

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "PS+PDC/BOTTOM"

	boundary_name = "NONE"
	method_name = "IDMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	chis = []
	xis = []
	Ss = []
					
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		engine_folder = os.path.join(DATA_FOLDER, base_folder, folder)
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()
		chi = np.unique([int(vals[i]['chi']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
		xi = np.zeros((len(lam), len(theta)))
		S = np.zeros((len(lam), len(theta)))

		for i in range(len(data_filenames)):

			with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			val_L = data['val_L'][:num_eigen]

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]

			xi[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = - N / np.log(np.abs(val_L[1]))
		
			S[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = engine.psi.compute_entropy(N // 2)

		chis.append(chi)
		xis.append(xi)
		Ss.append(S)

	ind = np.argsort(chis)
	chis = np.array(chis)[ind]
	xis = np.array(xis)[ind, :, :]
	Ss = np.array(Ss)[ind, :, :]

	cs = np.zeros((len(lam), len(theta), 1))

	for l in range(len(lam)):

		for t in range(len(theta)):

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[2],COLORS[3],COLORS[5],COLORS[4]], N=len(chis))

			fig, ax = plt.subplots(figsize=(1.8, 2.5))

			def f(x, a, b):
				return a / 6 * np.log(x) + b

			ind = np.where(chis >= 32)[0]

			popt, pcov = sco.curve_fit(f, xis[ind, l, t], Ss[ind, l, t], [2, 1])
			perr = np.sqrt(np.diag(pcov))

			print(popt, perr)

			cs[np.where(lam[l]==lam), np.where(theta[t]==theta)] = popt[0]

			for i in range(len(chis)):
				p, = ax.plot(xis[i, l, t], Ss[i, l, t], marker='o', markersize=10, linewidth=0,
				markerfacecolor=cmap(i), zorder=101)

			x = np.linspace(np.min(xis[:, l, t]) * 0.8, np.max(xis[:, l, t]) * 1.2, 100)
			p, = ax.plot(x, f(x, *popt), color=COLORS[0], linewidth=2, linestyle=(0, (0, 2)), zorder=110, label=r"$\mathrm{fit}^{\,}_{S}$")

			cb = fig.colorbar(
					mpl.cm.ScalarMappable(
						norm=mpl.colors.BoundaryNorm(
							boundaries=np.arange(len(chis)+1)-0.5,
							ncolors=len(chis)
						),
						cmap=cmap),
					ax=ax,
					drawedges=False)

			cb.ax.set_title(r"$\chi$", fontsize=12, pad=7)

			cb.set_ticks(np.arange(len(chis))[::4])
			cb.set_ticklabels(chis[::4])

			cb.ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(len(chis))))

			ax.set_xlabel(r"$\xi^{\,}_{\chi}$")
			ax.set_ylabel(r"$S^{\,}_{\chi}$")

			ax.set_xscale("log")

			ax.text(
				0.97, 0.02,
				"(b)",
				transform=ax.transAxes,
				ha="right",
				va="bottom",
				fontsize=12)

			ax.legend(
				loc="upper left",
				bbox_to_anchor=(0.1, 1.05)).set_zorder(101)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"entropy")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

if __name__ == "__main__":
	main()


