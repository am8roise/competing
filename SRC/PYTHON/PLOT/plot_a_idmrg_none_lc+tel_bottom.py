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

eps_norm = 1e-5
num_eigen = 10

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "LC+TEL/BOTTOM"

	boundary_name = "NONE"
	method_name = "IDMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
					
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
		xi = np.zeros((len(lam), len(theta)))
		spins = np.zeros((len(lam), len(theta), 6))
		spins_SPT = np.zeros((len(lam), len(theta), 6))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			val_L = data['val_L'][:num_eigen]

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]

			xi[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = - N / np.log(np.abs(val_L[1]))

			spinX, spinY, spinZ = data['mags']
			spinX_SPT, spinY_SPT, spinZ_SPT = data['mags_spt']

			spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.abs(np.mean(spinX))
			spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.abs(np.mean(spinY))
			spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.abs(np.mean(spinZ))

			spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = np.mean(np.abs(spinX))
			spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 4] = np.mean(np.abs(spinY))
			spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 5] = np.mean(np.abs(spinZ))
			
			spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.abs(np.mean(spinX_SPT))
			spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.abs(np.mean(spinY_SPT))
			spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.abs(np.mean(spinZ_SPT))

			spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = np.mean(np.abs(spinX_SPT))
			spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 4] = np.mean(np.abs(spinY_SPT))
			spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 5] = np.mean(np.abs(spinZ_SPT))
		
		for t in range(len(theta)):

			fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]}, figsize=(4.5, 2.5))
			plt.subplots_adjust(wspace=0.5)

			def f(x, a, b, c):
				return a * np.log(np.abs(x - b)) + c

			popt, pcov = sco.curve_fit(f, lam[55:65], np.log(xi[55:65, t]), [-0.2, 0.788, 1])
			perr = np.sqrt(np.diag(pcov))

			print(popt, perr)

			c = np.exp(popt[2])
			dc = perr[2] * c
			print(c, dc)

			p, = ax[0].plot(lam, xi[:, t], marker="o", color=COLORS[0], lw=0, markersize=7, zorder=101)

			x = np.linspace(0, lam[65], 1000)

			p, = ax[0].plot(x, np.exp(f(x, *popt)), color=COLORS[4], linewidth=2, linestyle='-', zorder=110, label=r"$\mathrm{fit}^{\,}_{\xi}$")

			ax[0].set_xlabel(fr"$\lambda$")
			ax[0].set_ylabel(r"$\xi$")

			ax[0].text(
				0.03, 0.97,
				"(c)",
				transform=ax[0].transAxes,
				ha="left",
				va="top",
				fontsize=12)

			ax[0].legend(
				loc="upper center",
				ncol=1,
				bbox_to_anchor=(0.5, 1.03)).set_zorder(1001)

			ax[0].axvline(np.pi/4, color="0", lw=0.5, linestyle="-")

			ax[0].set_xlim(0, 1)

			ax[0].set_xticks([0, 1])
			ax[0].set_yticks([0, 10, 20])

			ax[0].xaxis.set_minor_locator(mpl.ticker.FixedLocator([a for a in np.linspace(0, 1, 11)]))
			ax[0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

			axin = ax[0].inset_axes([0.21, 0.42, 0.5, 0.43])
			axin.set_zorder(100)

			p, = axin.plot(np.abs(lam[:65] - popt[1]), xi[:65, t], marker="o", color=COLORS[0], lw=0, markersize=7, zorder=101)

			p, = axin.plot(np.abs(x - popt[1]), np.exp(f(x, *popt)), color=COLORS[4], linewidth=2, linestyle='-', marker="none", zorder=102)

			axin.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{0}}|$", fontsize=10)

			axin.set_xscale("log")
			axin.set_yscale("log")

			axin.set_ylim(1, 40)

			axin.set_xticks([0.001, 0.1])
			axin.set_yticks([1, 10])

			axin.xaxis.set_minor_locator(mpl.ticker.FixedLocator([0.01]))
			axin.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

			axin.tick_params(labelsize=10)

			p, = ax[1].plot(lam, np.abs(spins[:, t, 3]) + np.abs(spins_SPT[:, t, 4]), marker="o", color=COLORS[2], lw=0, markersize=7, zorder=110)

			ax[1].axvline(np.pi/4, color="0", lw=0.5, linestyle="-")

			ax[1].set_ylim(0.8, 1)

			ax[1].set_xticks([0, 1])
			ax[1].set_yticks([0.8, 0.9, 1])

			ax[1].xaxis.set_minor_locator(mpl.ticker.FixedLocator([a for a in np.linspace(0, 1, 11)]))
			ax[1].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


			ax[1].set_xlabel(fr"$\lambda$")

			ax[1].set_ylabel(r"$|m^{x}_{\mathrm{sta}}|+|m^{\mathrm{SPT},y}_{\mathrm{sta}}|$")

			ax[1].text(
				0.03, 0.97,
				"(d)",
				transform=ax[1].transAxes,
				ha="left",
				va="top",
				fontsize=12)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_+_op")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

if __name__ == "__main__":
	main()

