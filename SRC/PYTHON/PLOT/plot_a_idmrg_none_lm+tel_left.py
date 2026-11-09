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
from tenet.utils.io import folder_names, load_file, save_graph
from tenet.utils.sty import COLORS
from tenet.utils.math import find_nearest
from tenet.linalg.sun import SUIrrep

su2 = SUIrrep(2, 2)
X, Y, Z = [2 * su2.generators[i] for i in range(3)]

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

plot_xi = False
plot_spin = True

eps_norm = 1e-5
num_eigen = 100

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "LM+TEL/LEFT"

	boundary_name = "NONE"
	method_name = "IDMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
					
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		engine_folder = os.path.join(DATA_FOLDER, base_folder, folder)
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
		xi = np.zeros((len(lam), len(theta)))
		spins = np.zeros((len(lam), len(theta), 6))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			val_L = data['val_L'][:num_eigen]

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]

			if plot_xi:

				xi[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = - N / np.log(np.abs(val_L[1]))

			if plot_spin:

				spinX, spinY, spinZ = data['mags']

				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.abs(np.mean(spinX))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.abs(np.mean(spinY))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.abs(np.mean(spinZ))

				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = np.mean(np.abs(spinX))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 4] = np.mean(np.abs(spinY))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 5] = np.mean(np.abs(spinZ))

				spinX, spinY, spinZ = data['mags_spt']
		
		if plot_xi:
		
			for t in range(len(theta)):

				fig, ax = plt.subplots(figsize=(8, 3))

				def f(x, a, b, c):
					return a / np.abs(x - b) + c

				popt, pcov = sco.curve_fit(f, lam[10:-10], np.log(xi[10:-10, t]))
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = ax.plot(1 / np.abs(lam - popt[1]), xi[:, t], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = ax.plot(1 / np.abs(lam[10:-10] - popt[1]), np.exp(f(lam[10:-10], *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=100, label=fr"$\xi\sim e^{{|\lambda-\lambda^{{\,}}_{{0}}|\times({np.abs(popt[0]):.3f}\pm{perr[0]:.3f})}}$")

				ax.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{0}}|,\quad \lambda^{{\,}}_{{0}}={popt[1]:.4f}\pm{perr[1]:.4f}$")
				ax.set_ylabel(r"$\xi$")

				ax.set_yscale('log')

				ax.legend(loc="upper right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_spin:

			for t in range(len(theta)):
			
				fig, ax = plt.subplots(figsize=(8, 3))
				
				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, lam[-12:], np.log(spins[-12:, t, 3]), [1, np.pi/4, 0])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = ax.plot(np.abs(lam[:] - popt[1]), spins[:, t, 3], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = ax.plot(np.abs(lam[-12:] - popt[1]), np.exp(f(lam[-12:], *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=100, label=fr"$m^{{x}}_{{\mathrm{{sta}}}}\sim|\lambda-\lambda^{{\,}}_{{0}}|^{{{popt[0]:.3f}\pm{perr[0]:.3f}}}$")

				ax.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{0}}|,\quad \lambda^{{\,}}_{{0}}={popt[1]:.4f}\pm{perr[1]:.4f}$")
				ax.set_ylabel(r"$m^{{x}}_{{\mathrm{{sta}}}}$")

				ax.set_xscale('log')
				ax.set_yscale('log')

				ax.legend(loc="lower right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
					
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"spin_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

if __name__ == "__main__":
	main()

