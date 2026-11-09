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

plot_xi = True
plot_spin = True
plot_mod = True

eps_norm = 1e-5
num_eigen = 100

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "LM+TML/TOP"

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
		spectrum = np.zeros((len(lam), len(theta), num_eigen))
		mod = np.zeros((len(lam), len(theta), num_eigen))

		for i in range(len(data_filenames)):

			with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)
		
			print(data_filenames[i])

			val_L = data['val_L']

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]

			val_L = val_L[:num_eigen]
			
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
		
			if plot_mod:

				spectrum[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = np.log10(np.abs(val_L[::-1]))
				mod[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = 1. / N * np.log(val_L[::-1]).imag % (2 * np.pi / N)
		if plot_xi:
		
			for l in range(len(lam)):

				fig, ax = plt.subplots(figsize=(8, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, theta[:10], np.log(xi[l, :10]), [1, 0, 0])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = ax.plot(np.abs(theta - popt[1]), xi[l, :], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = ax.plot(np.abs(theta[:10] - popt[1]), np.exp(f(theta[:10], *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=100, label=fr"$\xi\sim|\theta-\theta^{{\,}}_{{0}}|^{{{popt[0]:.3f}\pm{perr[0]:.3f}}}$")

				ax.set_xlabel(fr"$|\theta-\theta^{{\,}}_{{0}}|,\quad \theta^{{\,}}_{{0}}={popt[1]:.4f}\pm{perr[1]:.4f}$")
				ax.set_ylabel(r"$\xi$")

				ax.set_xscale('log')
				ax.set_yscale('log')

				ax.legend(loc="upper right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_lam={lam[l]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_spin:

			for l in range(len(lam)):
			
				fig, ax = plt.subplots(figsize=(8, 3))
				
				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, theta[:5], np.log(spins[l, :5, 5]), [1, 0, 0])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = ax.plot(np.abs(theta[:] - popt[1]), spins[l, :, 5], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = ax.plot(np.abs(theta[:5] - popt[1]), np.exp(f(theta[:5], *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=100, label=fr"$m^{{z}}_{{\mathrm{{sta}}}}\sim|\theta-\theta^{{\,}}_{{0}}|^{{{popt[0]:.3f}\pm{perr[0]:.3f}}}$")

				ax.set_xlabel(fr"$|\theta-\theta^{{\,}}_{{0}}|,\quad \theta^{{\,}}_{{0}}={popt[1]:.4f}\pm{perr[1]:.4f}$")
				ax.set_ylabel(r"$m^{{z}}_{{\mathrm{{sta}}}}$")

				ax.set_xscale('log')
				ax.set_yscale('log')

				ax.legend(loc="lower right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
					
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"spin_lam={lam[l]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_mod:

			fig, ax = plt.subplots(figsize=(8, 3))

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

			for t in range(len(theta)):
				plt.scatter([theta[t] for _ in range(len(mod[0, t, :]))], mod[0, t, :], marker='.', s=8, color=cmap(1 - np.abs(spectrum[0, t, :]) / np.max(np.abs(spectrum[0, t, :]))), zorder=100)
				print(spectrum[0, t, :])
			
			ax.minorticks_on()

			norm = mpl.colors.Normalize(vmin=np.min(spectrum), vmax=0)
			fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), location="right", ax=ax, label=fr"$\log_{{10}} |t_i|$")

			plt.xlabel(fr"$\theta$")

			plt.ylabel(fr"$\frac{{1}}{{{N}}} \Im \log t_i \quad \mathrm{{mod}}\ \frac{{2\pi}}{{{N}}}$")
			
			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"mod")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()


if __name__ == "__main__":
	main()

