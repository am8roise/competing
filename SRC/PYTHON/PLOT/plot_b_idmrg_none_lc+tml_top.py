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

plot_xi = True
plot_spin = True

eps_norm = 1e-5
num_eigen = 100

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "LC+TML/TOP"

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

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			if plot_xi:

				val_L = data['val_L'][:num_eigen]

				sort_L = np.argsort(np.abs(val_L))[::-1]
				val_L = val_L[sort_L]

				xi[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = - N / np.log(np.abs(val_L[1]))

			if plot_spin:

				spinX, spinY, spinZ = data['mags']

				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.abs(np.mean(spinX))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.abs(np.mean(spinY))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.abs(np.mean(spinZ))

				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = np.mean(np.abs(spinX))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 4] = np.mean(np.abs(spinY))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 5] = np.mean(np.abs(spinZ))
				
		if plot_xi:

			for l in range(len(lam)):

				fig, ax = plt.subplots(figsize=(4, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, theta[0:20], np.log(xi[l, 0:20]), [-2, 0, 0])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)
				print(theta[0:20])

				p, = ax.plot(theta, xi[l, :], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				x = np.linspace(theta[0], np.pi/4, 1000)

				p, = ax.plot(x, np.exp(f(x, *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi\sim|\theta-\theta^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")

				ax.set_xlabel(fr"$\theta,\quad \theta^{{\,}}_{{0}}={format_value(popt[1], perr[1])}$")
				ax.set_ylabel(r"$\xi$")

				ax.axvline(0, color="k", linestyle="--")

				ax.legend(loc="upper right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1, borderaxespad=0.3).set_zorder(200)

				ax.minorticks_on()

				# ax.set_xlim(-0.4, 0.8)
				# ax.set_ylim(0, 100)

				axin = ax.inset_axes([0.5, 0.4, 0.4, 0.38])
				axin.set_zorder(101)

				p, = axin.plot(np.abs(theta[0:] - popt[1]), xi[l, 0:], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = axin.plot(np.abs(x - popt[1]), np.exp(f(x, *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi\sim|\theta-\theta^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")

				axin.set_xlabel(fr"$|\theta-\theta^{{\,}}_{{0}}|$")

				axin.set_xscale("log")
				axin.set_yscale("log")

				# axin.set_ylim(1, 40)

				# axin.set_yticks([1, 10, 100])

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_lam={lam[l]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_spin:

			for l in range(len(lam)):

				fig, ax = plt.subplots(figsize=(4, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, theta[0:18], np.log(spins[l, 0:18, 5]), [1, 0, 0])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)
				print(theta[0:18])

				p, = ax.plot(theta, spins[l, :, 5], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				x = np.linspace(theta[0], np.pi/4, 1000)

				p, = ax.plot(x, np.exp(f(x, *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=110, label=fr"$m^{{x}}_{{\mathrm{{uni}}}}\sim|\theta-\theta^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")

				ax.set_xlabel(fr"$\theta,\quad \theta^{{\,}}_{{0}}={format_value(popt[1], perr[1])}$")
				ax.set_ylabel(r"$m^{{z}}_{{\mathrm{{sta}}}}$")

				ax.axvline(0, color="k", linestyle="--")

				ax.legend(loc="upper right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1, borderaxespad=0.3).set_zorder(200)

				ax.minorticks_on()

				ax.set_ylim(0, 0.65)

				axin = ax.inset_axes([0.55, 0.32, 0.4, 0.3])
				axin.set_zorder(101)

				p, = axin.plot(np.abs(theta[0:] - popt[1]), spins[l, 0:, 5], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = axin.plot(np.abs(x - popt[1]), np.exp(f(x, *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=110, label=fr"$m^{{z}}_{{\mathrm{{sta}}}}\sim|\theta-\theta^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")

				axin.set_xlabel(fr"$|\theta-\theta^{{\,}}_{{0}}|$")

				axin.set_xscale("log")
				axin.set_yscale("log")

				# axin.set_ylim(0.2, 3)

				# axin.set_yticks([0.1, 1])

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"spin_lam={lam[l]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

if __name__ == "__main__":
	main()

