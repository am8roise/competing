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

num_eigen = 10

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "LC+TLL/BOTTOM"

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

			with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

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
		
		if plot_xi:
		
			for t in range(len(theta)):

				if theta[t] != 0.1:
					continue
			
				start_ = -31
				end_ = -25

				fig, ax = plt.subplots(figsize=(2.5, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, lam[start_:end_], np.log(xi[start_:end_, t]), [-0.2, 0.788, 1])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)
				print(lam[start_:end_])

				p, = ax.plot(lam, xi[:, t], marker='o', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				x = np.linspace(lam[0], lam[end_], 1000)

				p, = ax.plot(x, np.exp(f(x, *popt)), color=COLORS[4], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi^{{\,}}_{{\chi}}\sim|\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|^{{{format_value(popt[0], perr[0])}}}$")

				ax.set_xlabel(fr"$\lambda,\quad \lambda^{{\,}}_{{\mathrm{{c}}}}={format_value(popt[1], perr[1])}$")
				ax.set_ylabel(r"$\xi^{{\,}}_{{\chi}}$")

				ax.legend(
					loc="upper left",
					handlelength=1,
					bbox_to_anchor=(0.05, 1.05)).set_zorder(200)

				ax.set_xlim(0, 0.95)
				ax.set_ylim(-5, 65)

				ax.set_xticks([0, np.pi/8, np.pi/4])
				ax.set_xticklabels([r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$"])

				ax.axvline(np.pi/4, color="k", linestyle="-", linewidth=0.5)

				axin = ax.inset_axes([0.2, 0.4, 0.45, 0.45])
				axin.set_zorder(101)

				p, = axin.plot(np.abs(lam[:end_+1] - popt[1]), xi[:end_+1, t], marker='o', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = axin.plot(np.abs(x - popt[1]), np.exp(f(x, *popt)), color=COLORS[4], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi^{{\,}}_{{\chi}}\sim|\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|^{{{format_value(popt[0], perr[0])}}}$")

				axin.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|$")

				axin.set_xscale("log")
				axin.set_yscale("log")

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_spin:

			for t in range(len(theta)):

				if theta[t] != 0.1:
					continue

				start_ = -31
				end_ = -25

				fig, ax = plt.subplots(figsize=(2.5, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, lam[start_:end_], np.log(spins[start_:end_, t, 3]), [-0.2, 0.788, 1])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)
				print(lam[start_:end_])

				p, = ax.plot(lam, spins[:, t, 3], marker='o', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				x = np.linspace(lam[0], lam[end_], 1000)

				p, = ax.plot(x, np.exp(f(x, *popt)), color=COLORS[4], linewidth=1.5, linestyle='-', zorder=110, label=fr"$|m^{{x}}_{{\mathrm{{sta}}}}|\sim|\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|^{{{popt[0]:.3f}\pm{perr[0]:.3f}}}$")

				ax.set_xlabel(fr"$\lambda,\quad \lambda^{{\,}}_{{\mathrm{{c}}}}={format_value(popt[1], perr[1])}$")
				ax.set_ylabel(r"$|m^{x}_{\mathrm{sta}}|$")

				ax.legend(
					loc="upper right",
					handlelength=1,
					bbox_to_anchor=(1.05, 1.05)).set_zorder(200)

				ax.set_xlim(0, 0.95)
				ax.set_ylim(-0.1, 1.2)

				ax.set_xticks([0, np.pi/8, np.pi/4])
				ax.set_xticklabels([r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$"])
				ax.set_yticks([0, 0.5, 1])
				ax.set_yticklabels([r"$0$", r"$\frac{1}{2}$", r"$1$"])

				ax.axvline(np.pi/4, color="k", linestyle='-', linewidth=0.5)

				axin = ax.inset_axes([0.25, 0.25, 0.4, 0.33])
				axin.set_zorder(101)

				p, = axin.plot(np.abs(lam[:end_+1] - popt[1]), spins[:end_+1, t, 3], marker='o', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = axin.plot(np.abs(x - popt[1]), np.exp(f(x, *popt)), color=COLORS[4], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi^{{\,}}_{{\chi}}\sim|\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|^{{{popt[0]:.3f}\pm{perr[0]:.3f}}}$")

				axin.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|$")

				axin.set_xscale("log")
				axin.set_yscale("log")

				axin.set_ylim(0.1, 2)

				axin.set_yticks([0.1, 1])

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"spin_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

if __name__ == "__main__":
	main()

