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
from tenet.utils.io import folder_names, load_file, save_graph, format_value, format_sign
from tenet.utils.sty import COLORS
from tenet.utils.math import find_nearest
from tenet.linalg.sun import SUIrrep

su2 = SUIrrep(2, 2)
X, Y, Z = [2 * su2.generators[i] for i in range(3)]

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

plot_corr = True

start_x = 3
end_x = 100
start_z = 3
end_z = 20

eps_norm = 1e-5
num_eigen = 300

def connected_correlation(N, corr, spin):

	r = (len(corr) + 1) // N
	spin_extended = np.concatenate([spin for _ in range(r)], axis=0)

	conn = corr - spin_extended[0] * spin_extended[1:]
	return conn

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "LC+TMB/BOTTOM"

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
		
		eta = np.zeros((len(lam), len(theta), 2))

		if N == 4:
			continue

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			print(data_filenames[i])

			if plot_corr:

				corrX, corrY, corrZ = data['corrs']

				corrX = np.abs(corrX)
				corrY = np.abs(corrY)
				corrZ = connected_correlation(N, corrZ, data['mags'][2, :])

				fig, ax = plt.subplots(figsize=(5, 3))
				
				def f(x, a, b):
					return a * x + b
				
				if vals[i]['lam'] < 0.9:

					sites = np.array(range(1, len(corrY) + 1))

					popt, pcov = sco.curve_fit(f, np.log(sites[start_x:end_x]), np.log(corrY[start_x:end_x]))
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = popt[0]

					p, = plt.plot(sites, corrY, marker='o', markersize=8, color=COLORS[3], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

					p, = ax.plot(sites[start_x:end_x], np.exp(f(np.log(sites[start_x:end_x]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					try:
						
						sites = np.array(range(1, len(corrZ) + 1))

						popt, pcov = sco.curve_fit(f, np.log(sites[start_z:end_z]), np.log(np.abs(corrZ)[start_z:end_z]))
						perr = np.sqrt(np.diag(pcov))

						print(popt, perr)

						eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[0]

						p, = ax.plot(sites[start_z:end_z], np.exp(f(np.log(sites[start_z:end_z]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{z}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")
					except:
						pass

					p, = plt.plot(sites, np.abs(corrZ), marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

					ax.set_xlabel(r'$s$')

					ax.set_xscale('log')
					ax.set_yscale('log')

					ax.legend(loc="lower left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
					
					plt.draw()

					if args.save:
						save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "corr"), f"x_{data_filenames[i][:-4]}")
						fig.clf()

					if args.show:
						plt.show()

					plt.close()

				if vals[i]['lam'] > 0.3:

					fig, ax = plt.subplots(figsize=(8, 3))

					def f(x, a, b, c):
						return a * x ** b * np.cos(x * c * 2*np.pi)

					sites = np.array(range(1, len(corrZ) + 1))

					try:
						popt, pcov = sco.curve_fit(f, sites[start_z:end_z], corrZ[start_z:end_z], [0.1, -1, 0.3])
						perr = np.sqrt(np.diag(pcov))

						print(popt, perr)

						eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[1]

						p, = ax.plot(sites[start_z:end_z], f(sites[start_z:end_z], *popt), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$C^{{z}}=\alpha\,s^{{{popt[1]:.2f}\pm{perr[1]:.2f}}}\cos\left[2\pi\times({popt[2]:.5f}\pm{perr[2]:.5f})+\beta\right]+\gamma$")
					except:
						pass

					p, = plt.plot(sites, corrZ, marker='.', markeredgewidth=1.5, markersize=6, color=COLORS[4], linewidth=0.5, linestyle='--', zorder=103, label=r"$C^{z}$")

					ax.set_xlim(-5, 120)
					ax.set_ylim(-0.2, 0.1)

					ax.set_xlabel(r'$s$')

					ax.legend(loc="lower right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
					
					plt.draw()

					if args.save:
						save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "corr"), f"z_{data_filenames[i][:-4]}")
						fig.clf()

					if args.show:
						plt.show()

					plt.close()

		if plot_corr:
			
			fig, ax = plt.subplots(figsize=(8, 3))

			ind = np.where(eta[:, 0, 0] != 0)[0]

			p, = plt.plot(lam[ind], eta[ind, 0, 0], linewidth=0, color=COLORS[0], marker='o', markersize=8, zorder=100, label=r"$\eta^{x}$")

			ind = np.where(eta[:, 0, 1] != 0)[0]

			p, = plt.plot(lam[ind], eta[ind, 0, 1], linewidth=0, color=COLORS[6], marker='o', markersize=8, zorder=100, label=r"$\eta^{z,\mathrm{c}}$")

			ax.set_xlabel(fr"$\lambda$")
			
			ax.legend(
				loc="upper right",
				ncol=1,
				bbox_to_anchor=(0.9, 1.1)).set_zorder(1000)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"eta")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()	

if __name__ == "__main__":
	main()

