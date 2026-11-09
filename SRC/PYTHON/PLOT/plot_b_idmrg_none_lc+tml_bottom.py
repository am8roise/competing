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

start_x = 2
end_x = 30
start_y = 5
end_y = 40
start_z = 40
end_z = 120

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
	simul_name = "LC+TML/BOTTOM"

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
		
		eta = np.zeros((len(lam), len(theta), 3))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			print(data_filenames[i])

			if plot_corr:

				corrX, corrY, corrZ = data['corrs']

				corrX = np.abs(corrX)
				corrY = np.abs(corrY)
				corrZ = np.abs(connected_correlation(N, corrZ, data["mags"][2, :]))

				fig, ax = plt.subplots(figsize=(5, 3))
				
				def f(x, a, b):
					return a * x + b			
				def g(x, a, b, c, d):
					return np.abs(a * x ** b * np.cos(x * c + d))
				
				if vals[i]['theta'] >= 0:

					sites = np.array(range(1, len(corrX) + 1))

					popt, pcov = sco.curve_fit(f, np.log(sites[start_x:end_x]), np.log(corrX[start_x:end_x]))
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = popt[0]

					p, = plt.plot(sites, corrX, marker='o', markersize=8, color=COLORS[3], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

					p, = ax.plot(sites[start_x:end_x], np.exp(f(np.log(sites[start_x:end_x]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					if vals[i]['theta'] in [0, 0.1]:
						sites = np.arange(2, len(corrY), 4)

						corrY = corrY[sites]

						popt, pcov = sco.curve_fit(f, np.log(sites[5:50]), np.log(corrY[5:50]))
						perr = np.sqrt(np.diag(pcov))

						print(popt, perr)

						eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[0]

						p, = plt.plot(sites, corrY, marker='o', markersize=8, color=COLORS[5], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{y}$")

						p, = ax.plot(sites[5:50], np.exp(f(np.log(sites[5:50]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{y}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")
					else:

						sites = np.array(range(1, len(corrY) + 1))

						popt, pcov = sco.curve_fit(f, np.log(sites[start_y:end_y]), np.log(corrY[start_y:end_y]))
						perr = np.sqrt(np.diag(pcov))

						print(popt, perr)

						eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[0]

						p, = plt.plot(sites, corrY, marker='o', markersize=8, color=COLORS[5], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{y}$")

						p, = ax.plot(sites[start_y:end_y], np.exp(f(np.log(sites[start_y:end_y]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{y}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					if vals[i]['theta'] in [0, 0.1]:
						sites = np.arange(3, len(corrZ), 4)

						corrZ = corrZ[sites]

						popt, pcov = sco.curve_fit(f, np.log(sites[0:30]), np.log(corrZ[0:30]))
						perr = np.sqrt(np.diag(pcov))

						print(popt, perr)

						eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = popt[0]

						p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{z}$")

						p, = ax.plot(sites[0:30], np.exp(f(np.log(sites[0:30]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{z}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")
					else:
						sites = np.array(range(1, len(corrZ) + 1))

						# popt, pcov = sco.curve_fit(g, sites[start_z:end_z], corrZ[start_z:end_z], [0.1, -1, np.pi/2, 0])
						popt, pcov = sco.curve_fit(f, np.log(sites[start_z:end_z]), np.log(corrZ[start_z:end_z]))
						perr = np.sqrt(np.diag(pcov))

						print(popt, perr)

						eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = popt[0]

						p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{z}$")

						# p, = ax.plot(sites[start_z:end_z], g(sites[start_z:end_z], *popt), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{z}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")
						p, = ax.plot(sites[start_z:end_z], np.exp(f(np.log(sites[start_z:end_z]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{z}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					# sites = np.array(range(1, len(corrZ) + 1))

					# if vals[i]['theta'] == np.round(0, 15):
					# 	ind = np.where(sites % 2 == 1)[0]
					# else:
					# 	ind = np.where(sites)[0]

					# popt, pcov = sco.curve_fit(f, np.log(sites[ind][start_z:end_z]), np.log(corrZ[ind][start_z:end_z]))
					# perr = np.sqrt(np.diag(pcov))

					# print(popt, perr)

					# if vals[i]['theta'] > 0.29 and vals[i]['theta'] < 0.3:
					# 	pass
					# else:
					# 	eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[0]

					# p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=101, label=r"$|C^{z}|$")
					
					# p, = ax.plot(sites[ind][start_z:end_z], np.exp(f(np.log(sites[ind][start_z:end_z]), *popt)), color=COLORS[5], linewidth=1.5, linestyle='--', zorder=103, label=fr"$|C^{{z}}|\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

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
					
		if plot_corr:
			
			fig, ax = plt.subplots(figsize=(8, 3))

			ind = np.where(eta[0, :, 0] != 0)[0]

			p, = plt.plot(theta[ind], eta[0, ind, 0], linewidth=0, color=COLORS[0], marker='o', markersize=8, zorder=100, label=r"$\eta^{x}$")

			ind = np.where(eta[0, :, 1] != 0)[0]

			p, = plt.plot(theta[ind], eta[0, ind, 1], linewidth=0, color=COLORS[2], marker='o', markersize=8, zorder=100, label=r"$\eta^{y}$")

			ind = np.where(eta[0, :, 2] != 0)[0]

			p, = plt.plot(theta[ind], eta[0, ind, 2], linewidth=0, color=COLORS[6], marker='o', markersize=8, zorder=100, label=r"$\eta^{z,\mathrm{c}}$")

			ax.set_xlabel(fr"$\theta$")
			
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

