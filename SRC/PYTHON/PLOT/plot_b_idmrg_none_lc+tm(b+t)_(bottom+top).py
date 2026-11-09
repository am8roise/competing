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

start_x = 20
end_x = 100
start_y = 20
end_y = 100
start_z = 20
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
	simul_name = "LC+TMT/TOP"

	boundary_name = "NONE"
	method_name = "IDMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
					
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		engine_folder = os.path.join(DATA_FOLDER, base_folder, folder)
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam_top = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
		eta_top = np.zeros((len(lam_top), len(theta), 3))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			print(data_filenames[i])

			if plot_corr:

				corrX, corrY, corrZ = data['corrs']

				corrX = np.abs(corrX)
				corrY = np.abs(corrY)
				corrZ = np.abs(corrZ)

				def f(x, a, b):
					return -a * x + b
				
				if vals[i]['lam'] > 2.85:

					fig, ax = plt.subplots(figsize=(5, 3))
					
					sites = np.array(range(1, len(corrX) + 1))

					popt, pcov = sco.curve_fit(f, np.log(sites[start_x:end_x]), np.log(corrX[start_x:end_x]))
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)	

					eta_top[np.where(vals[i]['lam']==lam_top), np.where(vals[i]['theta']==theta), 0] = popt[0]

					p, = plt.plot(sites, corrX, marker='o', markersize=8, color=COLORS[4], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

					p, = ax.plot(sites[start_x:end_x], np.exp(f(np.log(sites[start_x:end_x]), *popt)), color=COLORS[2], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					sites = np.array(range(1, len(corrY) + 1))

					popt, pcov = sco.curve_fit(f, np.log(sites[start_y:end_y]), np.log(corrY[start_y:end_y]))
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta_top[np.where(vals[i]['lam']==lam_top), np.where(vals[i]['theta']==theta), 1] = popt[0]

					p, = plt.plot(sites, corrY, marker='o', markersize=8, color=COLORS[3], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

					p, = ax.plot(sites[start_y:end_y], np.exp(f(np.log(sites[start_y:end_y]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					sites = np.array(range(1, len(corrZ) + 1))

					if vals[i]['lam'] == np.round(np.pi, 15):
						ind = np.where(sites % 2 == 1)[0]
					else:
						ind = np.where(sites)[0]

					popt, pcov = sco.curve_fit(f, np.log(sites[ind][start_z:end_z]), np.log(corrZ[ind][start_z:end_z]))
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta_top[np.where(vals[i]['lam']==lam_top), np.where(vals[i]['theta']==theta), 2] = popt[0]

					p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=101, label=r"$|C^{z}|$")
					
					p, = ax.plot(sites[ind][start_z:end_z], np.exp(f(np.log(sites[ind][start_z:end_z]), *popt)), color=COLORS[5], linewidth=1.5, linestyle='--', zorder=103, label=fr"$|C^{{z}}|\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					ax.set_ylabel(r'$s$')

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

		lam_bottom = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
		eta_bottom = np.zeros((len(lam_bottom), len(theta), 3))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			print(data_filenames[i])

			if plot_corr:

				corrX, corrY, corrZ = data['corrs']

				corrX = np.abs(corrX)
				corrY = np.abs(corrY)
				corrZ = np.abs(corrZ)

				def f(x, a, b):
					return -a * x + b
				
				if vals[i]['lam'] < 0.3:
				
					fig, ax = plt.subplots(figsize=(5, 3))

					sites = np.array(range(1, len(corrX) + 1))

					if vals[i]['lam'] == 0:
						popt, pcov = sco.curve_fit(f, np.log(sites[ind][5:10]), np.log(corrX[ind][5:10]))
						perr = np.sqrt(np.diag(pcov))
					else:
						popt, pcov = sco.curve_fit(f, np.log(sites[start_x:end_x]), np.log(corrX[start_x:end_x]))
						perr = np.sqrt(np.diag(pcov))

					print(popt, perr)	

					eta_bottom[np.where(vals[i]['lam']==lam_bottom), np.where(vals[i]['theta']==theta), 0] = popt[0]

					p, = plt.plot(sites, corrX, marker='o', markersize=8, color=COLORS[4], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

					p, = ax.plot(sites[start_x:end_x], np.exp(f(np.log(sites[start_x:end_x]), *popt)), color=COLORS[2], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					sites = np.array(range(1, len(corrY) + 1))

					if vals[i]['lam'] == 0:
						popt, pcov = sco.curve_fit(f, np.log(sites[ind][5:10]), np.log(corrY[ind][5:10]))
						perr = np.sqrt(np.diag(pcov))
					else:
						popt, pcov = sco.curve_fit(f, np.log(sites[start_y:end_y]), np.log(corrY[start_y:end_y]))
						perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta_bottom[np.where(vals[i]['lam']==lam_bottom), np.where(vals[i]['theta']==theta), 1] = popt[0]

					p, = plt.plot(sites, corrY, marker='o', markersize=8, color=COLORS[3], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

					p, = ax.plot(sites[start_y:end_y], np.exp(f(np.log(sites[start_y:end_y]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					sites = np.array(range(1, len(corrZ) + 1))

					if vals[i]['lam'] == np.round(0, 15):
						ind = np.where(sites % 2 == 1)[0]
					else:
						ind = np.where(sites)[0]

					if vals[i]['lam'] > 0.29:
						popt, pcov = sco.curve_fit(f, np.log(sites[ind][150:350]), np.log(corrZ[ind][150:350]))
						perr = np.sqrt(np.diag(pcov))
					else:
						popt, pcov = sco.curve_fit(f, np.log(sites[ind][start_z:end_z]), np.log(corrZ[ind][start_z:end_z]))
						perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta_bottom[np.where(vals[i]['lam']==lam_bottom), np.where(vals[i]['theta']==theta), 2] = popt[0]

					p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=101, label=r"$|C^{z}|$")
					
					if vals[i]['lam'] > 0.29:
						p, = ax.plot(sites[ind][150:350], np.exp(f(np.log(sites[ind][150:350]), *popt)), color=COLORS[5], linewidth=1.5, linestyle='--', zorder=103, label=fr"$|C^{{z}}|\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")
					else:
						p, = ax.plot(sites[ind][start_z:end_z], np.exp(f(np.log(sites[ind][start_z:end_z]), *popt)), color=COLORS[5], linewidth=1.5, linestyle='--', zorder=103, label=fr"$|C^{{z}}|\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					ax.set_ylabel(r'$s$')

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
			
		fig, ax = plt.subplots(figsize=(5, 2))

		lam = np.concatenate((lam_top - np.pi, lam_bottom), axis=0)
		eta = np.concatenate((eta_top, eta_bottom), axis=0)

		ind = np.where(eta[:, 0, 0] != 0)[0]

		p, = plt.plot(lam[ind], eta[ind, 0, 0], linewidth=0, color=COLORS[3], marker='o', markersize=4, zorder=102, label=r"$\eta^{x}$")

		ind = np.where(eta[:, 0, 1] != 0)[0]

		p, = plt.plot(lam[ind], eta[ind, 0, 1], linewidth=0, color=COLORS[2], marker='o', markersize=8, zorder=101, label=r"$\eta^{y}$")

		ind = np.where(eta[:, 0, 2] != 0)[0]

		p, = plt.plot(lam[ind], eta[ind, 0, 2], linewidth=0, color=COLORS[0], marker='o', markersize=12, zorder=100, label=r"$\eta^{z}$")

		ax.set_xlabel(fr"$\lambda$")

		ax.annotate(r"$\lambda^{\prime}_{\mathrm{Neel}^{\,}_{z}}$", (-0.2849-0.045, 1.3), rotation=90)
		ax.annotate(r"$\lambda^{\,}_{\mathrm{gFM}^{\,}_{z}}$", (0.2993+0.01, 1.3), rotation=-90)

		ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
		ax.set_yticks([0, 0.5, 1, 2])
		ax.set_yticklabels(["$0$", r"$\frac{1}{2}$", "$1$", "$2$"])

		ax.axvline(-0.2849, linestyle="-", color=COLORS[4], zorder=11)
		ax.axvline(0.2993, linestyle=(0, (0, 3)), color=COLORS[4], zorder=11)

		ax.axhline(0, linestyle="-", color="0.75", linewidth=0.5, zorder=10)
		ax.axhline(0.5, linestyle="-", color="0.75", linewidth=0.5, zorder=10)
		ax.axhline(1, linestyle="-", color="0.75", linewidth=0.5, zorder=10)
		ax.axhline(2, linestyle="-", color="0.75", linewidth=0.5, zorder=10)
		
		ax.legend(
			loc="center left",
			ncol=1,
			bbox_to_anchor=(0.95, 0.2)).set_zorder(1000)

		plt.draw()

		if args.save:

			model_name = "B"
			simul_name = "LC+TMB+T/BOTTOM+TOP"

			boundary_name = "NONE"
			method_name = "IDMRG"

			base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

			folder = "N=4_chi=256"

			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"eta")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

if __name__ == "__main__":
	main()

