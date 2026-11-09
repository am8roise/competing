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

plot_corr = True

start_x = 40
end_x = 10
start_y = 20
end_y = 20
start_z = 20
end_z = 30

def connected_correlation(N, corr, spin):

	r = (len(corr) + 1) // N
	spin_extended = np.concatenate([spin for _ in range(r)], axis=0)

	conn = corr - spin_extended[0] * spin_extended[1:]
	return conn

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "PC+TEL/BOTTOM"

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

			if plot_corr:

				with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
					engine = pickle.load(file)

				corrX, corrY, corrZ = data['corrs']

				corrX = np.abs(connected_correlation(N, corrX, data['mags'][0, :]))
				corrY = np.abs(corrY)
				corrZ = np.abs(corrZ)

				sites_x = np.arange(1, len(corrX) + 1)
				sites_y = np.arange(1, len(corrY) + 1)
				sites_z = np.arange(1, len(corrZ) + 1)

				select_x = np.where(corrX > 1E-7)[0]
				select_y = np.where(corrY > 1E-19)[0]
				select_z = np.where(corrZ > 1E-19)[0]

				corrX = corrX[select_x]
				corrY = corrY[select_y]
				corrZ = corrZ[select_z]

				sites_x = sites_x[select_x]
				sites_y = sites_y[select_y]
				sites_z = sites_z[select_z]
					
				fig, ax = plt.subplots(figsize=(8, 3))

				p, = plt.plot(sites_x, corrX, marker='x', markeredgewidth=1.5, markersize=8, color=COLORS[0], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				p, = plt.plot(sites_y, corrY, marker='.', markeredgewidth=1.5, markersize=8, color=COLORS[4], linewidth=0, zorder=100, label=r"$C^{y}$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				p, = plt.plot(sites_z, corrZ, marker='+', markeredgewidth=1.5, markersize=8, color=COLORS[6], linewidth=0, zorder=100, label=r"$C^{z}$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				_sites_x = sites_x[start_x:-end_x]
				_corrX = corrX[start_x:-end_x]

				_sites_y = sites_y[start_y:-end_y]
				_corrY = corrY[start_y:-end_y]

				_sites_z = sites_z[start_z:-end_z]
				_corrZ = corrZ[start_z:-end_z]
				
				def f(x, a, b):
					return a * np.exp(x) + b

				popt, pcov = sco.curve_fit(f, np.log(_sites_x), np.log(_corrX))
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				a = 1 / popt[0]
				da = a**2 * perr[0]

				p, = ax.plot(_sites_x, np.exp(f(np.log(_sites_x), *popt)), color=COLORS[2], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim e^{{-s/{format_value(np.abs(a), da)}}}$")

				popt, pcov = sco.curve_fit(f, np.log(_sites_y), np.log(_corrY))
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)
				
				a = 1 / popt[0]
				da = a**2 * perr[0]

				p, = ax.plot(_sites_y, np.exp(f(np.log(_sites_y), *popt)), color=COLORS[5], linewidth=1.5, linestyle='-', zorder=102, label=fr"$C^{{y}}\sim e^{{-s/{format_value(np.abs(a), da)}}}$")

				popt, pcov = sco.curve_fit(f, np.log(_sites_z), np.log(_corrZ))
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				a = 1 / popt[0]
				da = a**2 * perr[0]

				p, = ax.plot(_sites_z, np.exp(f(np.log(_sites_z), *popt)), color=COLORS[7], linewidth=1.5, linestyle='-', zorder=102, label=fr"$C^{{z}}\sim e^{{-s/{format_value(np.abs(a), da)}}}$")
				
				ax.minorticks_on()

				ax.set_xlabel(r'$s$')

				# ax.set_xscale('log')
				ax.set_yscale('log')

				ax.legend(loc="upper right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"corr")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

if __name__ == "__main__":
	main()

