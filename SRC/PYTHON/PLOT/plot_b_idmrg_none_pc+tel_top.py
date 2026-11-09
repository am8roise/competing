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

plot_corr = True

start_x = 10
end_x = 50
start_z = 0
end_z = 8

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "PC+TEL/TOP"

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
					
				fig, ax = plt.subplots(figsize=(8, 3))
				
				def f(x, a, b):
					return a * x + b

				sites = np.array(range(1, len(corrX) + 1))

				popt, pcov = sco.curve_fit(f, np.log(sites[start_x:end_x]), np.log(corrX[start_x:end_x]))
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = plt.plot(sites, corrX, marker='.', markeredgewidth=1.5, markersize=8, color=COLORS[0], linewidth=0, zorder=100, label=r"$C^{x}$")

				p, = ax.plot(sites[start_x:end_x], np.exp(f(np.log(sites[start_x:end_x]), *popt)), color=COLORS[2], linewidth=1.5, linestyle='-', zorder=102, label=fr"$C^{{x}}\sim s^{{{popt[0]:.3f}\pm{perr[0]:.3f}}}$")
				
				ind = np.where(sites % 4 == 0)[0]

				popt, pcov = sco.curve_fit(f, np.log(sites[ind][start_z:end_z]), np.log(corrZ[ind][start_z:end_z]))
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = plt.plot(sites[ind], corrZ[ind], marker='x', markeredgewidth=1.5, markersize=8, color=COLORS[6], linewidth=0, zorder=101, label=r"$C^{z}$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				
				p, = ax.plot(sites[ind][start_z:end_z], np.exp(f(np.log(sites[ind][start_z:end_z]), *popt)), color=COLORS[7], linewidth=1.5, linestyle='--', zorder=103, label=fr"$C^{{z}}\sim s^{{{popt[0]:.3f}\pm{perr[0]:.3f}}}$")

				ax.minorticks_on()

				ax.set_xlabel(r'$s$')

				ax.set_xscale('log')
				ax.set_yscale('log')

				ax.legend(loc="upper right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"corr_{data_filenames[i][:-4]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

if __name__ == "__main__":
	main()

