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

start_x = 20
end_x = 70
start_z = 10
end_z = 40
start_z_rescale = 10
end_z_rescale = 35

eps_norm = 1e-5
num_eigen = 50

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
	simul_name = "PC+PDC/TOP"

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

			val_L, vec_L = data['val_L'], data['vec_L'] 
			val_R, vec_R = data['val_R'], data['vec_R']

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]
			sort_R = [find_nearest(val_R, val_L[i]) for i in range(len(val_L))]
			val_R = val_R[sort_R]

			vec_L = vec_L[:, sort_L]
			vec_R = vec_R[:, sort_R]

			val_L = val_L[:num_eigen]
			val_R = val_R[:num_eigen]
			vec_L = vec_L[:, :num_eigen]
			vec_R = vec_R[:, :num_eigen]

			ind = np.where(np.abs(val_L) > eps_norm)[0]
			val_L = val_L[ind]
			val_R = val_R[ind]
			vec_L = vec_L[:, ind]
			vec_R = vec_R[:, ind]

			true_len = len(val_L)

			for j in range(vec_L.shape[1]):
				overlap = np.abs(np.dot(vec_L[:, j], vec_R[:, j]))
				vec_L[:, j] = vec_L[:, j]/np.sqrt(overlap)
				vec_R[:, j] = vec_R[:, j]/np.sqrt(overlap)
			
			# norm = np.abs(val_L)
			q = 1. / N * np.log(val_L).imag % (2 * np.pi / N)

			x_R = engine.psi.compute_overlap_R(X, vec_R)
			y_R = engine.psi.compute_overlap_R(Y, vec_R)
			z_R = engine.psi.compute_overlap_R(Z, vec_R)
			x_L = engine.psi.compute_overlap_L(X, vec_L)
			y_L = engine.psi.compute_overlap_L(Y, vec_L)
			z_L = engine.psi.compute_overlap_L(Z, vec_L)

			overlap_x = np.abs(x_L[0] * x_R[0])
			overlap_y = np.abs(y_L[0] * y_R[0])
			overlap_z = np.abs(z_L[0] * z_R[0])

			sort_x = np.argsort(overlap_x)[::-1]
			sort_y = np.argsort(overlap_y)[::-1]
			sort_z = np.argsort(overlap_z)[::-1]

			overlap_x = overlap_x[sort_x]
			overlap_y = overlap_y[sort_y]
			overlap_z = overlap_z[sort_z]

			print("overlaps", overlap_z[:5])
			print("q", q[sort_z][:5])
			print("norm", np.abs(val_L[sort_z][:5]))
			print("xi", - N / np.log(np.abs(val_L[sort_z][:5])))

			if plot_corr:

				with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
					engine = pickle.load(file)

				corrX, corrY, corrZ = data['corrs']
					
				corrZ = np.abs(connected_correlation(N, corrZ, data['mags'][2, :]))

				fig, ax = plt.subplots(figsize=(8, 3))
				
				def f(x, a, b):
					return a * x + b

				sites = np.array(range(1, len(corrX) + 1))

				popt, pcov = sco.curve_fit(f, sites[start_x:end_x], np.log(corrX[start_x:end_x]))
				perr = np.sqrt(np.diag(pcov))

				a = 1 / popt[0]
				da = a**2 * perr[0]

				print(popt, perr)

				p, = plt.plot(sites, corrX, marker='.', markeredgewidth=1.5, markersize=8, color=COLORS[0], linewidth=0, zorder=100, label=r"$C^{x}$")

				p, = ax.plot(sites[start_x:end_x], np.exp(f(sites[start_x:end_x], *popt)), color=COLORS[2], linewidth=1.5, linestyle='-', zorder=102, label=fr"$C^{{x}}\sim e^{{-s/({np.abs(a):.2f}\pm{da:.2f})}}$")

				def f(x, a, b):
					return a * x + b

				popt, pcov = sco.curve_fit(f, sites[start_z:end_z], np.log(corrZ[start_z:end_z]))
				perr = np.sqrt(np.diag(pcov))

				a = 1 / popt[0]
				da = a**2 * perr[0]

				print(popt, perr)

				p, = plt.plot(sites, corrZ, marker='x', markeredgewidth=1.5, markersize=8, color=COLORS[6], linewidth=0, zorder=101, label=r"$C^{z}$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				
				p, = ax.plot(sites[start_z:end_z], np.exp(f(sites[start_z:end_z], *popt)), color=COLORS[7], linewidth=1.5, linestyle='--', zorder=103, label=fr"$C^{{z}}\sim e^{{-s/({np.abs(a):.1f}\pm{da:.1f})}}$")

				ax.minorticks_on()

				ax.set_xlabel(r'$s$')

				# ax.set_xscale('log')
				ax.set_yscale('log')

				ax.legend(loc="lower left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"corr_{data_filenames[i][:-4]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()
			
				fig, ax = plt.subplots(figsize=(8, 3))

				def f(x, a, b, c, d):
					return a * np.cos(x * b * 2*np.pi + c) + d

				corrX, corrY, corrZ = data['corrs']
				corrZ = connected_correlation(N, corrZ, data['mags'][2, :])

				sites = np.array(range(1, len(corrZ) + 1))

				print(a)
				corrZ = corrZ * [np.exp(s / np.abs(a)) for s in sites]

				popt, pcov = sco.curve_fit(f, sites[start_z_rescale:end_z_rescale], corrZ[start_z_rescale:end_z_rescale], [0.4, 0.2, 1, 0])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = plt.plot(sites[:40], corrZ[:40], marker='.', markeredgewidth=1.5, markersize=6, color=COLORS[6], linewidth=0.5, linestyle='--', zorder=103, label=r"$C^{z}$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				
				p, = ax.plot(sites[start_z_rescale:end_z_rescale], f(sites[start_z_rescale:end_z_rescale], *popt), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$C^{{z}}=\alpha\,\cos\left[2\pi\times({popt[1]:.5f}\pm{perr[1]:.5f})+\beta\right]+\gamma$")

				ax.minorticks_on()

				ax.set_xlabel(r'$s$')

				ax.legend(loc="lower left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"corr_z")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

if __name__ == "__main__":
	main()

