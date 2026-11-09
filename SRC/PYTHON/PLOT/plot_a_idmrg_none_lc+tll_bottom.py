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

plot_xi = False
plot_spin = False
plot_both = True

eps_norm = 1e-5
num_eigen = 10

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
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
		spins_SPT = np.zeros((len(lam), len(theta), 6))

		for i in range(len(data_filenames)):

			with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			val_L = data['val_L'][:num_eigen]

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]

			if plot_xi or plot_both:

				xi[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = - N / np.log(np.abs(val_L[1]))

			if plot_spin or plot_both:

				spinX, spinY, spinZ = data['mags']
				spinX_SPT, spinY_SPT, spinZ_SPT = data['mags_spt']

				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.abs(np.mean(spinX))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.abs(np.mean(spinY))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.abs(np.mean(spinZ))

				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = np.mean(np.abs(spinX))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 4] = np.mean(np.abs(spinY))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 5] = np.mean(np.abs(spinZ))
				
				spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.abs(np.mean(spinX_SPT))
				spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.abs(np.mean(spinY_SPT))
				spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.abs(np.mean(spinZ_SPT))

				spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = np.mean(np.abs(spinX_SPT))
				spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 4] = np.mean(np.abs(spinY_SPT))
				spins_SPT[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 5] = np.mean(np.abs(spinZ_SPT))
		
		if plot_xi:
		
			for t in range(len(theta)):

				fig, ax = plt.subplots(figsize=(4, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, lam[-55:-34], np.log(xi[-55:-34, t]), [-0.2, 0.788, 1])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)
				print(lam[-55:-34])

				p, = ax.plot(lam, xi[:, t], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				x = np.linspace(0.6, lam[-34-1], 1000)

				p, = ax.plot(x, np.exp(f(x, *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi\sim|\lambda-\lambda^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")

				ax.set_xlabel(fr"$\lambda,\quad \lambda^{{\,}}_{{0}}={format_value(popt[1], perr[1])}$")
				ax.set_ylabel(r"$\xi$")

				ax.legend(loc="upper left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1, borderaxespad=0.3).set_zorder(101)

				ax.axvline(np.pi/4, color="k", linestyle="--")

				ax.minorticks_on()

				ax.set_xlim(0.55, 0.82)
				ax.set_ylim(1, 40)

				axin = ax.inset_axes([0.18, 0.43, 0.55, 0.35])
				axin.set_zorder(101)

				p, = axin.plot(np.abs(lam[:-34] - popt[1]), xi[:-34, t], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = axin.plot(np.abs(x - popt[1]), np.exp(f(x, *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi\sim|\lambda-\lambda^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")

				axin.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{0}}|$")

				axin.set_xscale("log")
				axin.set_yscale("log")

				axin.set_ylim(1, 40)

				axin.set_yticks([1, 10])

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()
			
			for t in range(len(theta)):

				fig, ax = plt.subplots(figsize=(4, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, lam[-33:-23], np.log(xi[-33:-23, t]), [-0.2, 0.78, 1])
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)
				print(lam[-33:-23])

				p, = ax.plot(lam, xi[:, t], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				x = np.linspace(lam[-33], 1, 1000)

				p, = ax.plot(x, np.exp(f(x, *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi\sim|\lambda-\lambda^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")

				ax.set_xlabel(fr"$\lambda,\quad \lambda^{{\,}}_{{0}}={format_value(popt[1], perr[1])}$")
				ax.set_ylabel(r"$\xi$")

				ax.legend(loc="upper left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1, borderaxespad=0.3).set_zorder(101)

				ax.axvline(np.pi/4, color="k", linestyle="--")

				ax.minorticks_on()

				ax.set_xlim(0.55, 0.82)
				ax.set_ylim(1, 40)

				axin = ax.inset_axes([0.18, 0.43, 0.55, 0.35])
				axin.set_zorder(101)

				p, = axin.plot(np.abs(lam[-33:] - popt[1]), xi[-33:, t], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = axin.plot(np.abs(x - popt[1]), np.exp(f(x, *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=110, label=fr"$\xi\sim|\lambda-\lambda^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")

				axin.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{0}}|$")

				axin.set_xscale("log")
				axin.set_yscale("log")

				axin.set_ylim(1, 40)

				axin.set_yticks([1, 10])

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_spt_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_spin:

			handles = []

			for t in range(len(theta)):

				fig, ax = plt.subplots(figsize=(6, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, lam[-100:-33], np.log(spins[-100:-33, t, 3]), [0.2, 0.788, 0])
				perr = np.sqrt(np.diag(pcov))

				print("spin", popt, perr)

				M = np.abs(spins[:, t, 3]) + np.abs(spins_SPT[:, t, 4])

				p, = ax.plot(lam, M, marker='none', color=COLORS[7], linewidth=1.5, linestyle="-", zorder=110, label=r"$|m^{x}_{\mathrm{sta}}|+|m^{\mathrm{SPT},y}_{\mathrm{sta}}|$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				handles.append(p)

				ind = np.where(np.abs(spins[:, t, 3]) > 1E-3)[0]

				p, = ax.plot(lam[ind], M[ind], marker='.', markersize=8, color=COLORS[1], linewidth=0, zorder=101, label=r"$|m^{x}_{\mathrm{sta}}|\neq 0$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				handles.append(p)

				ind = np.where(np.abs(spins_SPT[:, t, 4]) > 1E-3)[0]

				p, = ax.plot(lam[ind], M[ind], marker='x', markersize=5, color=COLORS[4], linewidth=0, zorder=102, label=r"$|m^{\mathrm{SPT},y}_{\mathrm{sta}}|\neq 0$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				handles.append(p)

				ax.axvline(np.pi/4, color="k", linestyle="--", linewidth=1)

				ax.minorticks_on()

				ax.set_xlim(0.58, 1.02)
				ax.set_ylim(0.5, 1)

				x = np.linspace(0.6, lam[-33-1], 1000)

				axin = ax.inset_axes([0.73, 0.3, 0.25, 0.35])
				axin.set_zorder(101)

				p, = axin.plot(np.abs(lam[:-33] - popt[1]), spins[:-33, t, 3], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = axin.plot(np.abs(x - popt[1]), np.exp(f(x, *popt)), color=COLORS[6], linewidth=1, linestyle=':', zorder=110, label=fr"$m^{{x}}_{{\mathrm{{sta}}}}\sim|\lambda-\lambda^{{\,}}_{{0}}|^{{{format_value(popt[0], perr[0])}}}$")
				handles.append(p)

				ax.set_xlabel(fr"$\lambda,\quad \lambda^{{\,}}_{{0}}={format_value(popt[1], perr[1])}$")
				axin.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{0}}|$")

				axin.set_xscale("log")
				axin.set_yscale("log")

				axin.set_ylim(0.4, 5)

				ax.legend(handles=handles, loc="lower left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"op")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_both:

			for t in range(len(theta)):

				fig, ax = plt.subplots(figsize=(3, 2))

				p, = ax.plot(lam, xi[:, t], marker='.', markersize=4, color=COLORS[0], linewidth=0, zorder=101)
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				ax.axvline(np.pi/4, color="k", linestyle="--", linewidth=0.75)

				ax.minorticks_on()

				ax.set_ylim(0, 30)

				ax.set_xticks([0.6, 0.8, 1])

				ax.set_xlabel(fr"$\lambda$")

				ax.set_ylabel(fr"$\xi$")

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()
				
				fig, ax = plt.subplots(figsize=(3, 2))

				p, = ax.plot(lam, np.abs(spins[:, t, 3]) + np.abs(spins_SPT[:, t, 4]), marker='.', markersize=4, color=COLORS[3], linewidth=0, zorder=110, label=r"$|m^{x}_{\mathrm{sta}}|+|m^{\mathrm{SPT},y}_{\mathrm{sta}}|$")
				p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				ax.axvline(np.pi/4, color="k", linestyle="--", linewidth=0.75)

				ax.minorticks_on()

				ax.set_ylim(0.8, 1)

				ax.set_xticks([0.6, 0.8, 1])

				ax.set_xlabel(fr"$\lambda$")

				ax.set_ylabel(r"$|m^{x}_{\mathrm{sta}}|+|m^{\mathrm{SPT},y}_{\mathrm{sta}}|$")

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"order_parameter")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

if __name__ == "__main__":
	main()

