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

plot_mod = True
plot_overlap = False
plot_corr = False
plot_xi = True

start_x = 5
end_x = 50
start_z = 20
end_z = 100
start_z_conn = 1
end_z_conn = 20

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
	simul_name = "LM+TMB/TOP"

	boundary_name = "NONE"
	method_name = "IDMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
					
	folders = ["N=4_chi=256"]
	for folder in folders:
			
		engine_folder = os.path.join(DATA_FOLDER, base_folder, folder)
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
		spectrum = np.zeros((len(lam), len(theta), num_eigen))
		mod = np.zeros((len(lam), len(theta), num_eigen))
		
		overlaps = np.zeros((len(lam), len(theta)))
		components = np.zeros((len(lam), len(theta), 2))

		eta = np.zeros((len(lam), len(theta), 2))

		xi = np.zeros((len(lam), len(theta)))

		for i in range(len(data_filenames)):

			with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			print(data_filenames[i])

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

			if plot_xi:

				xi[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = - N / np.log(np.abs(val_L[1]))

			if plot_mod:

				spectrum[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = np.log10(np.abs(val_L[::-1]))
				mod[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = 1. / N * np.log(val_L[::-1]).imag % (2 * np.pi / N)

			if plot_overlap:

				norm = np.abs(val_L)
				q = 1. / N * np.log(val_L).imag % (2 * np.pi / N)

				print(q[np.where((q > 0.01) & (q < np.pi/2-0.01))])

				x_R = engine.psi.compute_overlap_R(X, vec_R)
				y_R = engine.psi.compute_overlap_R(Y, vec_R)
				z_R = engine.psi.compute_overlap_R(Z, vec_R)
				x_L = engine.psi.compute_overlap_L(X, vec_L)
				y_L = engine.psi.compute_overlap_L(Y, vec_L)
				z_L = engine.psi.compute_overlap_L(Z, vec_L)

				overlap_x = np.abs(x_L[0] * x_R[0])
				overlap_y = np.abs(y_L[0] * y_R[0])
				overlap_z = np.abs(z_L[0] * z_R[0])

				fig, ax = plt.subplots(figsize=(5, 5))

				plt.scatter(q, np.log10(np.abs(x_L[0] * x_R[0])), color=COLORS[1], marker=4, s=50, label=fr"$O^\alpha = S^x,\ O^\beta = S^x$", zorder=100)
				plt.scatter(q, np.log10(np.abs(y_L[0] * y_R[0])), color=COLORS[4], marker=5, s=50, label=fr"$O^\alpha = S^y,\ O^\beta = S^y$", zorder=100)
				plt.scatter(q, np.log10(np.abs(z_L[0] * z_R[0])), color=COLORS[7], marker=6, s=50, label=fr"$O^\alpha = S^z,\ O^\beta = S^z$", zorder=100)
				
				ax.minorticks_on()

				ax.legend(loc="lower center").set_zorder(101)

				plt.xlabel(fr"$q_i$")
				plt.ylabel(fr"$\log_{{10}} |\langle O^\alpha_1 | \psi^\mathrm{{R}}_i \rangle \langle \psi^\mathrm{{L}}_i | O^\beta_1 \rangle|$")

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "OVERLAPS"), data_filenames[i][:-4])
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

				ind = np.argmax(overlap_z)

				components[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = overlap_z[ind]
				components[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = norm[ind]

				overlaps[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = q[ind]

			if plot_corr:

				corrX, corrY, corrZ = data['corrs']

				corrX *= [(-1)**l for l in range(1, len(corrX) + 1)]
				corrZ = np.abs(corrZ)

				fig, ax = plt.subplots(figsize=(8, 3))
				
				def f(x, a, b):
					return a * x + b

				sites = np.array(range(1, len(corrX) + 1))

				popt, pcov = sco.curve_fit(f, np.log(sites[start_x:end_x]), np.log(corrX[start_x:end_x]))
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = popt[0]

				p, = plt.plot(sites, corrX, marker='.', markeredgewidth=1.5, markersize=8, color=COLORS[0], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

				p, = ax.plot(sites[start_x:end_x], np.exp(f(np.log(sites[start_x:end_x]), *popt)), color=COLORS[2], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

				if np.where(vals[i]['lam']==lam)[0][0] <= 19:

					ind = np.where(sites % 2 == 1)[0]

					popt, pcov = sco.curve_fit(f, np.log(sites[ind][start_z:end_z]), np.log(corrZ[ind][start_z:end_z]))
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[0]

					p, = plt.plot(sites[ind], corrZ[ind], marker='x', markeredgewidth=1.5, markersize=8, color=COLORS[6], linewidth=0, zorder=101, label=r"$|C^{z}|$")
					p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
					
					p, = ax.plot(sites[ind][start_z:end_z], np.exp(f(np.log(sites[ind][start_z:end_z]), *popt)), color=COLORS[7], linewidth=1.5, linestyle='--', zorder=103, label=fr"$|C^{{z}}|\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

				ax.minorticks_on()

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

				if np.where(vals[i]['lam']==lam)[0][0] >= 20:
					
					global start_z_conn
					global end_z_conn

					if np.where(vals[i]['lam']==lam)[0][0] >= 45:
						start_z_conn = 8
						end_z_conn = 40

					corrX, corrY, corrZ = data['corrs']
					corrZ = connected_correlation(N, corrZ, data['mags'][2, :])

					fig, ax = plt.subplots(figsize=(8, 3))

					def g(x, a, b, c, d, e):
						return a * x ** b * np.cos(x * c * 2*np.pi + d) + e

					sites = np.array(range(1, len(corrZ) + 1))

					popt, pcov = sco.curve_fit(g, sites[start_z_conn:end_z_conn], corrZ[start_z_conn:end_z_conn], [0.1, -1, 0.4, 1, 0])
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[1]

					p, = plt.plot(sites, corrZ, marker='.', markeredgewidth=1.5, markersize=6, color=COLORS[6], linewidth=0.5, linestyle='--', zorder=103, label=r"$C^{z}$")
					p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
					
					p, = ax.plot(sites[start_z_conn:end_z_conn], g(sites[start_z_conn:end_z_conn], *popt), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$C^{{z}}=\alpha\,s^{{{popt[1]:.2f}\pm{perr[1]:.2f}}}\cos\left[2\pi\times({popt[2]:.5f}\pm{perr[2]:.5f})+\beta\right]+\gamma$")

					ax.minorticks_on()

					ax.set_xlabel(r'$s$')

					ax.legend(loc="lower left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
					
					plt.draw()

					if args.save:
						save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "corr"), f"z_{data_filenames[i][:-4]}")
						fig.clf()

					if args.show:
						plt.show()

					plt.close()
					
		if plot_xi:
		
			for t in range(len(theta)):

				fig, ax = plt.subplots(figsize=(8, 3))

				def f(x, a, b, c):
					return a * np.log(np.abs(x - b)) + c

				popt, pcov = sco.curve_fit(f, lam[22:30], np.log(xi[22:30, t]))
				perr = np.sqrt(np.diag(pcov))

				print(popt, perr)

				p, = ax.plot(lam - popt[1], xi[:, t], marker='.', markersize=8, color=COLORS[0], linewidth=0, zorder=101)

				p, = ax.plot(lam[22:30] - popt[1], np.exp(f(lam[22:30], *popt)), color=COLORS[6], linewidth=1.5, linestyle='-', zorder=100, label=fr"$\xi\sim|\lambda-\lambda^{{\,}}_{{0}}|^{{{popt[0]:.1f}\pm{perr[0]:.1f}}}$")

				ax.set_xlabel(fr"$|\lambda-\lambda^{{\,}}_{{0}}|,\quad \lambda^{{\,}}_{{0}}={popt[1]:.3f}\pm{perr[1]:.3f}$")
				ax.set_ylabel(r"$\xi$")

				ax.minorticks_on()

				# ax.set_xscale('log')
				ax.set_yscale('log')

				ax.legend(loc="lower left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_mod:

			fig, ax = plt.subplots(figsize=(8, 3))

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

			for l in range(len(lam)):
				ind = np.where(mod[l, 0, :])[0]

				x = mod[l, 0, ind]
				y = spectrum[l, 0, ind]
				
				plt.scatter([lam[l] for _ in range(len(x))], x, marker='.', s=8, color=cmap(1 - np.abs(y) / np.max(np.abs(y))), zorder=100)
			
			ax.minorticks_on()

			norm = mpl.colors.Normalize(vmin=np.min(spectrum), vmax=0)
			fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), location="right", ax=ax, label=fr"$\log_{{10}} |t^{{\,}}_{{i}}|$")

			plt.xlabel(fr"$\lambda$")

			plt.ylabel(fr"$\frac{{1}}{{{N}}} \Im \log t^{{\,}}_{{i}} \quad \mathrm{{mod}}\ \frac{{2\pi}}{{{N}}}$")
			
			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"mod")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

		if plot_overlap:
			
			fig, ax = plt.subplots(figsize=(8, 3))

			p, = plt.plot(lam, overlaps[:, 0], linewidth=0, color=COLORS[0], marker='x', markersize=6, markeredgewidth=1.5, zorder=100)
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			axin = ax.inset_axes([0.65, 0.32, 0.33, 0.4])
			axin.set_zorder(101)

			def g(x, a, b, c):
				return a * np.log(x - b) * np.heaviside(x - b, 1) + c

			ind = np.where((lam > 0.31) & (lam < 0.39))[0]

			popt, pcov = sco.curve_fit(g, lam[ind], np.log(overlaps[ind, 0]), [0.5, 0.3, -1])
			perr = np.sqrt(np.diag(pcov))

			print(popt, perr)

			p, = axin.plot(np.log(lam[ind] - popt[1]), np.log(overlaps[ind, 0]), linewidth=0, color=COLORS[0], marker='x', markersize=6, markeredgewidth=1.5, zorder=100)
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			a = np.exp(popt[2])
			da = perr[2] * np.exp(popt[2])

			p, = axin.plot(np.log(lam[ind] - popt[1]), g(lam[ind], *popt), linestyle='-', color=COLORS[7], linewidth=1, zorder=99, marker="none", label=fr"$({format_value(a, da)}) \times \Theta(\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}) \times |\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|^{{{format_value(popt[0], perr[0])}}}$")
			handles = [p]

			ax.minorticks_on()

			ax.set_xlabel(fr"$\lambda,\quad \lambda^{{\,}}_{{\mathrm{{c}}}}={format_value(popt[1], perr[1])}$")
			ax.set_ylabel(fr"$q^{{\,}}_{{i^{{*}}}}$")

			axin.set_xlabel(fr"$\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}$")
			axin.set_ylabel(fr"$\ln q^{{\,}}_{{i^{{*}}}}$")
			
			ax.legend(handles=handles, loc="upper left", ncol=1, labelspacing=0.8, columnspacing=2, handletextpad=0.8, handlelength=1.6).set_zorder(200)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"overlap")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()	

			fig, ax = plt.subplots(figsize=(8, 3))

			p, = plt.plot(lam, components[:, 0, 0], linewidth=0, color=COLORS[0], marker='o', markersize=6, markeredgewidth=1.5, zorder=100, label=f"overlap")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			p, = plt.plot(lam, components[:, 0, 1], linewidth=0, color=COLORS[7], marker='x', markersize=6, markeredgewidth=1.5, zorder=100, label=f"norm")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			ax.minorticks_on()

			ax.set_xlabel(fr"$\lambda$")
			
			ax.legend(loc="center right", ncol=1, labelspacing=0.8, columnspacing=2, handletextpad=0.8, handlelength=1.6).set_zorder(200)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"components")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()	

		if plot_corr:
			
			fig, ax = plt.subplots(figsize=(8, 3))

			p, = plt.plot(lam, eta[:, 0, 0], linewidth=0, color=COLORS[0], marker='o', markersize=6, markeredgewidth=1.5, zorder=100, label=r"$\eta^{x}$")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			p, = plt.plot(lam, eta[:, 0, 1], linewidth=0, color=COLORS[6], marker='x', markersize=6, markeredgewidth=1.5, zorder=100, label=r"$\eta^{z,\mathrm{c}}$")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			ax.minorticks_on()

			ax.set_xlabel(fr"$\lambda$")
			
			ax.legend(loc="center left", ncol=1, labelspacing=0.8, columnspacing=2, handletextpad=0.8, handlelength=1.6).set_zorder(200)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"eta")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()	

if __name__ == "__main__":
	main()

