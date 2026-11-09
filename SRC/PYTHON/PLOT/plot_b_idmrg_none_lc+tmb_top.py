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
plot_norm = False
plot_overlap = False
plot_corr = False
plot_xi = False

start_x = 5
end_x = 50
start_z = 20
end_z = 100
start_z_conn = 1
end_z_conn = 20

eps_norm = 10 ** (-2)
num_eigen = 4000

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
	simul_name = "LC+TMB/TOP"

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
		
		spectrum = 10 * np.ones((len(lam), len(theta), num_eigen))
		mod = 10 * np.ones((len(lam), len(theta), num_eigen))
		wave = 10 * np.ones((len(lam), len(theta), num_eigen))
		
		overlaps = 10 * np.ones((len(lam), len(theta), 2))
		components = 10 * np.ones((len(lam), len(theta), 4))

		for i in range(len(data_filenames)):

			with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
				engine = pickle.load(file)

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			print(data_filenames[i])

			val_L = data['val_L'] 

			# val_L, vec_L = data['val_L'], data['vec_L'] 
			# val_R, vec_R = data['val_R'], data['vec_R']

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]
			# sort_R = [find_nearest(val_R, val_L[i]) for i in range(len(val_L))]
			# val_R = val_R[sort_R]

			# vec_L = vec_L[:, sort_L]
			# vec_R = vec_R[:, sort_R]

			val_L = val_L[:num_eigen]
			# val_R = val_R[:num_eigen]
			# vec_L = vec_L[:, :num_eigen]
			# vec_R = vec_R[:, :num_eigen]

			ind = np.where(np.abs(val_L) > eps_norm)[0]
			val_L = val_L[ind]
			# val_R = val_R[ind]
			# vec_L = vec_L[:, ind]
			# vec_R = vec_R[:, ind]

			true_len = len(val_L)

			# for j in range(vec_L.shape[1]):
			# 	overlap = np.abs(np.dot(vec_L[:, j], vec_R[:, j]))
			# 	vec_L[:, j] = vec_L[:, j]/np.sqrt(overlap)
			# 	vec_R[:, j] = vec_R[:, j]/np.sqrt(overlap)

			if plot_xi:

				xi[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = - N / np.log(np.abs(val_L[1]))

			if plot_mod:

				spectrum[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = np.log10(np.abs(val_L[::-1]))
				mod[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = 1. / N * np.log(val_L[::-1]).imag % (2 * np.pi / N)

			if plot_norm:
				
				spectrum[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = np.log10(np.abs(val_L))
				mod[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = 1. / N * np.log(val_L).imag % (2 * np.pi / N)
				wave[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = 1. / N * np.log(val_L).imag

			if plot_overlap:

				norm = np.abs(val_L)
				q = 1. / N * np.log(val_L).imag % (2 * np.pi / N)

				# print(q[np.where((q > 0.1) & (q < np.pi/4))])

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

				ind = np.where((q > 0.1) & (q < np.pi/4))[0]

				_q = q[ind]
				_overlap_z = overlap_z[ind]
				_norm = norm[ind]

				print("overlap", np.log10(_overlap_z[:10]))
				print("norm", np.log10(_norm[:10]))

				if len(ind) != 0:

					ind = np.argmax(_overlap_z)

					components[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = _overlap_z[ind]
					components[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = _norm[ind]

					overlaps[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = _q[ind]

				ind = np.where((q < 0.1) | (q > np.pi/2-0.1))[0]

				_q = q[ind]
				_overlap_z = overlap_z[ind]
				_norm = norm[ind]
				print(ind)

				if len(ind) != 0:

					ind = np.argmax(_overlap_z)

					components[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = _overlap_z[ind]
					components[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = _norm[ind]

					overlaps[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = _q[ind]
					
		if plot_mod:

			fig, ax = plt.subplots(figsize=(8, 3))

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

			t = np.where(theta == np.round(np.pi/4, 15))[0]

			for l in range(len(lam)):
				ind = np.where(mod[l, t, :] != 10)[1]

				x = mod[l, t, ind]
				y = spectrum[l, t, ind]
				
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

		if plot_norm:

			fig, ax = plt.subplots(figsize=(5, 4))
		
			t = np.where(theta == np.round(np.pi/4, 15))[0]

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[8],COLORS[0],COLORS[5],COLORS[5],COLORS[0],COLORS[8]], N=256)
			
			for l in range(len(lam)):
				ind = np.where(mod[l, t, :] != 10)[1]

				x = mod[l, t, ind]
				y = spectrum[l, t, ind]
				z = wave[l, t, ind]

				sort = np.argsort(z)[::-1]

				plt.scatter([lam[l] for _ in range(len(y))], np.array(y[sort]), marker='.', color=cmap(np.abs(x[sort]) / np.max(x[sort])), zorder=100)

			ax.minorticks_on()

			norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi / N)
			fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), location="right", ax=ax, label=fr"$q_i$")
			
			plt.xlabel(fr"$\lambda$")

			plt.ylabel(fr"$\log_{{10}} |t_i|$")
			
			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"norm")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

		if plot_overlap:

			# fig, ax = plt.subplots(figsize=(8, 3))

			# p, = plt.plot(lam, overlaps[:, t], linewidth=0, color=COLORS[0], marker='x', markersize=6, markeredgewidth=1.5, zorder=100)
			# p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			# axin = ax.inset_axes([0.65, 0.32, 0.33, 0.4])
			# axin.set_zorder(101)

			# def g(x, a, b, c):
			# 	return a * np.log(x - b) * np.heaviside(x - b, 1) + c

			# ind = np.where((lam > 0.31) & (lam < 0.39))[0]

			# popt, pcov = sco.curve_fit(g, lam[ind], np.log(overlaps[ind, t]), [0.5, 0.3, -1])
			# perr = np.sqrt(np.diag(pcov))

			# print(popt, perr)

			# p, = axin.plot(np.log(lam[ind] - popt[1]), np.log(overlaps[ind, 0]), linewidth=0, color=COLORS[0], marker='x', markersize=6, markeredgewidth=1.5, zorder=100)
			# p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			# a = np.exp(popt[2])
			# da = perr[2] * np.exp(popt[2])

			# p, = axin.plot(np.log(lam[ind] - popt[1]), g(lam[ind], *popt), linestyle='-', color=COLORS[7], linewidth=1, zorder=99, marker="none", label=fr"$({format_value(a, da)}) \times \Theta(\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}) \times |\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|^{{{format_value(popt[0], perr[0])}}}$")
			# handles = [p]

			# ax.minorticks_on()

			# ax.set_xlabel(fr"$\lambda,\quad \lambda^{{\,}}_{{\mathrm{{c}}}}={format_value(popt[1], perr[1])}$")
			# ax.set_ylabel(fr"$q^{{\,}}_{{i^{{*}}}}$")

			# axin.set_xlabel(fr"$\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}$")
			# axin.set_ylabel(fr"$\ln q^{{\,}}_{{i^{{*}}}}$")
			
			# ax.legend(handles=handles, loc="upper left", ncol=1, labelspacing=0.8, columnspacing=2, handletextpad=0.8, handlelength=1.6).set_zorder(200)

			# plt.draw()

			# if args.save:
			# 	save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"overlap")
			# 	fig.clf()

			# if args.show:
			# 	plt.show()

			# plt.close()	

			fig, ax = plt.subplots(figsize=(8, 3))

			t = np.where(theta == np.round(np.pi/4, 15))[0]

			ind = np.where(overlaps[:, t, 0] != 10)[0]

			_lam = lam[ind]
			_overlaps = overlaps[ind, :, :]
			_components = components[ind, :, :]

			p, = plt.plot(_lam, np.log10(_components[:, t, 0]), linewidth=0, color=COLORS[0], marker='o', markersize=6, markeredgewidth=1.5, zorder=100, label=r"$\log|w^{\,}_{i^{*}}|$")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			p, = plt.plot(_lam, np.log10(_components[:, t, 1]), linewidth=0, color=COLORS[2], marker='x', markersize=6, markeredgewidth=1.5, zorder=100, label=r"$\log|t^{\,}_{i^{*}}|$")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			ind = np.where(overlaps[:, t, 1] != 10)[0]

			_lam = lam[ind]
			_overlaps = overlaps[ind, :, :]
			_components = components[ind, :, :]

			ind = np.where(_lam >= 1.85)[0]

			_lam = lam[ind]
			_overlaps = overlaps[ind, :, :]
			_components = components[ind, :, :]


			p, = plt.plot(_lam, np.log10(_components[:, t, 2]), linewidth=0, color=COLORS[3], marker='+', markersize=6, markeredgewidth=1.5, zorder=100, label=r"$\log|w^{\,}_{i^{**}}|$")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			p, = plt.plot(_lam, np.log10(_components[:, t, 3]), linewidth=0, color=COLORS[5], marker='v', markersize=6, markeredgewidth=1.5, zorder=100, label=r"$\log|t^{\,}_{i^{**}}|$")
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

if __name__ == "__main__":
	main()

