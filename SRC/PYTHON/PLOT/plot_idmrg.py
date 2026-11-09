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

def connected_correlation(N, corr, spin):

	r = (len(corr) + 1) // N
	spin_extended = np.concatenate([spin for _ in range(r)], axis=0)

	conn = corr - spin_extended[0] * spin_extended[1:] 
	return conn

su2 = SUIrrep(2, 2)
X, Y, Z = [2 * su2.generators[i] for i in range(3)]

Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

plot_xi = False
plot_q_mod = False
plot_norm_transfer = False
plot_ratio = False
plot_overlap = False
plot_corr = True
plot_spin = False
plot_spin_spt = False
plot_complex = False

eps_norm = 10**(-4)
num_eigen = 50

start_z_rescale = 10
end_z_rescale = 100

plt.rcParams.update({
    'savefig.format': 'png'
})

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "B"
	simul_name = "PC+TEB/BOTTOM"

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
		spectrum_log = np.zeros((len(lam), len(theta), num_eigen))
		q_mod = np.zeros((len(lam), len(theta), num_eigen))
		q = np.zeros((len(lam), len(theta), num_eigen))
		ang = np.zeros((len(lam), len(theta), 3, num_eigen))
		spins = np.zeros((len(lam), len(theta), 6))
		spins_spt = np.zeros((len(lam), len(theta), 6))
		comp = np.zeros((len(lam), len(theta), num_eigen), dtype=np.complex128)

		for i in range(len(data_filenames)):

			print(data_filenames[i])

			# with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
			# 	engine = pickle.load(file)
			# 	print(engine.num_sweeps, engine.dE[-1], engine.dS[-1])

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			# val_L = data['val_L'][:num_eigen]

			# # # val_L, vec_L = data['val_L'], data['vec_L'] 
			# # # val_R, vec_R = data['val_R'], data['vec_R']

			# sort_L = np.argsort(np.abs(val_L))[::-1]
			# val_L = val_L[sort_L]
			# # # sort_R = [find_nearest(val_R, val_L[i]) for i in range(len(val_L))]
			# # # val_R = val_R[sort_R]

			# # # vec_L = vec_L[:, sort_L]
			# # # vec_R = vec_R[:, sort_R]

			# # val_L = val_L[:num_eigen]
			# # # val_R = val_R[:num_eigen]
			# # # vec_L = vec_L[:, :num_eigen]
			# # # vec_R = vec_R[:, :num_eigen]

			# # ind = np.where(np.abs(val_L) > eps_norm)[0]
			# # val_L = val_L[ind]
			# # # val_R = val_R[ind]
			# # # vec_L = vec_L[:, ind]
			# # # vec_R = vec_R[:, ind]
			
			# true_len = len(val_L)

			# # # for j in range(vec_L.shape[1]):
			# # # 	overlap = np.abs(np.dot(vec_L[:, j], vec_R[:, j]))
			# # # 	vec_L[:, j] = vec_L[:, j]/np.sqrt(overlap)
			# # # 	vec_R[:, j] = vec_R[:, j]/np.sqrt(overlap)

			if plot_complex:

				comp[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = val_L

			if plot_xi:

				xi[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = - N / np.log(np.abs(val_L[1]))

			if plot_q_mod:
				
				spectrum_log[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = np.log10(np.abs(val_L))
				q_mod[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = 1. / N * np.log(val_L).imag % (2 * np.pi / N)

			if plot_norm_transfer:
				
				spectrum_log[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = np.log10(np.abs(val_L))
				q_mod[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = 1. / N * np.log(val_L).imag % (2 * np.pi / N)
				q[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = 1. / N * np.log(val_L).imag
				# print(1. / N * np.log(val_L[:10]).imag % (2 * np.pi / N))

			if plot_ratio:

				with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
					engine = pickle.load(file)

				x_R = engine.psi.compute_overlap_R(X, vec_R)
				y_R = engine.psi.compute_overlap_R(Y, vec_R)
				z_R = engine.psi.compute_overlap_R(Z, vec_R)
				x_L = engine.psi.compute_overlap_L(X, vec_L)
				y_L = engine.psi.compute_overlap_L(Y, vec_L)
				z_L = engine.psi.compute_overlap_L(Z, vec_L)
				
				x = x_R[0]/x_R[1]
				x = x[np.argsort(np.abs(x_R[0] * x_L[0]))[::-1]]
				ang[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0, :true_len] = np.angle(x)

				y = y_R[0]/y_R[1]
				y = y[np.argsort(np.abs(y_R[0] * y_L[0]))[::-1]]
				ang[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1, :true_len] = np.angle(y)

				z = z_R[0]/z_R[1]
				z = z[np.argsort(np.abs(z_R[0] * z_L[0]))[::-1]]
				ang[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2, :true_len] = np.angle(z)

				fig, ax = plt.subplots(figsize=(5, 5))

				plt.scatter(range(len(x)), np.angle(x), color=COLORS[0], marker=4, s=50, label=fr"$O^\alpha = S^x,\ O^\beta = S^x$", zorder=100)
				plt.scatter(range(len(y)), np.angle(y), color=COLORS[2], marker=5, s=50, label=fr"$O^\alpha = S^y,\ O^\beta = S^y$", zorder=100)
				plt.scatter(range(len(z)), np.angle(z), color=COLORS[4], marker=6, s=50, label=fr"$O^\alpha = S^z,\ O^\beta = S^z$", zorder=100)
				
				ax.minorticks_on()

				ax.legend(loc="lower center").set_zorder(101)

				plt.xlabel(fr"$q_i$")
				plt.ylabel(fr"$\log_{{10}} |\langle O^\alpha_1 | \psi^\mathrm{{R}}_i \rangle \langle \psi^\mathrm{{L}}_i | O^\beta_1 \rangle|$")

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "q_overlap_"), data_filenames[i][:-4])
					fig.clf()

				if args.show:
					plt.show()

				plt.close()
			
			if plot_overlap:

				with open(os.path.join(engine_folder, data_filenames[i]), 'rb') as file:
					engine = pickle.load(file)

				x_R = engine.psi.compute_overlap_R(X, vec_R)
				y_R = engine.psi.compute_overlap_R(Y, vec_R)
				z_R = engine.psi.compute_overlap_R(Z, vec_R)
				x_L = engine.psi.compute_overlap_L(X, vec_L)
				y_L = engine.psi.compute_overlap_L(Y, vec_L)
				z_L = engine.psi.compute_overlap_L(Z, vec_L)
				
				q_mod_one = 1. / N * np.log(val_L).imag % (2 * np.pi / N)

				fig, ax = plt.subplots(figsize=(5, 5))

				plt.scatter(q_mod_one, np.log10(np.abs(x_L[0] * x_R[0])), color=COLORS[1], marker=4, s=50, label=fr"$O^\alpha = S^x,\ O^\beta = S^x$", zorder=100)
				plt.scatter(q_mod_one, np.log10(np.abs(y_L[0] * y_R[0])), color=COLORS[4], marker=5, s=50, label=fr"$O^\alpha = S^y,\ O^\beta = S^y$", zorder=100)
				plt.scatter(q_mod_one, np.log10(np.abs(z_L[0] * z_R[0])), color=COLORS[7], marker=6, s=50, label=fr"$O^\alpha = S^z,\ O^\beta = S^z$", zorder=100)
				
				ax.minorticks_on()

				ax.legend(loc="lower center").set_zorder(101)

				plt.xlabel(fr"$q_i$")
				plt.ylabel(fr"$\log_{{10}} |\langle O^\alpha_1 | \psi^\mathrm{{R}}_i \rangle \langle \psi^\mathrm{{L}}_i | O^\beta_1 \rangle|$")

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "q_overlap"), data_filenames[i][:-4])
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			if plot_corr:

				corrX, corrY, corrZ = data['corrs']
				# corrX, corrY, corrZ = corrX[:30], corrY[:30], corrZ[:30]
				
				# corrX = np.abs(connected_correlation(N, corrX, data['mags'][0, :]))
				corrX = np.abs(corrX)
				corrY = np.abs(corrY)
				corrZ = np.abs(corrZ)
				# corrZ = np.abs(connected_correlation(N, corrZ, data['mags'][2, :]))
			
				fig, ax = plt.subplots(figsize=(5, 4))

				plt.plot(range(1, len(corrX)+1), corrX, color=COLORS[0], marker=4, label=fr"$O^\alpha = S^x,\ O^\beta = S^x$", zorder=100)
				plt.plot(range(1, len(corrY)+1), corrY, color=COLORS[2], marker=5, label=fr"$O^\alpha = S^y,\ O^\beta = S^y$", zorder=100)
				plt.plot(range(1, len(corrZ)+1), corrZ, color=COLORS[4], marker=6, label=fr"$O^\alpha = S^z,\ O^\beta = S^z$", zorder=100)

				ax.minorticks_on()

				ax.set_xscale('log')
				ax.set_yscale('log')

				ax.legend(loc='center right').set_zorder(101)

				plt.xlabel(fr"$x$")
				plt.ylabel(fr"$\langle O^\alpha_1 O^\beta_x \rangle \mathrm e^{{\frac{{x-1}}{{\xi}}}}$")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "corr"), data_filenames[i][:-4])
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

				# fig, ax = plt.subplots(figsize=(8, 3))

				# def f(x, a, b, c, d, e):
				# 	return a * np.cos(x * (b * 2*np.pi) + c) * x ** (-d) + e

				# corrX, corrY, corrZ = data['corrs']
				# corrZ = connected_correlation(N, corrZ, data['mags'][2, :])

				# sites = np.array(range(1, len(corrZ) + 1))

				# # corrZ = corrZ * [np.exp(-s * np.log(np.abs(val_L[1])) / N) for s in sites]

				# try:
				# 	popt, pcov = sco.curve_fit(f, sites[start_z_rescale:end_z_rescale], corrZ[start_z_rescale:end_z_rescale], [0.4, 0.4, 1, 1, 0])
				# 	perr = np.sqrt(np.diag(pcov))

				# 	print(popt, perr)

				# 	p, = ax.plot(sites[start_z_rescale:end_z_rescale], f(sites[start_z_rescale:end_z_rescale], *popt), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$\xi={1/popt[3]},\ q={popt[1]}$")
				# except:
				# 	pass
				
				# p, = plt.plot(sites, corrZ, marker='o', markersize=6, color=COLORS[6], linewidth=0.5, linestyle='--', zorder=101, label=r"$C^{z}$")
				# p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				# ax.minorticks_on()

				# ax.set_xlabel(r'$s$')

				# ax.legend(loc="lower left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
				
				# plt.draw()

				# if args.save:
				# 	save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "CORR_Z"), f"{data_filenames[i][:-4]}")
				# 	fig.clf()

				# if args.show:
				# 	plt.show()

				# plt.close()
					
				# fig, ax = plt.subplots(figsize=(5, 4))
				
				# def f(x, a, b):
				# 	return a * x + b

				# sites = range(1, len(corrX)+1)

				# popt, pcov = sco.curve_fit(f, np.log(sites[10:-50]), np.log(np.abs(corrX[10:-50])))
				# perr = np.sqrt(np.diag(pcov))

				# print(popt, perr)

				# p, = ax.plot(sites[10:], np.exp(f(np.log(sites[10:]), *popt)), linewidth=1, zorder=105)

				# plt.plot(sites, corrX, color=COLORS[1], marker=4, zorder=100)

				# ax.minorticks_on()

				# ax.set_xscale('log')
				# ax.set_yscale('log')

				# ax.legend(loc='center right').set_zorder(101)
				
				# plt.draw()

				# if args.save:
				# 	save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "corr"), data_filenames[i][:-4])
				# 	fig.clf()

				# if args.show:
				# 	plt.show()

				# plt.close()

			if plot_spin:

				spinX, spinY, spinZ = data['mags']

				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.abs(np.mean(spinX))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.abs(np.mean(spinY))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.abs(np.mean(spinZ))

				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = np.mean(np.abs(spinX))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 4] = np.mean(np.abs(spinY))
				spins[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 5] = np.mean(np.abs(spinZ))
			
				fig, ax = plt.subplots(figsize=(5, 4))

				plt.plot(range(1, N+1), spinX, color=COLORS[1], marker=4, label=fr"$O^\alpha = S^x,\ O^\beta = S^x$", zorder=100)
				plt.plot(range(1, N+1), spinY, color=COLORS[4], marker=5, label=fr"$O^\alpha = S^y,\ O^\beta = S^y$", zorder=100)
				plt.plot(range(1, N+1), spinZ, color=COLORS[6], marker=6, label=fr"$O^\alpha = S^z,\ O^\beta = S^z$", zorder=100)

				ax.minorticks_on()

				ax.legend(loc='center right').set_zorder(101)

				plt.xlabel(fr"$l$")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "spin"), data_filenames[i][:-4])
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			if plot_spin_spt:

				spinX, spinY, spinZ = data['mags_spt']

				spins_spt[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.abs(np.mean(spinX))
				spins_spt[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.abs(np.mean(spinY))
				spins_spt[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.abs(np.mean(spinZ))

				spins_spt[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 3] = np.mean(np.abs(spinX))
				spins_spt[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 4] = np.mean(np.abs(spinY))
				spins_spt[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 5] = np.mean(np.abs(spinZ))
			
				fig, ax = plt.subplots(figsize=(5, 4))

				plt.plot(range(1, N-1), spinX, color=COLORS[1], marker=4, label=fr"$O^\alpha = S^x,\ O^\beta = S^x$", zorder=100)
				plt.plot(range(1, N-1), spinY, color=COLORS[4], marker=5, label=fr"$O^\alpha = S^y,\ O^\beta = S^y$", zorder=100)
				plt.plot(range(1, N-1), spinZ, color=COLORS[7], marker=6, label=fr"$O^\alpha = S^z,\ O^\beta = S^z$", zorder=100)

				ax.minorticks_on()

				ax.legend(loc='center right').set_zorder(101)

				plt.xlabel(fr"$l$")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "spin_spt"), data_filenames[i][:-4])
					fig.clf()

				if args.show:
					plt.show()

				plt.close()
		
		if plot_xi:
		
			fig, ax = plt.subplots(figsize=(5, 4))

			# def f(x, a, b, c):
			# 	return a * np.log(np.abs(x - b)) + c

			# popt, pcov = sco.curve_fit(f, lam[22:30], np.log(xi[22:30, 0]))
			# perr = np.sqrt(np.diag(pcov))

			# print(popt, perr)

			p, = ax.plot(np.abs(theta), xi[0, :], marker='.',  linewidth=0, zorder=105)

			# p, = ax.plot(np.abs(lam[22:30] - popt[1]), np.exp(f(lam[22:30], *popt)), linewidth=1, zorder=105)

			ax.set_xscale('log')
			ax.set_yscale('log')

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()		

			fig, ax = plt.subplots(figsize=(5, 4))

			# def f(x, a, b, c):
			# 	return a * np.log(np.abs(x - b)) + c

			# popt, pcov = sco.curve_fit(f, lam[22:30], np.log(xi[22:30, 0]))
			# perr = np.sqrt(np.diag(pcov))

			# print(popt, perr)

			p, = ax.plot(np.abs(lam), xi[:, 0], marker='.',  linewidth=0, zorder=105)

			# p, = ax.plot(np.abs(lam[22:30] - popt[1]), np.exp(f(lam[22:30], *popt)), linewidth=1, zorder=105)

			ax.set_xscale('log')
			ax.set_yscale('log')

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"xi_")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()
		
		if plot_spin:
		
			fig, ax = plt.subplots(figsize=(5, 4))

			def f(x, a, b, c):
				return a * np.log(np.abs(x - b)) + c

			# popt, pcov = sco.curve_fit(f, theta[22:28], np.log(spins[0, 22:28, 0]), [1, 1, 5])
			# perr = np.sqrt(np.diag(pcov))

			# print(popt, perr)

			p, = ax.plot(np.abs(theta[:]), spins[0, :, 0], marker='.',  linewidth=0, zorder=105, label="x_uni")
			p, = ax.plot(np.abs(theta[:]), spins[0, :, 1], marker='.',  linewidth=0, zorder=105, label="y_uni")
			p, = ax.plot(np.abs(theta[:]), spins[0, :, 2], marker='.',  linewidth=0, zorder=105, label="z_uni")
			p, = ax.plot(np.abs(theta[:]), spins[0, :, 3], marker='.',  linewidth=0, zorder=105, label="x_sta")
			p, = ax.plot(np.abs(theta[:]), spins[0, :, 4], marker='.',  linewidth=0, zorder=105, label="y_sta")
			p, = ax.plot(np.abs(theta[:]), spins[0, :, 5], marker='.',  linewidth=0, zorder=105, label="z_sta")

			ax.legend()

			# p, = ax.plot(np.abs(theta[22:28] - popt[1]), np.exp(f(theta[22:28], *popt)), linewidth=1, zorder=105)

			# ax.set_xscale('log')
			# ax.set_yscale('log')
				
			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"spin_lam")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()		

			fig, ax = plt.subplots(figsize=(5, 4))

			def f(x, a, b, c):
				return a * np.log(np.abs(x - b)) + c

			# popt, pcov = sco.curve_fit(f, lam[22:28], np.log(spins[22:28, 0, 0]), [1, 1, 5])
			# perr = np.sqrt(np.diag(pcov))

			# print(popt, perr)

			p, = ax.plot(np.abs(lam[:]), spins[:, 0, 2], marker='.',  linewidth=0, zorder=105)

			# p, = ax.plot(np.abs(lam[22:28] - popt[1]), np.exp(f(lam[22:28], *popt)), linewidth=1, zorder=105)

			# ax.set_xscale('log')
			# ax.set_yscale('log')
				
			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"spin_lam_")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()
	
		if plot_spin_spt:
		
			fig, ax = plt.subplots(figsize=(5, 4))

			def f(x, a, b, c):
				return a * np.log(np.abs(x - b)) + c

			# popt, pcov = sco.curve_fit(f, theta[22:28], np.log(spins[0, 22:28, 0]), [1, 1, 5])
			# perr = np.sqrt(np.diag(pcov))

			# print(popt, perr)

			p, = ax.plot(np.abs(theta[:]), spins_spt[0, :, 0], marker='.',  linewidth=0, zorder=105, label="x_spt_uni")
			p, = ax.plot(np.abs(theta[:]), spins_spt[0, :, 1], marker='.',  linewidth=0, zorder=105, label="y_spt_uni")
			p, = ax.plot(np.abs(theta[:]), spins_spt[0, :, 2], marker='.',  linewidth=0, zorder=105, label="z_spt_uni")
			p, = ax.plot(np.abs(theta[:]), spins_spt[0, :, 3], marker='.',  linewidth=0, zorder=105, label="x_spt_sta")
			p, = ax.plot(np.abs(theta[:]), spins_spt[0, :, 4], marker='.',  linewidth=0, zorder=105, label="y_spt_sta")
			p, = ax.plot(np.abs(theta[:]), spins_spt[0, :, 5], marker='.',  linewidth=0, zorder=105, label="z_spt_sta")

			ax.legend()

			# p, = ax.plot(np.abs(theta[22:28] - popt[1]), np.exp(f(theta[22:28], *popt)), linewidth=1, zorder=105)

			# ax.set_xscale('log')
			# ax.set_yscale('log')
				
			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"spin_lam_spt")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()		

		if plot_q_mod:

			for t in range(len(theta)):

				fig, ax = plt.subplots(figsize=(5, 4))

				cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[6],COLORS[0],COLORS[5],COLORS[3]], N=256)

				mins = []
				for l in range(len(lam)):
					plt.scatter([lam[l] for _ in range(len(q_mod[l, t, :]))], q_mod[l, t, :], marker='.', s=5, color=cmap(1 - np.abs(spectrum_log[l, t, :]) / np.max(np.abs(spectrum_log[l, t, :]))), zorder=100)
					mins.append(np.min(spectrum_log[l, t, :]))
				
				ax.minorticks_on()

				# ax.legend(loc='center left').set_zorder(101)

				norm = mpl.colors.Normalize(vmin=np.min(mins), vmax=0)
				fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), location="right", ax=ax, label=fr"$\log_{{10}} |t_i|$")

				plt.xlabel(fr"$\theta$")

				plt.ylabel(fr"$\frac{{1}}{{{N}}} \Im \log t_i \quad \mathrm{{mod}}\ \frac{{2\pi}}{{{N}}}$")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"q_mod_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			for l in range(len(lam)):

				fig, ax = plt.subplots(figsize=(5, 4))

				cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[5],COLORS[6],COLORS[0],COLORS[4],COLORS[3]], N=256)

				mins = []
				for t in range(len(theta)):
					plt.scatter([theta[t] for _ in range(len(q_mod[l, t, :]))], q_mod[l, t, :], marker='.', s=5, color=cmap(1 - np.abs(spectrum_log[l, t, :]) / np.max(np.abs(spectrum_log[l, t, :]))), zorder=100)
					mins.append(np.min(spectrum_log[l, t, :]))
				
				ax.minorticks_on()

				# ax.legend(loc='center left').set_zorder(101)

				norm = mpl.colors.Normalize(vmin=np.min(mins), vmax=0)
				fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), location="right", ax=ax, label=fr"$\log_{{10}} |t_i|$")

				plt.xlabel(fr"$\theta$")

				plt.ylabel(fr"$\frac{{1}}{{{N}}} \Im \log t_i \quad \mathrm{{mod}}\ \frac{{2\pi}}{{{N}}}$")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"q_mod_lam={lam[l]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_norm_transfer:

			for l in range(len(lam)):
			
				fig, ax = plt.subplots(figsize=(5, 4))

				cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)
				
				for t in range(len(theta)):
					sort = np.argsort(q[l, t, :])[::-1]
					q_mod_sorted = np.array(q_mod[l, t, :])[sort]
					plt.scatter([theta[t] for _ in range(len(q_mod[l, t, :]))], np.array(spectrum_log[l, t, :])[sort], marker='.', color=cmap(np.abs(q_mod_sorted) / np.max(q_mod_sorted)), zorder=100)

				ax.minorticks_on()

				# ax.legend(loc="center right").set_zorder(101)

				norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi / N)
				fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), location="right", ax=ax, label=fr"$q_i$")
				
				plt.xlabel(fr"$\theta$")

				plt.ylabel(fr"$\log_{{10}} |t_i|$")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"norm_lam={lam[l]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			for t in range(len(theta)):
			
				fig, ax = plt.subplots(figsize=(5, 4))

				cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[8],COLORS[0],COLORS[5],COLORS[5],COLORS[0],COLORS[8]], N=256)
				
				for l in range(len(lam)):
					sort = np.argsort(q[l, t, :])[::-1]
					q_mod_sorted = np.array(q_mod[l, t, :])[sort]
					plt.scatter([lam[l] for _ in range(len(q_mod[l, t, :]))], np.array(spectrum_log[l, t, :])[sort], marker='.', color=cmap(np.abs(q_mod_sorted) / np.max(q_mod_sorted)), zorder=100)
					if np.max(np.array(spectrum_log[l, t, :])[sort]) > 1:
						print("********************", lam[l])

				ax.minorticks_on()

				# ax.legend(loc="center right").set_zorder(101)

				norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi / N)
				fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), location="right", ax=ax, label=fr"$q_i$")
				
				plt.xlabel(fr"$\lambda$")

				plt.ylabel(fr"$\log_{{10}} |t_i|$")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"norm_theta={theta[t]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_ratio:

			for i in range(10):

				fig, ax = plt.subplots(figsize=(5, 4))


				plt.scatter(theta, np.abs(ang[0, :, 0, i]), color=COLORS[0], marker=4, s=50, label=fr"$O = S^x$", zorder=100)
				plt.scatter(theta, np.abs(ang[0, :, 1, i]), color=COLORS[4], marker=5, s=50, label=fr"$O = S^y$", zorder=100)
				plt.scatter(theta, np.abs(ang[0, :, 2, i]), color=COLORS[7], marker=6, s=50, label=fr"$O = S^z$", zorder=100)
				
				ax.minorticks_on()

				ax.legend(loc='lower left').set_zorder(101)
			
				# ax.set_ylim(2.15, 3.2)

				plt.xlabel(fr"$\theta$")
				plt.ylabel(fr"$\left| \arg \frac{{\langle O_2 | \psi^\mathrm{{R}}_{{i^*}} \rangle}}{{\langle O_1 | \psi^\mathrm{{R}}_{{i^*}} \rangle}} \right|, \quad i^* = \mathrm{{argmax}}_i \langle O_1 | \psi^\mathrm{{R}}_i \rangle $")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "q_ratio"), f"q_ratio_{i}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			for i in range(10):

				fig, ax = plt.subplots(figsize=(5, 4))

				plt.scatter(lam, np.abs(ang[:, 0, 0, i]), color=COLORS[0], marker=4, s=50, label=fr"$O = S^x$", zorder=100)
				plt.scatter(lam, np.abs(ang[:, 0, 1, i]), color=COLORS[4], marker=5, s=50, label=fr"$O = S^y$", zorder=100)
				plt.scatter(lam, np.abs(ang[:, 0, 2, i]), color=COLORS[7], marker=6, s=50, label=fr"$O = S^z$", zorder=100)
				
				ax.minorticks_on()

				ax.legend(loc='lower left').set_zorder(101)
			
				# ax.set_ylim(2.15, 3.2)

				plt.xlabel(fr"$\theta$")
				plt.ylabel(fr"$\left| \arg \frac{{\langle O_2 | \psi^\mathrm{{R}}_{{i^*}} \rangle}}{{\langle O_1 | \psi^\mathrm{{R}}_{{i^*}} \rangle}} \right|, \quad i^* = \mathrm{{argmax}}_i \langle O_1 | \psi^\mathrm{{R}}_i \rangle $")
				
				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "q_ratio_"), f"q_ratio_{i}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

		if plot_complex:
		
			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)
			plt.rcParams.update({'axes.prop_cycle': (cycler(color=cmap(np.linspace(0, 1, len(lam)))))})
		
			fig, ax = plt.subplots(figsize=(8, 5))

			for l in range(len(lam)):
				ind = [2, 3, 4, 5]
				plt.scatter(comp[l, 0, ind].real, comp[l, 0, ind].imag, marker='.', zorder=100)

			norm = mpl.colors.Normalize(vmin=np.min(lam), vmax=np.max(lam))
			fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), location="right", ax=ax, label=fr"$\lambda$")

			plt.xlabel(fr"$\Re t$")
			plt.ylabel(fr"$\Im t$")
			
			# ax.set_xscale('log')

			ax.minorticks_on()

			axin = ax.inset_axes([0.2, 0.3, 0.3, 0.2])
			
			for l in range(len(lam)):
				ind = [2, 3, 4, 5]
				axin.scatter(comp[l, 0, ind].real, comp[l, 0, ind].imag, marker='.', zorder=100)

			axin.set_xlim(0.93, 1)
			axin.set_ylim(-0.1, 0.1)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"complex")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

if __name__ == "__main__":
	main()

