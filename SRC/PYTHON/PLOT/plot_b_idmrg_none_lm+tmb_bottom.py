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
plot_overlap = True
plot_corr = False

start_x = 40
end_x = 100
start_z = 40
end_z = 100
start_z_conn = 1
end_z_conn = 1

eps_norm = 10**(-6e-1)
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
	simul_name = "LM+TMB/BOTTOM"

	boundary_name = "NONE"
	method_name = "IDMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
					
	folders = ["N=10_chi=256", "N=4_chi=256"]
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

		if N == 4:
			continue

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

			if plot_mod:

				spectrum[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = np.log10(np.abs(val_L[::-1]))
				mod[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :true_len] = 1. / N * np.log(val_L[::-1]).imag % (2 * np.pi / N)

			if plot_corr:

				corrX, corrY, corrZ = data['corrs']

				corrX = np.abs(corrX)
				corrY = np.abs(connected_correlation(N, corrY, data['mags'][1, :]))
				corrZ = np.abs(connected_correlation(N, corrZ, data['mags'][2, :]))

				fig, ax = plt.subplots(figsize=(8, 3))
				
				def f(x, a, b):
					return a * x + b
				
				if vals[i]['lam'] >= 0:
				# if vals[i]['lam'] < 0.29:

					sites = np.array(range(1, len(corrY) + 1))

					popt, pcov = sco.curve_fit(f, np.log(sites[start_x:end_x]), np.log(corrY[start_x:end_x]))
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = popt[0]

					p, = plt.plot(sites, corrY, marker='o', markersize=8, color=COLORS[0], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

					p, = ax.plot(sites[start_x:end_x], np.exp(f(np.log(sites[start_x:end_x]), *popt)), color=COLORS[2], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

					sites = np.array(range(1, len(corrZ) + 1))

					ind = np.where(sites % 2 == 1)[0]

					popt, pcov = sco.curve_fit(f, np.log(sites[ind][start_z:end_z]), np.log(corrZ[ind][start_z:end_z]))
					perr = np.sqrt(np.diag(pcov))

					print(popt, perr)

					eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[0]

					p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=101, label=r"$|C^{z}|$")
					
					p, = ax.plot(sites[ind][start_z:end_z], np.exp(f(np.log(sites[ind][start_z:end_z]), *popt)), color=COLORS[5], linewidth=1.5, linestyle='--', zorder=103, label=fr"$|C^{{z}}|\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")


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

			if plot_overlap:

				norm = np.abs(val_L)
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

				if i <= 19:
					
					ind = np.where((q < 0.005))[0]
					q = q[ind]

				elif i <= 22: 

					ind = np.where((q > 0.01) & (q < 2*np.pi/20 - 0.001))[0]
					q = q[ind]

				elif i >= 23:

					ind = np.where((q > 2*np.pi/20 + 0.001) & (q < 2*np.pi/10 - 0.001))[0] 
					q = q[ind]

				overlaps[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = q[np.argmax(overlap_z[ind])]

		if plot_mod:

			fig, ax = plt.subplots(figsize=(4.5, 2))

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[4],COLORS[5],COLORS[3],COLORS[2]], N=256)

			for l in range(len(lam)):

				plt.scatter([lam[l] for _ in range(len(mod[l, 0, :]))], mod[l, 0, :], marker='o', s=12, linewidths=0.1, edgecolors="1", color=cmap(1 - np.abs(spectrum[l, 0, :]) / np.max(np.abs(spectrum[l, 0, :]))), zorder=100)

			norm = mpl.colors.Normalize(vmin=np.min(spectrum), vmax=0)
			cb = fig.colorbar(
					plt.cm.ScalarMappable(norm=norm, cmap=cmap),
					location="right",
					ax=ax,
					ticks=[-0.4, -0.2, 0],
					format=mpl.ticker.FixedFormatter(["$-0.4$", "$-0.2$", "$0$"]),
					aspect=15,
					pad=0.02)
			cb.set_label(label=fr"$\log_{{10}} |t^{{\,}}_{{i}}|$")

			ax.plot(lam, overlaps[:, 0], marker='none', linewidth=10, color=COLORS[5], zorder=10)
			ax.plot(lam, np.pi / 5 - overlaps[:, 0], marker='none', linewidth=10, color=COLORS[5], zorder=10)

			def f(x, a, b, c):
				return a * np.abs(x - b) ** c * np.heaviside(x - b, 1)

			ind = np.where((lam > 0.2) & (lam < 0.39))[0]

			popt, pcov = sco.curve_fit(f, lam[ind], overlaps[ind, 0], [1, 0.313, 0.5])
			perr = np.sqrt(np.diag(pcov))

			print(popt, perr)

			x = np.linspace(np.min(lam), np.max(lam), 1000)
			p, = ax.plot(x, f(x, *popt), linestyle=(0, (3, 3)), color="0", linewidth=1, zorder=101, marker="none", label=r"$\mathrm{fit}^{\,}_{q}$")
			handles = [p]

			ax.set_ylim(-np.pi/42, np.pi/5+np.pi/42)

			ax.set_xticks([0, np.pi/8, np.pi/4])
			ax.set_xticklabels(["$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$"])
			
			ax.set_yticks([0, np.pi/10, np.pi/5])
			ax.set_yticklabels(["$0$", r"$\frac{\pi}{10}$", r"$\frac{\pi}{5}$"])
			
			ax.axvline(0.2993, color="0", ls="-", lw=0.5, zorder=11)

			plt.xlabel(fr"$\lambda$")

			plt.ylabel(fr"$\frac{{1}}{{{N}}}\,\Im\,\log t^{{\,}}_{{i}} \quad \mathrm{{mod}}\ \frac{{2\pi}}{{{N}}}$")

			for spine in ax.spines.values():
				spine.set_zorder(999)

			ax.legend(
				handles=handles,
				loc="center right",
				ncol=1,
				bbox_to_anchor=(1.01, 0.5)).set_zorder(1000)
			
			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"mod")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

		if plot_overlap:
			
			fig, ax = plt.subplots(figsize=(4.5, 2))

			p, = plt.plot(lam, overlaps[:, 0], linewidth=0, color=COLORS[0], marker='o', markersize=10, zorder=100)
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			axin = ax.inset_axes([0.65, 0.32, 0.33, 0.4])
			axin.set_zorder(101)

			def g(x, a, b, c):
				return a * np.log(x - b) * np.heaviside(x - b, 1) + c

			ind = np.where((lam > 0.31) & (lam < 0.39))[0]

			popt, pcov = sco.curve_fit(g, lam[ind], np.log(overlaps[ind, 0]), [0.5, 0.3, -1])
			perr = np.sqrt(np.diag(pcov))

			print(popt, perr)

			p, = axin.plot(lam[ind] - popt[1], overlaps[ind, 0], linewidth=0, color=COLORS[0], marker="o", markersize=10, zorder=100)
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			a = np.exp(popt[2])
			da = perr[2] * np.exp(popt[2])

			p, = axin.plot(lam[ind] - popt[1], np.exp(g(lam[ind], *popt)), linestyle='-', color=COLORS[4], linewidth=1, zorder=99, marker="none", label=fr"$({format_value(a, da)}) \times \Theta(\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}) \times |\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}|^{{{format_value(popt[0], perr[0])}}}$")
			handles = [p]

			axin.set_xscale("log")
			axin.set_yscale("log")

			ax.set_xlabel(fr"$\lambda,\quad \lambda^{{\,}}_{{\mathrm{{c}}}}={format_value(popt[1], perr[1])}$")
			ax.set_ylabel(fr"$q^{{\,}}_{{i^{{*}}}}$")

			axin.set_xlabel(fr"$\lambda-\lambda^{{\,}}_{{\mathrm{{c}}}}$", fontsize=10)

			axin.tick_params(labelsize=10)

			axin.legend(
				handles=handles,
				loc="lower right",
				ncol=1,
				bbox_to_anchor=(-0.5, 1.4)).set_zorder(1000)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"overlap")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()	

			# fig, ax = plt.subplots(figsize=(8, 3))

			# p, = plt.plot(lam, components[:, 0, 0], linewidth=0, color=COLORS[0], marker='o', markersize=6, markeredgewidth=1.5, zorder=100, label=f"overlap")
			# p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			# p, = plt.plot(lam, components[:, 0, 1], linewidth=0, color=COLORS[7], marker='x', markersize=6, markeredgewidth=1.5, zorder=100, label=f"norm")
			# p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			# ax.minorticks_on()

			# ax.set_xlabel(fr"$\lambda$")
			
			# ax.legend(loc="center right", ncol=1, labelspacing=0.8, columnspacing=2, handletextpad=0.8, handlelength=1.6).set_zorder(200)

			# plt.draw()

			# if args.save:
			# 	save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"components")
			# 	fig.clf()

			# if args.show:
			# 	plt.show()

			# plt.close()	

		if plot_corr:
			
			fig, ax = plt.subplots(figsize=(8, 3))

			ind = np.where(eta[:, 0, 0] != 0)[0]

			p, = plt.plot(lam[ind], eta[ind, 0, 0], linewidth=0, color=COLORS[0], marker='o', markersize=8, zorder=100, label=r"$\eta^{x}$")

			p, = plt.plot(lam[ind], eta[ind, 0, 1], linewidth=0, color=COLORS[6], marker='o', markersize=8, zorder=100, label=r"$\eta^{z,\mathrm{c}}$")

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





# import os
# import numpy as np
# import argparse
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from cycler import cycler
# import pandas as pd
# import seaborn as sns
# import scipy.optimize as sco

# from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
# from tenet.utils.io import folder_names, load_file, save_graph, format_value, format_sign
# from tenet.utils.sty import COLORS
# from tenet.utils.math import find_nearest
# from tenet.linalg.sun import SUIrrep

# su2 = SUIrrep(2, 2)
# X, Y, Z = [2 * su2.generators[i] for i in range(3)]

# parser = argparse.ArgumentParser()
# parser.add_argument("--save", action="store_true", help="Save the graph(s).")
# parser.add_argument("--show", action="store_true", help="Show the graph(s).")

# args = parser.parse_args()

# plot_corr = True

# start_x = 2
# end_x = 30
# start_y = 5
# end_y = 100
# start_z = 10
# end_z = 100

# eps_norm = 1e-5
# num_eigen = 300

# def connected_correlation(N, corr, spin):

# 	r = (len(corr) + 1) // N
# 	spin_extended = np.concatenate([spin for _ in range(r)], axis=0)

# 	conn = corr - spin_extended[0] * spin_extended[1:]
# 	return conn

# def main():
# 	"""
# 	Executes the plotting process.
# 	"""
# 	model_name = "B"
# 	simul_name = "LM+TMB/BOTTOM"

# 	boundary_name = "NONE"
# 	method_name = "IDMRG"

# 	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

# 	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
					
# 	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):

# 		if folder == "N=4_chi=256":
# 			continue
			
# 		engine_folder = os.path.join(DATA_FOLDER, base_folder, folder)
# 		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

# 		vals, data_filenames = load_file(data_folder, expression)

# 		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

# 		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
# 		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
# 		eta = np.zeros((len(lam), len(theta), 3))

# 		for i in range(len(data_filenames)):

# 			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
# 				data = pickle.load(file)

# 			print(data_filenames[i])

# 			if plot_corr:

# 				corrX, corrY, corrZ = data['corrs']

# 				# corrX = np.abs(corrX)
# 				# corrY = np.abs(corrY)
# 				corrX = np.abs(connected_correlation(N, corrX, data["mags"][0, :]))
# 				corrY = np.abs(connected_correlation(N, corrY, data["mags"][1, :]))
# 				corrZ = np.abs(connected_correlation(N, corrZ, data["mags"][2, :]))

# 				fig, ax = plt.subplots(figsize=(5, 3))
				
# 				def f(x, a, b):
# 					return a * x + b			
# 				def g(x, a, b, c, d):
# 					return a * np.log(x) + np.log(np.abs(np.cos(x * b + c))) + d
				
# 				if vals[i]['lam'] > 0:

# 					sites = np.array(range(1, len(corrX) + 1))

# 					popt, pcov = sco.curve_fit(f, np.log(sites[start_x:end_x]), np.log(corrX[start_x:end_x]))
# 					perr = np.sqrt(np.diag(pcov))

# 					print(popt, perr)

# 					eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = popt[0]

# 					p, = plt.plot(sites, corrX, marker='o', markersize=8, color=COLORS[3], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{x}$")

# 					p, = ax.plot(sites[start_x:end_x], np.exp(f(np.log(sites[start_x:end_x]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{x}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

# 					sites = np.array(range(1, len(corrY) + 1))

# 					popt, pcov = sco.curve_fit(f, np.log(sites[start_y:end_y]), np.log(corrY[start_y:end_y]))
# 					perr = np.sqrt(np.diag(pcov))

# 					print(popt, perr)

# 					eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[0]

# 					p, = plt.plot(sites, corrY, marker='o', markersize=8, color=COLORS[5], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{y}$")

# 					p, = ax.plot(sites[start_y:end_y], np.exp(f(np.log(sites[start_y:end_y]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{y}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

# 					if vals[i]['lam'] > 0.37:

# 						sites = np.array(range(1, len(corrZ) + 1))

# 						popt, pcov = sco.curve_fit(g, sites[start_z:end_z], np.log(corrZ[start_z:end_z]), [-2, 2*np.pi/10, 1, 0])
# 						perr = np.sqrt(np.diag(pcov))

# 						print(popt, perr)

# 						eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = popt[0]

# 						p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{z}$")

# 						p, = ax.plot(sites[start_z:end_z], np.exp(g(sites[start_z:end_z], *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{z}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")
					
# 					else: 

# 						sites = np.array(range(1, len(corrZ) + 1))

# 						popt, pcov = sco.curve_fit(f, np.log(sites[start_z:end_z]), np.log(corrZ[start_z:end_z]))
# 						perr = np.sqrt(np.diag(pcov))

# 						print(popt, perr)

# 						eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = popt[0]

# 						p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=100, label=r"$(-1)^{{s}} C^{z}$")

# 						p, = ax.plot(sites[start_z:end_z], np.exp(f(np.log(sites[start_z:end_z]), *popt)), color=COLORS[0], linewidth=1.5, linestyle='-', zorder=102, label=fr"$(-1)^{{s}} C^{{z}}\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

# 					# sites = np.array(range(1, len(corrZ) + 1))

# 					# if vals[i]['theta'] == np.round(0, 15):
# 					# 	ind = np.where(sites % 2 == 1)[0]
# 					# else:
# 					# 	ind = np.where(sites)[0]

# 					# popt, pcov = sco.curve_fit(f, np.log(sites[ind][start_z:end_z]), np.log(corrZ[ind][start_z:end_z]))
# 					# perr = np.sqrt(np.diag(pcov))

# 					# print(popt, perr)

# 					# if vals[i]['theta'] > 0.29 and vals[i]['theta'] < 0.3:
# 					# 	pass
# 					# else:
# 					# 	eta[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = popt[0]

# 					# p, = plt.plot(sites, corrZ, marker='o', markersize=8, color=COLORS[6], linewidth=0, zorder=101, label=r"$|C^{z}|$")
					
# 					# p, = ax.plot(sites[ind][start_z:end_z], np.exp(f(np.log(sites[ind][start_z:end_z]), *popt)), color=COLORS[5], linewidth=1.5, linestyle='--', zorder=103, label=fr"$|C^{{z}}|\sim s^{{{popt[0]:.4f}\pm{perr[0]:.4f}}}$")

# 					ax.set_xlabel(r'$s$')

# 					ax.set_xscale('log')
# 					ax.set_yscale('log')

# 					ax.legend(loc="lower left", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=1).set_zorder(101)
					
# 					plt.draw()

# 				if args.save:
# 					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder, "corr"), f"x_{data_filenames[i][:-4]}")
# 					fig.clf()

# 				if args.show:
# 					plt.show()

# 				plt.close()
					
# 		if plot_corr:
			
# 			fig, ax = plt.subplots(figsize=(8, 3))

# 			ind = np.where(eta[:, 0, 0] != 0)[0]

# 			p, = plt.plot(lam[ind], eta[ind, 0, 0], linewidth=0, color=COLORS[0], marker='o', markersize=8, zorder=100, label=r"$\eta^{x}$")

# 			ind = np.where(eta[:, 0, 1] != 0)[0]

# 			p, = plt.plot(lam[ind], eta[ind, 0, 1], linewidth=0, color=COLORS[2], marker='o', markersize=8, zorder=100, label=r"$\eta^{y}$")

# 			ind = np.where(eta[:, 0, 2] != 0)[0]

# 			p, = plt.plot(lam[ind], eta[ind, 0, 2], linewidth=0, color=COLORS[6], marker='o', markersize=8, zorder=100, label=r"$\eta^{z,\mathrm{c}}$")

# 			ax.set_xlabel(fr"$\lambda$")
			
# 			ax.legend(
# 				loc="upper right",
# 				ncol=1,
# 				bbox_to_anchor=(0.9, 1.1)).set_zorder(1000)

# 			plt.draw()

# 			if args.save:
# 				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"eta")
# 				fig.clf()

# 			if args.show:
# 				plt.show()

# 			plt.close()	

# if __name__ == "__main__":
# 	main()

