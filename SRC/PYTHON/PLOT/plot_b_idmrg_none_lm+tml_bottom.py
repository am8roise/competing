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

plot_mod = True
plot_corr = False

eps_norm = 1e-5
num_eigen = 200

start_z = 2
end_z = 20

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
	simul_name = "LM+TML/BOTTOM"

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
		
		spectrum = np.zeros((len(lam), len(theta), num_eigen))
		mod = np.zeros((len(lam), len(theta), num_eigen))

		line = np.zeros((len(lam), len(theta)))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			print(data_filenames[i])

			val_L = data['val_L']

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]

			val_L = val_L[:num_eigen]

			if plot_mod:

				spectrum[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = np.log10(np.abs(val_L[::-1]))
				mod[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = 1. / N * np.log(val_L[::-1]).imag % (2 * np.pi / N)

				q = 1. / N * np.log(val_L[::-1]).imag % (2 * np.pi / N)
				try:
					ind = np.where((q > 0.009) & (q < 0.5))[0]
					m = np.min(q[ind])
				except:
					ind = np.where((q > 0.0001) & (q < 0.5))[0]
					m = np.min(q[ind])

				line[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = m

		if plot_mod:

			fig, ax = plt.subplots(figsize=(4.5, 2))

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[4],COLORS[5],COLORS[3],COLORS[2]], N=256)

			for t in range(len(theta)):

				plt.scatter([theta[t] for _ in range(len(mod[0, t, :]))], mod[0, t, :], marker='o', s=12, linewidths=0.1, edgecolors="1", color=cmap(1 - np.abs(spectrum[0, t, :]) / np.max(np.abs(spectrum[0, t, :]))), zorder=100)

			norm = mpl.colors.Normalize(vmin=np.min(spectrum), vmax=0)
			cb = fig.colorbar(
					plt.cm.ScalarMappable(norm=norm, cmap=cmap),
					location="right",
					ax=ax,
					ticks=[-0.3, -0.2, -0.1, 0],
					format=mpl.ticker.FixedFormatter(["$-0.3$", "$-0.2$", "$-0.1$", "$0$"]),
					aspect=15,
					pad=0.02)
			cb.set_label(label=fr"$\log_{{10}} |t^{{\,}}_{{i}}|$")

			ax.plot(theta, line[0, :], marker='none', linewidth=10, color=COLORS[5], zorder=10)

			ax.set_xticks([0, np.pi/8, np.pi/4])
			ax.set_xticklabels(["$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$"])
			
			ax.set_yticks([0, np.pi/4, np.pi/2])
			ax.set_yticklabels(["$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
			
			plt.xlabel(fr"$\theta$")

			plt.ylabel(fr"$\frac{{1}}{{{N}}}\,\Im\,\log t^{{\,}}_{{i}} \quad \mathrm{{mod}}\ \frac{{2\pi}}{{{N}}}$", x=-0.25)
			
			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"mod")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()
		
			# fig, ax = plt.subplots(figsize=(8, 3))

			# p, = plt.plot(theta, line[0, :], linewidth=0, color=COLORS[0], marker='o', markersize=6, markeredgewidth=1.5, zorder=100)
			# p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			# ax.plot(theta, np.pi/10 * np.sin(2 * theta), linestyle='--', color=COLORS[3], zorder=101)

			# ax.minorticks_on()

			# plt.xlabel(fr"$\theta$")
			
			# plt.draw()

			# if args.save:
			# 	save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"line")
			# 	fig.clf()

			# if args.show:
			# 	plt.show()

			# plt.close()

if __name__ == "__main__":
	main()

