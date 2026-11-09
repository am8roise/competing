import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import seaborn as sns

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph
from tenet.utils.sty import COLORS
from tenet.utils.math import magnitude

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "SE+GBL/LEFT"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	k = 6

	n_eps_degen = 1
	eps_degen = np.round(np.linspace(1e-10, 2, n_eps_degen), 10)

	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		if (N == N_one) or N_all:
			
			lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
			theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

			degens = np.zeros((len(lam), len(theta), n_eps_degen))

			for i in range(len(data_filenames)):

				with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
					engine = pickle.load(file)

				E = engine["energies"]

				E_flat = []
				for E_sector in E.values():
					E_flat = np.append(E_flat, E_sector)
				E_flat = np.sort(E_flat)[:k]

				gaps = E_flat[1:] - E_flat[:-1]

				for e in range(n_eps_degen):
					
					degens[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), e] = len(np.where(E_flat-E_flat[0]<eps_degen[e])[0])

			degens_ll = degens
			degens_lr = degens[:, -2::-1, :]
			degens_ur = degens[-2::-1, -2::-1, :]
			degens_ul = degens[-2::-1, :, :]

			degens_l = np.concatenate((degens_ll, degens_lr), axis=1)
			degens_u = np.concatenate((degens_ul, degens_ur), axis=1)

			degens_square = np.concatenate((degens_l, degens_u), axis=0)
			lam_square = np.concatenate((lam, np.pi/4 + lam[1:]))
			theta_square = np.concatenate((theta, np.pi/4 + theta[1:]))

			for e in range(n_eps_degen):

				cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[2],COLORS[3],COLORS[5]], N=k)

				fig, ax = plt.subplots(figsize=(8, 6.4))

				data = pd.DataFrame(degens_square[::-1, :, e], index=np.round(lam_square, 4)[::-1], columns=np.round(theta_square, 4))

				axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, vmin=0.5, vmax=k+0.5, rasterized=True)

				cb = fig.colorbar(ax.collections[0], label=r"$n \, : \, E^{\,}_{m} <  E^{\,}_{0} + \Delta \varepsilon, \quad 0 \leq m < n$")
				cb.ax.tick_params(size=0)
				cb.set_ticks(np.arange(1, k+1))
				tick_texts = cb.ax.set_yticklabels([fr"{int(i+1)}" for i in range(k-1)] + [fr"{k}+"])

				plt.xlabel(r"$\theta$")
				plt.ylabel(r"$\lambda$")

				plt.xticks([0*len(theta_square), 0.5*len(theta_square), 1*len(theta_square)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
				plt.yticks([1*len(lam_square), 0.5*len(lam_square), 0*len(lam_square)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])			
				plt.xticks(rotation=0)
				plt.yticks(rotation=90)

				for _, spine in axs.spines.items(): 
				    spine.set_visible(True) 

				if eps_degen[e] > 1e-3:
					plt.title(fr"$\Delta \varepsilon = {eps_degen[e]}$")
				else:
					plt.title(fr"$\Delta \varepsilon = 10^{{{magnitude(eps_degen[e])}}}$")

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"eps={eps_degen[e]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

if __name__ == "__main__":
	main()
