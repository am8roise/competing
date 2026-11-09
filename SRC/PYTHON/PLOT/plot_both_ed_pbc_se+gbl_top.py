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

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def formatter(L, labels):

	def _formatter(y, pos):

		eps = 1e-6
		for (frac, label) in labels:
			if np.abs(y - frac * L) < eps:
				return label
		return ""

	return _formatter

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "SE+GBL/TOP"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = False

	k = 6

	n_eps_degen = 1
	eps_degen = np.round(np.linspace(0.7, 0.7, n_eps_degen), 10)
		
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
					
					degens[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), e] = len(np.where(E_flat-E_flat[0] < eps_degen[e])[0])

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

				data = pd.DataFrame(degens_square[::-1, :, e], index=np.round(lam_square, 4)[::-1], columns=np.round(theta_square, 4))

				cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[4],COLORS[5],COLORS[0],COLORS[2]], N=np.max(data))

				fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.7))
				plt.subplots_adjust(wspace=0.1)

				axs = sns.heatmap(data, ax=ax[0], square=True, cbar=False, cmap=cmap, vmin=0.5, vmax=np.max(data)+0.5, rasterized=True)

				for spine in axs.spines.items():
					spine[1].set_visible(True)

				ax[0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(len(theta_square) / 2))
				ax[0].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(len(theta_square) / 10))
				
				labels = [
					(1.0, r"$\frac{\pi}{2}$"),
					(0.5, r"$\frac{\pi}{4}$"),
					(0.0, r"$0$")]

				ax[0].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter(len(theta_square), labels)))

				ax[0].set_yticks([])

				ax[0].tick_params(axis="x", labelrotation=0)
				ax[0].tick_params(axis="y", labelrotation=90)

				ax[0].text(
					0.98, 0.98,
					"(a)",
					transform=ax[0].transAxes,
					ha="right",
					va="top",
					fontsize=12)

				ax[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(len(lam_square) / 2))
				ax[0].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(len(lam_square) / 10))

				labels = [
					(0.0, r"$\pi$"),
					(0.5, r"$\frac{3\pi}{4}$"),
					(1.0, r"$\frac{\pi}{2}$")]

				ax[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter(len(lam_square), labels)))

				ax[0].set_xlabel(r"$\theta$")
				ax[0].set_ylabel(r"$\lambda$")

				ax[0].annotate(
					r"Neel$^{\mathrm{SPT}}_{y}$" "\n" r"$n = 2\, (8)$",
					xy=(0.2, 0.15),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)
				ax[0].annotate(
					r"Neel$^{\mathrm{SPT}}_{x}$" "\n" r"$n = 2\, (8)$",
					xy=(0.8, 0.15),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)
				ax[0].annotate(
					r"FM$^{\,}_{x}$" "\n" r"$n = 2\, (2)$",
					xy=(0.2, 0.85),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)
				ax[0].annotate(
					r"FM$^{\,}_{y}$" "\n" r"$n = 2\, (2)$",
					xy=(0.92, 0.77),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)
				ax[0].annotate(
					r"Neel$^{\,}_{z}$" "\n" r"$n = 2\, (2)$",
					xy=(0.5, 0.5),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)

	model_name = "B"
	simul_name = "SE+GBL/TOP"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = False

	k = 6

	n_eps_degen = 1
	eps_degen = np.round(np.linspace(0.7, 0.7, n_eps_degen), 10)
		
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
					
					degens[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), e] = len(np.where(E_flat-E_flat[0] < eps_degen[e])[0])

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

				data = pd.DataFrame(degens_square[::-1, :, e], index=np.round(lam_square, 4)[::-1], columns=np.round(theta_square, 4))

				cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[4],COLORS[5],COLORS[0],COLORS[2]], N=np.max(data))

				axs = sns.heatmap(data, ax=ax[1], square=True, cbar=False, cmap=cmap, vmin=0.5, vmax=np.max(data)+0.5, rasterized=True)

				for spine in axs.spines.items():
					spine[1].set_visible(True)

				pos_left  = ax[0].get_position()
				pos_right = ax[1].get_position()
				
				x0 = pos_left.x0 + 0.1
				x1 = pos_right.x1 - 0.1
				width = x1 - x0

				height = 0.04
				pad = 0.085
				y = pos_left.y1 + pad

				cax = fig.add_axes([x0, y, width, height])

				cb = fig.colorbar(ax[1].collections[0], cax=cax, location="top", ticks=np.arange(1, np.max(data) + 1), ticklocation="bottom")
				cb.set_label(label=r"$n \, : \, E^{\,}_{m} <  E^{\,}_{0} + \Delta \varepsilon, \quad 0 \leq m < n$", labelpad=-36)
				cb.ax.tick_params(which='both', size=0)

				ax[1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(len(theta_square) / 2))
				ax[1].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(len(theta_square) / 10))

				labels = [
					(1.0, r"$\frac{\pi}{2}$"),
					(0.5, r"$\frac{\pi}{4}$"),
					(0.0, r"$0$")]

				ax[1].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(formatter(len(theta_square), labels)))

				ax[1].set_yticks([])

				ax[1].tick_params(axis="x", labelrotation=0)
				ax[1].tick_params(axis="y", labelrotation=90)

				ax[1].set_xlabel(r"$\theta$")

				ax[1].text(
					0.98, 0.98,
					"(b)",
					transform=ax[1].transAxes,
					ha="right",
					va="top",
					fontsize=12)

				ax[1].annotate(
					r"Neel$^{\mathrm{SPT}}_{x}$" "\n" r"$n = 2\, (8)$",
					xy=(0.2, 0.15),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)
				ax[1].annotate(
					r"Neel$^{\mathrm{SPT}}_{y}$" "\n" r"$n = 2\, (8)$",
					xy=(0.8, 0.15),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)
				ax[1].annotate(
					r"FM$^{\,}_{x}$" "\n" r"$n = 2\, (2)$",
					xy=(0.2, 0.85),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)
				ax[1].annotate(
					r"FM$^{\,}_{y}$" "\n" r"$n = 2\, (2)$",
					xy=(0.92, 0.77),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)
				ax[1].annotate(
					r"Neel$^{\,}_{z}$" "\n" r"$n = 2\, (2)$",
					xy=(0.5, 0.5),
					xycoords="axes fraction",
					ha="center",
					va="center",
					fontsize=10,
					color="0",
					bbox=dict(
						boxstyle="square",
						fc="1",
						ec="0",
						lw=0.5,
						pad=0.3,
					),
				)

				plt.draw()

	if args.save:

		model_name = "BOTH"
		simul_name = "SE+GBL/TOP"

		boundary_name = "PBC" if PBCs else "OBC"
		method_name = "ED"

		base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

		folder = "N=14"

		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"eps={eps_degen[e]}")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()
