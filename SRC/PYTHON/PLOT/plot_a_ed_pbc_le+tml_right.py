import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import scipy.optimize as sco 

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph
from tenet.utils.sty import COLORS
from tenet.utils.math import find_nearest

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "LE+TML/RIGHT"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	k = 5

	Ns = []
	diffs_Ns = []
			
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		if (N == N_one) or N_all:

			lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
			theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

			Ns.append(N)

			diffs = np.zeros((len(lam), len(theta), k-1))

			for i in range(len(data_filenames)):

				with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
					engine = pickle.load(file)

				E = engine["energies"]

				E_flat = []
				for E_sector in E.values():
					E_flat = np.append(E_flat, E_sector)
				E_flat = np.sort(E_flat)[:k]

				gaps = E_flat[1:] - E_flat[:-1]

				diffs[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = gaps
					
			diffs_Ns.append(diffs)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

	cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[5],COLORS[4],COLORS[3],COLORS[2],COLORS[6],COLORS[0],COLORS[1]], N=256)

	plt.rcParams.update({'axes.prop_cycle': cycler(
		markersize=([4, 6, 8, 10]*len(Ns))[:len(Ns)],
		color=cmap(np.linspace(0, 1, len(Ns))))})

	fig, ax = plt.subplots(1, k-1, figsize=(4*(k-1)-1, 4))

	for l in range(len(lam)):
		for i in range(k-1):

			for n in range(len(Ns)):
				p = ax[i].plot(theta, Ns[n]*diffs_Ns[n, l, :, i], alpha=1, linestyle=":", linewidth=0.6, marker="o", zorder=100, label=fr"$2N = {Ns[n]}$")
				[p[s].set_marker(mpl.markers.MarkerStyle(p[s].get_marker(), capstyle="round")) for s in range(len(p))]

			ax[i].minorticks_on()
			ax[i].set_xlabel(r"$\theta$")
			ax[i].set_title(fr"$2N(E^{{\,}}_{{{i+1}}} - E^{{\,}}_{{{i}}})$", pad=10, fontsize=18)

			ax[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto", integer=True))

		ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, ncol=1, labelspacing=0.8).set_zorder(200)

		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"lam={np.round(lam[l], 3)}")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

	cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[3],COLORS[2],COLORS[6],COLORS[0],"#A77139",COLORS[1],COLORS[5],COLORS[4]], N=256)

	plt.rcParams.update({'axes.prop_cycle': cycler(
		markersize=([4, 8]*len(Ns))[:len(Ns)],
		linestyle=(["-", (1, (2, 3)), (0, (0, 2))]*len(Ns))[:len(Ns)],
		color=cmap(np.linspace(0, 1, len(Ns))))})

	fig, ax = plt.subplots(figsize=(4, 3))

	handles = []
	for l in range(len(lam)):

		# this to unpack the big number of data points in this region of theta
		ind_1 = np.where((theta > np.pi/2+0.4) & (theta < 2.15))[0]

		ind_2 = np.where((theta > 1.819) & (theta < np.pi/2+0.4))[0]
		ind_2 = ind_2[2::5]

		ind_3 = np.where((theta > 1.81) & (theta < 1.819))[0]

		zoom = np.concatenate((ind_3, ind_2, ind_1))
		# 

		for n in range(len(Ns)):
			p, = ax.plot(theta[zoom] - np.pi/2, Ns[n]*diffs_Ns[n, l, zoom, 1], marker="o", linewidth=1, zorder=1, label=fr"$2N = {Ns[n]}$")
			handles.append(p)

		ax.set_xlabel(r"$\theta$")
		ax.set_ylabel(r"$2N\,(E^{\,}_{2}-E^{\,}_{1})$")

		ax.set_xlim(0.22, 0.64)
		ax.set_ylim(1E-15, 1E3)

		ax.set_yscale("log")

		ax.set_yticks([1E-14, 1E-10, 1E-6, 1E-2, 1E2])

		ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([10.0 ** k for k in range(-20, 10)]))
		ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

		ax.legend(
			handles=handles,
			loc="center left",
			handlelength=1.5,
			handletextpad=0.6,
			bbox_to_anchor=(0.9, 0.5)).set_zorder(1000)

		theta_star = np.zeros((len(Ns)))

		for n in range(len(Ns)):
			window_theta_star = np.where((theta > 1.81) & (diffs_Ns[n, l, :, 1] > 1e-6))[0]
			ind = find_nearest(diffs_Ns[n, l, window_theta_star, 1], 1e-5)
			theta_star[n] = theta[window_theta_star][ind]

		axin = ax.inset_axes([0.54, 0.25, 0.33, 0.55])
		axin.set_zorder(101)

		p, = axin.plot(1 / Ns ** 2, theta_star - np.pi/2, marker="o", color=COLORS[0], markersize=10, linewidth=0, zorder=100)

		def f(N, a, b):
			return a*1/N + b

		ind = np.where(Ns >= 10)[0]

		popt, pcov = sco.curve_fit(f, Ns[ind] ** 2, theta_star[ind] - np.pi/2)
		perr = np.sqrt(np.diag(pcov))

		print(popt, perr)

		x = np.linspace(np.min(Ns[ind]), 1e10, 100)

		p, = axin.plot(1 / x ** 2, f(x ** 2, *popt),  linestyle="-", color=COLORS[4], linewidth=2, zorder=101, marker="none", label=fr"$A\times(2N)^{{-2}}+\theta^{{\,}}_{{\mathrm{{Neel}}^{{\mathrm{{I}}}}_{{z}}}}$")

		axin.legend(
			loc="upper center",
			handlelength=1.5,
			handletextpad=0.6,
			labelspacing=1.2,
			bbox_to_anchor=(-0.6, 1.6)).set_zorder(1001)	

		axin.set_xlim(0, 0.03)
		axin.set_ylim(0.22, 0.38)

		axin.set_xlabel(r"$(2N)^{-2}$", fontsize=10)
		axin.set_ylabel(r"$\theta^{\,}_{\mathrm{Neel}^{\mathrm{I}}_{z},2N}$", fontsize=10)

		axin.set_xticks([1/20**2, 1/10**2, 1/Ns[0]**2])
		axin.set_xticklabels([r"$\frac{1}{20^{2}}$", r"$\frac{1}{10^{2}}$", fr"$\frac{1}{{{Ns[0]:.0f}^{{2}}}}$"])

		# axin.set_yticks([0.3, ])

		axin.xaxis.set_minor_locator(mpl.ticker.FixedLocator(1 / Ns ** 2))
		axin.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

		axin.tick_params(labelsize=10)

		for spine in ax.spines.values():
			spine.set_zorder(10)
		for spine in axin.spines.values():
			spine.set_zorder(999)

		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"E2E1_lam={np.round(lam[l], 3)}")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

if __name__ == "__main__":
	main()
