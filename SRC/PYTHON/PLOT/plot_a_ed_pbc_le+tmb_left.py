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

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "A"
	simul_name = "LE+TMB/LEFT"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 20
	N_all = True

	k = 5

	plot_tricritical = True
	plot_momentum = False
	plot_lines = False

	n_eps_degen = 1
	eps_degen = np.round(np.logspace(-7, -7, n_eps_degen), 10)

	Ns = []
	lam_star_Ns = []
	diffs_Ns = []
			
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		if (N == N_one) or N_all:

			lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
			theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
			Ns.append(N)

			lam_star = np.zeros((len(theta)))
			diffs = np.zeros((len(lam), len(theta), k-1))
			
			if plot_tricritical:

				found = False
				for e in range(n_eps_degen):
						
					for i in range(len(data_filenames)):

						if not found:

							with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
								engine = pickle.load(file)

							E = engine["energies"]

							E_flat = []
							for E_sector in E.values():
								E_flat = np.append(E_flat, E_sector)
							E_flat = np.sort(E_flat)[:k]

							gaps = E_flat[1:] - E_flat[:-1]
							
							degen = len(np.where(E_flat-E_flat[0]<eps_degen[e])[0])
							
							if degen == 2:
								if not lam[i] > 0.6:
									lam_star[np.where(vals[i]['theta']==theta)] = lam[i]
									found = True

									print(N, lam[i])

				lam_star_Ns.append(lam_star)

			if plot_momentum:
				
				for i in range(len(data_filenames)):

					with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
						engine = pickle.load(file)

					E = engine["energies"]

					fig, ax = plt.subplots(figsize=(8, 6.4))

					for k in E.keys():
					
						momentum = k * 2 * np.pi / N
						
						p = ax.plot([momentum for _ in range(len(E[k]))], E[k], alpha=1, marker=".", color=COLORS[0], markeredgewidth=1.4, markersize=6, linewidth=0, zorder=100)
						[p[s].set_marker(mpl.markers.MarkerStyle(p[s].get_marker(), capstyle="round")) for s in range(len(p))]

					ax.minorticks_on()

					ax.set_xticks([0, np.pi, 2*np.pi])
					ax.set_xticklabels([r"0", r"$\pi$", r"$2\pi$"])

					plt.xlabel(r"$k$")
					plt.ylabel(r"$E^{\,}_{k}$")

					plt.draw()

					if args.save:
						save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"lam={np.round(vals[i]['lam'], 3)}_theta={np.round(vals[i]['theta'], 3)}.png")
						fig.clf()

					if args.show:
						plt.show()

					plt.close()

			if plot_lines:
				
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

	if plot_tricritical:

		ind = np.argsort(Ns)

		Ns = np.array(Ns)[ind] ** 2
		lam_star_Ns = np.array(lam_star_Ns)[ind, :]

		ind = np.where(Ns >= 8 ** 2)[0]

		Ns_fit = Ns[ind]
		lam_star_Ns_fit = lam_star_Ns[ind, :]

		def f(N, a, b):
			return a * 1 / N + b

		for t in range(len(theta)):
			
			fig, ax = plt.subplots(figsize=(2, 2.5))

			popt, pcov = sco.curve_fit(f, Ns_fit, lam_star_Ns_fit[:, t])
			perr = np.sqrt(np.diag(pcov))

			print(popt, perr)

			p, = ax.plot(1/Ns, lam_star_Ns[:, t], marker="o", markerfacecolor=COLORS[0], linewidth=0, markersize=12, zorder=100)
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			x = np.linspace(np.min(Ns), 1e10, 100)

			ax.plot(1/x, f(x, *popt), marker="none", linestyle="-", lw=2, color=COLORS[4], zorder=101, label=fr"$A\times(2N)^{{-2}}+\lambda^{{\,}}_{{\mathrm{{gFM}}^{{\,}}_{{z}}}}$")

			plt.xlabel(r"$(2N)^{-2}$")
			plt.ylabel(r"$\lambda^{\,}_{\mathrm{gFM}^{\,}_{z},2N}$")

			ax.set_xlim(0, 0.03)
			ax.set_ylim(0.18, 0.32)

			ax.set_xticks([1/20**2, 1/10**2, 1/Ns[0]])
			ax.set_xticklabels([r"$\frac{1}{20^{2}}$", r"$\frac{1}{10^{2}}$", fr"$\frac{1}{{{np.sqrt(Ns[0]):.0f}^{{2}}}}$"])

			ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(1/Ns))
			ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

			ax.set_yticks([0.25, 0.3])

			ax.text(
				0.98, 0.97,
				"(a)",
				transform=ax.transAxes,
				ha="right",
				va="top",
				fontsize=12)

			ax.legend(
				loc="lower left",
				handlelength=1.2,
				handletextpad=0.6,
				bbox_to_anchor=(-0.4, 0.05)).set_zorder(1000)	

			for spine in ax.spines.values():
				spine.set_zorder(999)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"tricritical")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

	if plot_lines:

		ind = np.argsort(Ns)

		Ns = np.array(Ns)[ind]
		diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[7],COLORS[0],COLORS[1],COLORS[2],COLORS[3],COLORS[5]], N=256)

		plt.rcParams.update({'axes.prop_cycle': cycler(marker=([".", "|", "+"]*len(Ns))[:len(Ns)], color=cmap(np.linspace(0, 1, len(Ns))))})

		for t in range(len(theta)):

			fig, ax = plt.subplots(1, k-1, figsize=(4*(k-1)-1, 4))

			for i in range(k-1):

				for n in range(len(Ns)):
					p = ax[i].plot(lam, Ns[n]*diffs_Ns[n, :, t, i], alpha=1, linestyle=":", markeredgewidth=1.4, markersize=6, linewidth=0.6, zorder=100, label=fr"$2N = {Ns[n]}$")
					[p[s].set_marker(mpl.markers.MarkerStyle(p[s].get_marker(), capstyle="round")) for s in range(len(p))]

				ax[i].minorticks_on()
				ax[i].set_xlabel(r"$\theta$")
				ax[i].set_title(fr"$2N(E^{{\,}}_{{{i+1}}} - E^{{\,}}_{{{i}}})$", pad=10, fontsize=18)

				ax[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto", integer=True))
				ax[i].set_yscale('log')

			ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, ncol=1, labelspacing=0.8).set_zorder(200)

			plt.draw()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"theta={np.round(theta[t], 3)}")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

if __name__ == "__main__":
	main()