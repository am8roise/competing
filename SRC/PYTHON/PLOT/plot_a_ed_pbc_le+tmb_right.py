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
	simul_name = "LE+TMB/RIGHT"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	k = 5

	n_eps_degen = 120 * 10 + 1
	eps_degen = np.round(np.linspace(0.5, 1.7, n_eps_degen), 10)

	Ns = []
	lam_star_Ns = []
			
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		if (N == N_one) or N_all:

			lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
			theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

			Ns.append(N)

			lam_star = np.zeros((len(theta)))
			
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
						
						if degen == 4:
							if not lam[i] > 0.6:
								lam_star[np.where(vals[i]['theta']==theta)] = lam[i]
								found = True

			lam_star_Ns.append(lam_star)
					
	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind] ** 2
	lam_star_Ns = np.array(lam_star_Ns)[ind, :]

	ind = np.where(Ns >= 10 ** 2)[0]

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

		ax.plot(1/x, f(x, *popt), marker="none", linestyle="-", lw=2, color=COLORS[4], zorder=101, label=fr"$B\times(2N)^{{-2}}+\pi-\lambda^{{\,}}_{{\mathrm{{Neel}}^{{\,}}_{{z}}}}$")

		plt.xlabel(r"$(2N)^{-2}$")
		plt.ylabel(r"$\pi-\lambda^{\,}_{\mathrm{Neel}^{\,}_{z},2N}$", y=0.43)

		ax.set_xlim(0, 0.03)
		ax.set_ylim(0.27, 0.39)

		ax.set_xticks([1/20**2, 1/10**2, 1/Ns[0]])
		ax.set_xticklabels([r"$\frac{1}{20^{2}}$", r"$\frac{1}{10^{2}}$", fr"$\frac{1}{{{np.sqrt(Ns[0]):.0f}^{{2}}}}$"])

		ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(1/Ns))
		ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

		ax.set_yticks([0.3, 0.35])

		ax.text(
			0.98, 0.97,
			"(b)",
			transform=ax.transAxes,
			ha="right",
			va="top",
			fontsize=12)

		ax.legend(
			loc="upper left",
			handlelength=1.2,
			handletextpad=0.6,
			bbox_to_anchor=(-0.4, 0.95)).set_zorder(1000)	

		for spine in ax.spines.values():
			spine.set_zorder(999)

		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"tricritical")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

if __name__ == "__main__":
	main()