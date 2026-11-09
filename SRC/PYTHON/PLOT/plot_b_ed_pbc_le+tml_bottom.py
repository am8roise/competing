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
	model_name = "B"
	simul_name = "LE+TML/BOTTOM"

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

	print("theta", theta)
	for n in range(len(Ns)):
		print("N", Ns[n])
		print("E3-E0", diffs_Ns[n, 0, :, 2]+diffs_Ns[n, 0, :, 1]+diffs_Ns[n, 0, :, 0])

	cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[7],COLORS[0],COLORS[1],COLORS[2],COLORS[3],COLORS[5]], N=256)

	plt.rcParams.update({'axes.prop_cycle': cycler(color=cmap(np.linspace(0, 1, len(Ns))))})

	for l in range(len(lam)):

		fig, ax = plt.subplots(1, k-1, figsize=(4*(k-1)-1, 4))

		for i in range(k-1):

			for n in range(len(Ns)):

				e = 0
				for j in range(i+1):
					e += diffs_Ns[n, l, :, j]

				p = ax[i].plot(theta, Ns[n]*e, alpha=1, linestyle=":", markersize=6, marker='o', linewidth=0.6, zorder=100, label=fr"$2N = {Ns[n]}$")
				[p[s].set_marker(mpl.markers.MarkerStyle(p[s].get_marker(), capstyle="round")) for s in range(len(p))]

			ax[i].minorticks_on()
			ax[i].set_xlabel(r"$\theta$")
			ax[i].set_title(fr"$2N(E^{{\,}}_{{{i+1}}} - E^{{\,}}_{{0}})$", pad=10, fontsize=18)

			ax[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto", integer=True))

		ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, ncol=1, labelspacing=0.8).set_zorder(200)

		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"_lam={np.round(lam[l], 3)}")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()
		
	def first_local_maximum(A):
		"""
		Return the index of the first local maximum encountered
		when scanning from left to right.

		A local maximum at index i satisfies:
			A[i] > A[i-1] and A[i] > A[i+1]

		Returns
		-------
		int or None
			Index of the first local maximum, or None if none exists.
		"""
		n = len(A)
		if n < 3:
			return None

		for i in range(1, n - 1):
			if (A[i] > A[i - 1]) and (A[i] > A[i + 1]):
				return i

		return None

	for l in range(len(lam)):

		fig, ax = plt.subplots(figsize=(4, 4))

		# ind = np.where((Ns % 4 == 2) & (Ns > 6))[0]

		# theta_star = np.zeros((len(Ns[ind])))
		# for n in range(len(Ns[ind])):
		# 	theta_star[n] = theta[first_local_maximum(diffs_Ns[ind[n], l, :, 0])]

		# p = ax.plot(1 / Ns[ind], theta_star, marker='o', linestyle=":", markeredgewidth=1.4, markersize=6, linewidth=0.6, zorder=100)

		ind_Ns = np.where((Ns % 4 == 0) & (Ns > 8))[0]

		theta_star = []
		for n in ind_Ns:
			ind =  np.where(diffs_Ns[n, l, :, 3] > 1E-10)[0]
			theta_star.append(theta[ind[0]])
		theta_star = np.array(theta_star)

		p = ax.plot(1 / Ns[ind_Ns], theta_star, marker='o', linestyle=":", markeredgewidth=1.4, markersize=6, linewidth=0.6, zorder=100)

		def f(x, a, b, c):
			return a * 1 / x ** b + c

		popt, pcov = sco.curve_fit(f, Ns[ind_Ns], theta_star, [1, 1, 0])
		perr = np.sqrt(np.diag(pcov))

		print(popt, perr)

		p, = ax.plot(1 / Ns[ind_Ns], f(Ns[ind_Ns], *popt), linestyle='--', color=COLORS[2], linewidth=2, zorder=99, marker="none")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		ax.minorticks_on()

		ax.set_xlabel(r"$2N$")

		# ax.set_xlim(0, 0.15)
		# ax.set_ylim(0, 0.3)

		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"point")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

if __name__ == "__main__":
	main()
