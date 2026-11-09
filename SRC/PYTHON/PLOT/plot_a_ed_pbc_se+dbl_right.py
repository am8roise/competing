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
	simul_name = "SE+DBL/RIGHT"

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

				diffs[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = gaps[:k-1]
		
		diffs_Ns.append(diffs)

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

	cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[7],COLORS[0],COLORS[1],COLORS[2],COLORS[3],COLORS[5]], N=256)

	plt.rcParams.update({'axes.prop_cycle': cycler(marker=([".", "|", "+"]*len(Ns))[:len(Ns)], color=cmap(np.linspace(0, 1, len(Ns))))})

	theta_star_of_lam = np.zeros((len(lam), 2))

	for l in range(len(lam)):

		fig, ax = plt.subplots(figsize=(8, 3))

		handles = []
		for n in range(len(Ns)):
			zoom = np.where(diffs_Ns[n, l, :, 1] > 0)[0]
			p, = ax.plot(theta[zoom], Ns[n]*diffs_Ns[n, l, zoom, 1], alpha=1, linestyle=":", markeredgewidth=1.4, markersize=6, linewidth=0.6, zorder=100, label=fr"$2N = {Ns[n]}$")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
			handles.append(p)

		ax.minorticks_on()
		ax.set_xlabel(r"$\theta$")

		ax.set_yscale("log")

		theta_star = np.zeros((len(Ns)))

		for n in range(len(Ns)):
			if lam[l] > 0.785:
				window_theta_star = np.where((theta > 1.81) & (diffs_Ns[n, l, :, 1] > 1e-6))[0]
			else:
				window_theta_star = np.where(diffs_Ns[n, l, :, 1] > 0)[0]
			ind = find_nearest(diffs_Ns[n, l, window_theta_star, 1], 1e-5)
			theta_star[n] = theta[window_theta_star][ind]

		axin = ax.inset_axes([0.65, 0.35, 0.33, 0.6])
		axin.set_zorder(201)

		p, = axin.plot(1/Ns ** 2, theta_star, alpha=1, marker="o", color="k", markeredgewidth=1.4, markersize=6, linewidth=0, zorder=100, label=r"$\theta^{\,}_{\mathrm{FM}^{\mathrm{A}}_{z},2N}$")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		handles.append(p)

		def f(N, a, b):
			return a * 1 / N + b

		ind = np.where(Ns >= 10)

		popt, pcov = sco.curve_fit(f, Ns[ind] ** 2, theta_star[ind])
		perr = np.sqrt(np.diag(pcov))

		print(lam[l], popt, perr)

		theta_star_of_lam[l, 0] = popt[1]
		theta_star_of_lam[l, 1] = perr[1]

		p, = axin.plot(1/Ns[ind] ** 2, f(Ns[ind] ** 2, *popt), linestyle=':', color="k", linewidth=0.4, zorder=99, marker="none", label=fr"$\theta^{{\,}}_{{\mathrm{{Neel}}^{{\,}}_{{z}}}} = ({popt[1]:.3f} \pm {perr[1]:.3f})$")
		p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		handles.append(p)

		axin.minorticks_on()

		axin.set_xlabel(r"$(2N)^{-1}$")
		axin.set_ylabel(r"$\theta^{\,}_{\mathrm{FM}^{\mathrm{A}}_{z},2N}$")

		ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.05), borderaxespad=0, ncol=3, labelspacing=0.6, columnspacing=2, handletextpad=0.6, handlelength=1.6).set_zorder(200)

		plt.draw()

		if args.save:
			if l == 29:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"E2E1_lam={np.round(lam[l], 3)}")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

	fig, ax = plt.subplots(figsize=(2, 2))

	lams = np.concatenate((lam, np.pi/2-lam[::-1])) + np.pi/2
	thetas = np.concatenate((theta_star_of_lam[:, 0], theta_star_of_lam[::-1, 0])) - np.pi/2
	dts = np.concatenate((theta_star_of_lam[:, 1], theta_star_of_lam[::-1, 1]))

	plt.fill_betweenx(lams, thetas - dts, thetas + dts, color=COLORS[2], lw=0.1, alpha=1, label=fr"$\partial\bigl(\mathrm{{Neel}}^{{\,}}_{{z}}\bigr)$")

	lams = np.concatenate((lam, np.pi/2-lam[::-1])) + np.pi/2
	thetas = np.concatenate((3*np.pi/2-theta_star_of_lam[:, 0], 3*np.pi/2-theta_star_of_lam[::-1, 0])) - np.pi/2
	dts = np.concatenate((theta_star_of_lam[:, 1], theta_star_of_lam[::-1, 1]))

	plt.fill_betweenx(lams, thetas - dts, thetas + dts, color=COLORS[2], lw=0.1, alpha=1)

	ax.plot(3*np.pi/4 - np.pi/2, np.pi/2-0.285 + np.pi/2, markersize=8, marker="o", color=COLORS[4], linewidth=0, zorder=200, label=r"$\Bigl(\frac{\pi}{4},\lambda^{{\,}}_{{\mathrm{{Neel}}^{{\,}}_{{z}}}}\Bigr)$")
	ax.plot(3*np.pi/4 - np.pi/2, 0.285 + np.pi/2, markersize=8, marker="o", color=COLORS[5], linewidth=0, zorder=200, label=r"$\Bigl(\frac{\pi}{4},\lambda^{{\mathrm{{SPT}}}}_{{\mathrm{{Neel}}^{{\,}}_{{z}}}}\Bigr)$")
	ax.plot(0.233, np.pi/4 + np.pi/2, markersize=8, marker="o", color=COLORS[0], linewidth=0, zorder=200, label=r"$\Bigl(\theta^{{\,}}_{{\mathrm{{Neel}}^{{\mathrm{{I}}}}_{{z}}}},\frac{3\pi}{4}\Bigr)$")
	ax.plot(np.pi/2 - 0.233, np.pi/4 + np.pi/2, markersize=8, marker="o", color=COLORS[1], linewidth=0, zorder=200, label=r"$\Bigl(\theta^{{\prime}}_{{\mathrm{{Neel}}^{{\mathrm{{I}}}}_{{z}}}},\frac{3\pi}{4}\Bigr)$")

	for spine in ax.spines.items():
		spine[1].set_visible(True)

	xticks = [0, np.pi/4, np.pi/2]
	xlabels = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]

	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels)

	yticks = [np.pi/2, 3*np.pi/4, np.pi]
	ylabels = [r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]

	ax.set_yticks(yticks)
	ax.set_yticklabels(ylabels)

	ax.tick_params(axis="x", labelrotation=0)
	ax.tick_params(axis="y", labelrotation=90)

	ax.set_aspect("equal")

	ax.set_xlabel(r"$\theta$")
	ax.set_ylabel(r"$\lambda$")

	ax.text(
		0.98, 0.03,
		"(a)",
		transform=ax.transAxes,
		ha="right",
		va="bottom",
		fontsize=12)

	ax.legend(
		loc="lower center",
		ncol=1,
		handlelength=1.5,
		handletextpad=0.8,
		labelspacing=0.5,
		bbox_to_anchor=(0.5, 0.9)).set_zorder(1000)

	plt.draw()

	if args.save:
		save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"boundary_square")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()

if __name__ == "__main__":
	main()
