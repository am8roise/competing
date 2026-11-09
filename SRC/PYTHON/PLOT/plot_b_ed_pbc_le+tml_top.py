import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import scipy.optimize as sco

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph, format_value, format_sign
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
	simul_name = "LE+TML/TOP"

	PBCs = True

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "ED"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	N_one = 14
	N_all = True

	k = 5

	plot_tricritical = True
	plot_lines = True

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

	if plot_tricritical:

		ind = np.argsort(Ns)

		Ns = np.array(Ns)[ind]
		diffs_Ns = np.array(diffs_Ns)[ind, :, :, :]

		fig, ax = plt.subplots(1, 2, figsize=(5, 2))
		plt.subplots_adjust(wspace=0.4)

		handles = []
		handles_ = []

		def f(x, a, b, c, d):
			return a * np.abs(x - d) ** b + c

		def g(N, b):
			return np.ones(len(N)) * b

		bs = np.zeros((len(theta)))
		dbs = np.zeros((len(theta)))

		for t in range(len(theta)):
		
			ind = np.where((Ns >= 18))[0]

			popt, pcov = sco.curve_fit(g, Ns[ind], diffs_Ns[ind, 0, t, 1])
			perr = np.sqrt(np.diag(pcov))

			bs[t] = popt[0]
			dbs[t] = perr[0]

			if t == 0:
				p, = ax[0].plot(Ns, diffs_Ns[:, 0, t, 1], marker="o", linewidth=0, color=COLORS[3], markersize=8, label=fr"$\theta=0$", zorder=100)
				handles.append(p)

				p, = ax[0].plot(Ns[ind], g(Ns[ind], *popt), linestyle="-", color=p.get_color(), linewidth=1.5, marker="none", label=fr"${format_value(popt[0], perr[0])}$", zorder=99)
				handles_.append(p)
			elif t == 1:
				p, = ax[0].plot(Ns, diffs_Ns[:, 0, t, 1], marker="o", linewidth=0, color=COLORS[2], markersize=8, label=r"$\theta=\frac{\pi}{4}\times\frac{1}{100}$", zorder=100)
				handles.append(p)

				p, = ax[0].plot(Ns[ind], g(Ns[ind], *popt), linestyle="-", color=p.get_color(), linewidth=1.5, marker="none", label=fr"${format_value(popt[0], perr[0])}$", zorder=99)
				handles_.append(p)
			elif t == 5:
				p, = ax[0].plot(Ns, diffs_Ns[:, 0, t, 1], marker="o", linewidth=0, color=COLORS[0], markersize=8, label=r"$\theta=\frac{\pi}{4}\times\frac{5}{100}$", zorder=100)
				handles.append(p)

				p, = ax[0].plot(Ns[ind], g(Ns[ind], *popt), linestyle="-", color=p.get_color(), linewidth=1.5, marker="none", label=fr"${format_value(popt[0], perr[0])}$", zorder=99)
				handles_.append(p)

		p, = ax[1].plot(theta, bs, linewidth=0, color=COLORS[0], marker="o", markersize=10, zorder=100)

		ind = np.where(theta < 0.04)[0]

		popt, pcov = sco.curve_fit(f, theta[ind], bs[ind], [3, 1, 0, 0])
		perr = np.sqrt(np.diag(pcov))

		print(popt, perr)

		p, = ax[1].plot(theta, f(theta, *popt), color=COLORS[4], linestyle="-", linewidth=2, zorder=101, marker="none",label=r"$\mathrm{fit}^{\,}_{\Delta}$")

		ax[0].set_xlabel(r"$2N$")
		ax[1].set_xlabel(r"$\theta$")

		ax[0].set_ylabel(r"$E^{\,}_{2}-E^{\,}_{1}$")
		ax[1].set_ylabel(r"$\Delta\coloneqq\lim\limits_{2N\to\infty}E^{\,}_{2}-E^{\,}_{1}$")

		ax[1].set_xlim(-0.05, 0.85)
		ax[0].set_ylim(-0.02, 0.18)
		ax[1].set_ylim(-0.2, 2.4)

		ax[1].set_xticks([0, np.pi/4 * 1/4, np.pi/4 * 2/4, np.pi/4])
		ax[1].set_xticklabels(["$0$", r"$\frac{\pi}{16}$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$"])

		ax[0].set_yticks([0, 0.1])
		ax[0].set_yticklabels(["$0$", "$0.1$"])

		ax[1].xaxis.set_minor_locator(mpl.ticker.FixedLocator([np.pi/4 * k * 1/20 for k in range(21)]))
		ax[1].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

		for spine in ax[1].spines.values():
			spine.set_zorder(1000)

		ax[0].text(
			0.03, 0.97,
			"(a)",
			transform=ax[0].transAxes,
			ha="left",
			va="top",
			fontsize=12)

		ax[1].text(
			0.03, 0.97,
			"(b)",
			transform=ax[1].transAxes,
			ha="left",
			va="top",
			fontsize=12)

		ax[0].legend(
			handles=[*handles, *handles_],
			loc="lower left",
			ncol=2,
			handlelength=1.5,
			columnspacing=2,
			handletextpad=0.8,
			bbox_to_anchor=(0, 1.06)).set_zorder(1000)

		ax[1].legend(
			handles=[p],
			loc="lower center",
			handlelength=1.5,
			handletextpad=0.8,
			bbox_to_anchor=(0.45, 0.93)).set_zorder(1000)

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

		plt.rcParams.update({'axes.prop_cycle': cycler(color=cmap(np.linspace(0, 1, len(Ns))))})

		for l in range(len(lam)):

			fig, ax = plt.subplots(1, k-1, figsize=(4*(k-1)-1, 4))

			for i in range(k-1):

				for n in range(len(Ns)):
					p = ax[i].plot(theta, Ns[n]*diffs_Ns[n, l, :, i], alpha=1, linestyle=":", marker='o', markersize=6, linewidth=0.6, zorder=100, label=fr"$2N = {Ns[n]}$")
					[p[s].set_marker(mpl.markers.MarkerStyle(p[s].get_marker(), capstyle="round")) for s in range(len(p))]

				ax[i].minorticks_on()
				ax[i].set_xlabel(r"$\theta$")
				if i==2:
					ax[i].set_title(fr"$2N(E^{{\,}}_{{{i+1}}} - E^{{\,}}_{{{i}}})$", loc="right", pad=10, fontsize=18)
				else:
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

if __name__ == "__main__":
	main()
