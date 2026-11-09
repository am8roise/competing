import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import scipy.optimize as sco
import matplotlib as mpl
from cycler import cycler

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
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
	simul_name = "LM+TMB/BOTTOM"

	PBCs = False

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "DMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	r_spins = 440
			
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()
		chi = np.unique([int(vals[i]['chi']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

		mags = np.zeros((len(lam), len(theta), 1))
		ft_maxs_spins_z = np.zeros((len(lam), len(theta), 2))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			m = data["mags"][2, :]

			fit_range = range(N//2 - r_spins//2, N//2 + r_spins//2)

			m = m[fit_range]

			mags[vals[i]['lam']==lam, vals[i]['theta']==theta, 0] = np.mean(m)

			ft = np.fft.fft(m)
			ft_freq = np.fft.fftfreq(len(m))

			sort = np.argsort(ft_freq)
	
			ft = np.abs(ft[sort])
			ft_freq = ft_freq[sort]

			cond = (ft_freq > 0.2) & (ft_freq < 0.45)
			ft = ft[cond]
			ft_freq = ft_freq[cond]

			ind_max_freq = np.argmax(ft)

			max_freq = ft_freq[ind_max_freq]
			max_amp = ft[ind_max_freq]

			if (max_freq < 0.27) | (max_amp < 1e-4):
				max_freq = 0

			ft_maxs_spins_z[vals[i]["lam"]==lam, vals[i]["theta"]==theta, 0] = max_freq
			ft_maxs_spins_z[vals[i]["lam"]==lam, vals[i]["theta"]==theta, 1] = max_amp
					
		for t in range(len(theta)):
				
			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[5],COLORS[5],COLORS[3],COLORS[3],COLORS[3]], N=len(mags[:, t, 0]))

			def f(s, a, b, c, d):
				return a*np.heaviside(s-b, 1)*np.abs(s-b)**(c) + d
			def g(s, a, b):
				return a*s + b

			def r2(y, fit):

				ss_res = np.sum((y-fit)**2)
				ss_tot = np.sum((y-np.mean(y))**2)

				return (1 - ss_res/ss_tot)

			fig, ax = plt.subplots(1, 2, figsize=(7.7, 3.5), gridspec_kw={"width_ratios": [1.8, 1]})
			plt.subplots_adjust(wspace=0.4)

			axins1 = ax[0].inset_axes([0.65, 0.27, 0.3, 0.4])
			axins1.set_zorder(102)

			rang = np.append(np.arange(0, 7), np.arange(8, 16))
			r_ = lam[rang]
	
			poptf, pcov = sco.curve_fit(f, r_, np.abs(mags[rang, t, 0]), [1, 0.3, 0.15, 0])
			perrf = np.sqrt(np.diag(pcov))
			print(poptf, perrf)

			r_ = lam[8:15] - poptf[1]
	
			poptg, pcov = sco.curve_fit(g, np.log(r_), np.log(np.abs(mags[8:15, t, 0])), [0.1, 1])
			perrg = np.sqrt(np.diag(pcov))
			print(poptg, perrg)

			# r2_ = r2(np.log(np.abs(mags[:, t, 0])), g(np.log(r_), *poptg))
			# print(r2_)

			axins1.plot(np.log(r_), g(np.log(r_), *poptg), ls="-", lw=2, color=COLORS[6], zorder=101, label=r"$\beta^{\,}_{\mathrm{tri}}\,\ln |\lambda-\lambda^{\,}_{2N,\mathrm{tri}}|+\mathrm{const}$")

			#= Compute equation of linear relation =#
			popt, pcov = sco.curve_fit(g, ft_maxs_spins_z[8:, t, 0], np.abs(mags[8:, t, 0]), [-1, 1])
			perr = np.sqrt(np.diag(pcov))

			print(popt, perr)

			ax[0].plot(lam, ft_maxs_spins_z[:, t, 0], linewidth=1, marker=".", color=COLORS[0], markersize=5, zorder=100, label=r"$f^{z}$")	
			p, = ax[0].plot(lam, np.abs(mags[:, t, 0]), linewidth=1, marker="x", color=COLORS[3], markersize=4, markeredgewidth=1.2, zorder=100, label=r"$|m^{z}_{\mathrm{uni}}|$")
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		
			for j in range(len(mags[8:, t, 0])):
				ax[1].scatter(ft_maxs_spins_z[j+8, t, 0], np.abs(mags[j+8, t, 0]), marker="o", color=cmap(j), s=40, zorder=100)

			p, = axins1.plot(np.log(lam[8:]-poptf[1]), np.log(np.abs(mags[8:, t, 0])), linewidth=0.7, marker="x", color=COLORS[3], markersize=6, markeredgewidth=1.5, zorder=100)	
			p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))
			ax[0].plot(lam[8:], 100*lam[8:], ls="-", lw=2, color=COLORS[6], zorder=101, label=r"fit$^{\,}_{\mathrm{log}}$")
			
			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[5],COLORS[5],COLORS[3],COLORS[3],COLORS[3]], N=256)
			cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=np.min(lam[8:]), vmax=np.max(lam[8:])), cmap=cmap), ax=ax[1], drawedges=False)
			cb.set_label(r"$\lambda$")

			ax[0].annotate("(c)", xy=(0.4, 0.05), xycoords="axes fraction", fontsize=16)
			ax[1].annotate("(d)", xy=(0.4, 0.05), xycoords="axes fraction", fontsize=16)

			ax[0].set_xlabel(r"$\lambda$")
			ax[1].set_xlabel(r"$f^{z}$")
			ax[1].set_ylabel(r"$|m^{z}_{\mathrm{uni}}|$")
			axins1.set_xlabel(r"$\ln |\lambda-\lambda^{\,}_{\mathrm{tri}}|$", fontsize=14)
			axins1.set_ylabel(r"$\ln |m^{z}_{\mathrm{uni}}|$", fontsize=14)

			ax[0].legend(loc="center left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.02, 0.3)).set_zorder(200)

			# ax[1].set_xticks([0.3, 0.35, 0.4])
			# ax[2].set_yticks([0.3, 0.4])
			# axins1.set_xticks([-5, -3, -1])
			axins1.tick_params(labelsize=14)
		
			ax[0].set_ylim([-0.02, 0.42])
			ax[1].set_xlim([0.29, 0.4])
			axins1.set_xlim([-5.5, -0])
			axins1.set_ylim([-1.6, -0.85])

			ax[0].minorticks_on()
			ax[1].minorticks_on()
			axins1.minorticks_on()

			if args.save:
				save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"win={r_spins}")
				fig.clf()

			if args.show:
				plt.show()

			plt.close()

if __name__ == "__main__":
	main()
