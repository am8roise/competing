import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import scipy.optimize as sco

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph, format_value
from tenet.utils.sty import COLORS
from tenet.utils.math import PI

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

	plot_spin = False
	plot_spin_spt = False
	plot_entropy = False
	plot_correlations = True
			
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		engine_folder = os.path.join(DATA_FOLDER, base_folder, folder)
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			if plot_spin:

				fig, ax = plt.subplots(figsize=(3, 3))

				p, = ax.plot(range(1, N + 1), data['mags'][0], marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$\ev*{S^{x}_{j}}$")
				p, = ax.plot(range(1, N + 1), data['mags'][1], marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$\ev*{S^{y}_{j}}$")
				p, = ax.plot(range(1, N + 1), data['mags'][2], marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$\ev*{S^{z}_{j}}$")

				ax.set_xlabel(r"$j$")
			
				ax.legend(
					loc="upper left",
					ncol=1,
					bbox_to_anchor=(1.1, 1.1)).set_zorder(1001)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"mags_{data_filenames[i][:-4]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			if plot_spin_spt:

				fig, ax = plt.subplots(figsize=(3, 3))

				p, = ax.plot(range(1, N - 2), data['mags_spt'][0], marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$\ev*{S^{x,\mathrm{SPT}}_{j}}$")
				p, = ax.plot(range(1, N - 2), data['mags_spt'][1], marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$\ev*{S^{y,\mathrm{SPT}}_{j}}$")
				p, = ax.plot(range(1, N - 2), data['mags_spt'][2], marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$\ev*{S^{z,\mathrm{SPT}}_{j}}$")

				ax.set_xlabel(r"$j$")
			
				ax.legend(
					loc="upper left",
					ncol=1,
					bbox_to_anchor=(1.1, 1.1)).set_zorder(1001)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"mags_spt_{data_filenames[i][:-4]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			if plot_entropy:

				fig, ax = plt.subplots(figsize=(3, 3))

				win = N // 4

				S = data["entropy"]
				sites = np.arange(1, len(S) + 1)

				sites = sites[N // 2 - win // 2 - 1: N // 2 + win // 2]
				S = S[sites]

				p, = ax.plot(sites, S, marker='o', markersize=5, linewidth=0.5, zorder=100)

				def f(l, c, const):
					return c / 6 * np.log(2 * N // PI * np.sin(PI * l / N)) + const

				popt, pcov = sco.curve_fit(f, sites, S)
				perr = np.sqrt(np.diag(pcov))

				print(popt)
	
				label = rf"$S=\frac{{{format_value(popt[0], perr[0])}}}{6}\ln\left[\frac{{4N}}{{\pi}}\sin\frac{{\pi\ell}}{{2N}}\right]+{format_value(popt[1], perr[1])}$"

				p, = ax.plot(sites, f(sites, *popt), linestyle=":", marker="none", label=label, zorder=101)

				ax.set_xlabel(r"$\ell$")
				ax.set_ylabel(r"$S$")
			
				ax.legend(
					loc="upper left",
					ncol=1,
					bbox_to_anchor=(1.1, 1.1)).set_zorder(1001)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"entropy_{data_filenames[i][:-4]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()

			if plot_correlations:

				fig, ax = plt.subplots(figsize=(3, 3))

				corrX = data["corrs"][0, :]
				corrY = data["corrs"][1, :]
				corrZ = data["corrs"][2, :]

				sites = np.arange(1, len(corrX) + 1)

				p, = ax.plot(sites, corrZ, marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$C^{x}$")
				p, = ax.plot(sites, corrY, marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$C^{y}$")
				p, = ax.plot(sites, corrZ, marker='o', markersize=5, linewidth=0.5, zorder=100, label=r"$C^{z}$")

				ax.set_xlabel(r"$s$")
			
				ax.legend(
					loc="upper left",
					ncol=1,
					bbox_to_anchor=(1.1, 1.1)).set_zorder(1001)

				plt.draw()

				if args.save:
					save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder), f"corrs_{data_filenames[i][:-4]}")
					fig.clf()

				if args.show:
					plt.show()

				plt.close()
		
				# 	plt.plot(range(1, length(corrZ)), corrZ, marker=6, color="salmon", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{{\,}}_{{z}}$"), zorder=100)

				# 	sites = range(1, r)[start_:end+1-end_]
				# 	sites_z = range(1, r)[start_z:end+1-end_]

				# 	# corrX = corrX[start_:end+1-end_]
				# 	# corrY = corrY[start_:end+1-end_]
					
				# 	py"""
				# 	import numpy 

				# 	def f(l, a, b, c):
				# 		return a*(l**(-b)) + c
				# 	def g(l, a, b, c):
				# 		return a*numpy.exp(-l/b) + c
				# 	def oz(l, a, b, c, d):
				# 		return a*numpy.cos(2*numpy.pi*b*l + c) + d
				# 	def oz_(l, a, b, c, d, e):
				# 		return a*numpy.cos(2*numpy.pi*b*l + c)*(l**(-d)) + e 
				# 	"""

				# 	try
				# 		ind = numpy.where(sites_z .% 4 .== 0)[1] .+ 1
				# 		sites_ = sites_z[ind]
				# 		corrZ_ = corrZ[ind]

				# 		poptfz, pcov = scipy.optimize.curve_fit(py"f", collect(sites_), numpy.abs(corrZ_))
				# 		perrfz = numpy.sqrt(numpy.diag(pcov))
				# 		chi2 = scipy.stats.chisquare(numpy.abs(corrZ_), py"f"(collect(sites_), poptfz...), length(sites_)-4)
				# 		pfz = chi2[2]

				# 		plt.plot(sites_, py"f"(collect(sites_), poptfz...), ls="--", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\,}_{z}| = %$(round(poptfz[1]; digits=2)) \ell^{- %$(round(poptfz[2]; digits=2))} + %$(round(poptfz[3]; digits=2)), \ p = %$(round(pfz[1]; digits=3))$"))
				# 	catch
				# 		poptfz = 0
				# 		perrfz = 0
				# 		pfz = 0
				# 	end

				# 	try
				# 		poptfx, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrX))
				# 		perrfx = numpy.sqrt(numpy.diag(pcov))
				# 		chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"f"(collect(sites), poptfx...), length(sites)-4)
				# 		pfx = chi2[2]

				# 		plt.plot(sites, py"f"(collect(sites), poptfx...), ls="--", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\,}_{x}| = %$(round(poptfx[1]; digits=2)) \ell^{- %$(round(poptfx[2]; digits=2))} + %$(round(poptfx[3]; digits=2)), \ p = %$(round(pfx[1]; digits=3))$"))
				# 	catch
				# 		poptfx = 0
				# 		perrfx = 0
				# 		pfx = 0
				# 	end

				# 	try
				# 		poptgx, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrX))
				# 		perrgx = numpy.sqrt(numpy.diag(pcov))
				# 		chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"g"(collect(sites), poptgx...), length(sites)-4)
				# 		pgx = chi2[2]

				# 		plt.plot(sites, py"g"(collect(sites), poptgx...), ls=":", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\,}_{x}| = %$(round(poptgx[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgx[2]; digits=2))}} + %$(round(poptgx[3]; digits=2)), \ p = %$(round(pgx[1]; digits=3))$"))
				# 	catch
				# 		poptgx = 0
				# 		perrgx = 0
				# 		pgx = 0
				# 	end

				# 	try
				# 		poptfy, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrY))
				# 		perrfy = numpy.sqrt(numpy.diag(pcov))
				# 		chi2 = scipy.stats.chisquare(numpy.abs(corrY), py"f"(collect(sites), poptfy...), length(sites)-4)
				# 		pfy = chi2[2]

				# 		plt.plot(sites, py"f"(collect(sites), poptfy...), ls="--", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\,}_{y}| = %$(round(poptfy[1]; digits=2)) \ell^{- %$(round(poptfy[2]; digits=2))} + %$(round(poptfy[3]; digits=2)), \ p = %$(round(pfy[1]; digits=3))$"))
				# 	catch
				# 		poptfy = 0
				# 		perrfy = 0
				# 		pfy = 0
				# 	end

				# 	try
				# 		poptgy, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrY))
				# 		perrgy = numpy.sqrt(numpy.diag(pcov))
				# 		chi2 = scipy.stats.chisquare(numpy.abs(corrY), py"g"(collect(sites), poptgy...), length(sites)-4)
				# 		pgy = chi2[2]

				# 		plt.plot(sites, py"g"(collect(sites), poptgy...), ls=":", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\,}_{y}| = %$(round(poptgy[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgy[2]; digits=2))}} + %$(round(poptgy[3]; digits=2)), \ p = %$(round(pgy[1]; digits=3))$"))
				# 	catch
				# 		poptgy = 0
				# 		perrgy = 0
				# 		pgy = 0
				# 	end

				# 	try
				# 		sites = range(1, r)[start_z:end+1-end_]
				# 		corrZ_ = corrZ[sites]

				# 		poptoz, pcov = scipy.optimize.curve_fit(py"oz_", collect(sites), corrZ_, [0.4, 0.4, 1, 2, 0])
				# 		# poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(sites), corrZ_, [0.4, 0.2, 1, 0])
				# 		perroz = numpy.sqrt(numpy.diag(pcov))
				# 		chi2 = scipy.stats.chisquare(numpy.abs(corrZ_), py"oz_"(collect(sites), poptoz...), length(sites)-4)
				# 		poz = chi2[2]

				# 		println(poptoz)

				# 		plt.plot(sites, py"oz_"(collect(sites), poptoz...), ls="-", lw=0.7, color="k", zorder=102, label=latexstring(L"$|C^{\,}_{z}| = %$(round(poptoz[1]; digits=2)) \cos(2 \pi \cdot %$(round(poptoz[2]; digits=4)) + %$(round(poptoz[3]; digits=2))) + %$(round(poptoz[4]; digits=2)), \ p = %$(round(poz[1]; digits=3))$"))
				# 	catch
				# 		poptoz = 0
				# 		perroz = 0
				# 		poz = 0
				# 	end
							
				# 	ax.minorticks_on()
				# 	ax.grid(which="minor", linewidth=0.2)
				# 	ax.grid(which="major", linewidth=0.6)
				# 	ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

				# 	plt.xlabel(latexstring(L"$\ell$"))

				# 	# ax.set_yscale("log")

				# 	plt.draw()

				# 	if save_plot
				# 		py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations/"))
				# 		fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
				# 		ax.cla()
				# 	end

				# 	if plot_correlations_ft | plot_lines_correlations_ft

				# 		# ft = numpy.fft.fft(corrX)
				# 		# ftfreq = numpy.fft.fftfreq(length(corrX))
				# 		# sort = numpy.argsort(ftfreq).+1
				# 		# plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$x$"))

				# 		# ft = numpy.fft.fft(corrY)
				# 		# ftfreq = numpy.fft.fftfreq(length(corrY))
				# 		# sort = numpy.argsort(ftfreq).+1
				# 		# plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$y$"))

				# 		# corrZ = numpy.abs(corrZ) .* [numpy.exp(s / 2.5) for s in range(1, length(corrZ))]
				# 		# println(corrZ)
				# 		# corrZ = corrZ[10:60]

				# 		ft = numpy.fft.fft(corrZ)
				# 		ftfreq = numpy.fft.fftfreq(length(corrZ))
				# 		sort = numpy.argsort(ftfreq).+1
				# 		plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$z$"))
						
				# 		if plot_lines_correlations_ft

				# 			ft = numpy.abs(ft[sort])
				# 			ftfreq = ftfreq[sort]
				# 			cond = (ftfreq.>0.2) .& (ftfreq.<0.45)
				# 			ft = ft[cond]
				# 			ftfreq = ftfreq[cond]
				# 			ind_max_freq = numpy.argmax(ft)+1
				# 			max_freq = ftfreq[ind_max_freq]
				# 			max_amp = ft[ind_max_freq]

				# 			if max_freq<0.25
				# 				max_freq = 0
				# 			end

				# 			ft_maxs_corrs_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= max_freq
				# 			ft_maxs_corrs_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= max_amp

				# 		end

				# 		ax.minorticks_on()
				# 		ax.grid(which="minor", linewidth=0.2)
				# 		ax.grid(which="major", linewidth=0.6)
				# 		ax.legend(loc="center right").set_zorder(101)

				# 		plt.xlabel("frequency")
				# 		plt.ylabel("amplitude")

				# 		ax.set_yscale("log")

				# 		plt.draw()

				# 		if save_plot
				# 			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations_ft/"))
				# 			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations_ft/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
				# 			ax.cla()
				# 		end
				# 	end
				# end
if __name__ == "__main__":
	main()
