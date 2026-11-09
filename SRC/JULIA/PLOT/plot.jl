ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")
include("../utils/math.jl")

model_name = "B"
simul_name = "PS+TLL/BOTTOM"

PBCs = true

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

plot_degeneracies = false
plot_scaling_gaps = false
plot_lines_gaps = false
plot_scaling_en = false
plot_bulkenergies = false

plot_entropies = false
plot_lines_c = false
plot_entanglement = false

plot_spins = true
plot_spins_spt = false
plot_spins_ft = false
plot_lines_spins_ft = false
plot_lines_mag = false
plot_binder = false

plot_correlations = false
plot_correlations_ends = false
plot_correlations_ft = false
plot_lines_correlations_ft = false

plot_spt_correlations = false

plot_grid_evs = false

r = 96
start_ = 30
start_z = 2
end_ = 20
win = 10
epsDegen = 0.3
nE = 8

save_plot = true

fig, ax = plt.subplots(figsize=(8, 4))

Ns = []
diffsNs = []
ENs = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))
	# folder = "N=256_chi=128"

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	Es = numpy.zeros((length(lam), length(theta), nE))
	diffs = numpy.zeros((length(lam), length(theta), nE-1))
	degens = numpy.zeros((length(lam), length(theta), 1))
	cs = numpy.zeros((nE, length(lam), length(theta), 2))
	mags = numpy.zeros((nE, length(lam), length(theta), 6))
	ft_maxs_corrs_z = numpy.zeros((nE, length(lam), length(theta), 2))
	ft_maxs_spins_z = numpy.zeros((nE, length(lam), length(theta), 2))
	evs_2 = numpy.zeros((nE, length(lam), length(theta), 3))
	evs_4 = numpy.zeros((nE, length(lam), length(theta), 3))
	evs_6 = numpy.zeros((nE, length(lam), length(theta), 3))
	evs_8 = numpy.zeros((nE, length(lam), length(theta), 3))
	evs_10 = numpy.zeros((nE, length(lam), length(theta), 3))
	binder = numpy.zeros((nE, length(lam), length(theta), 3))

	for i=1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))
	
		k = length(data["energies"])
		
		println(data_filenames[i])
		# println(data["energies"][1]/N)

		# ------------------------------
		if true
		# ------------------------------

			if plot_degeneracies | plot_scaling_gaps | plot_lines_gaps | plot_scaling_en

				E = numpy.sort(data["energies"])[1:nE]

				# EB = data["bulkEnergies"]			
				# E = [numpy.min(EB[l]) for l=1:nE]
				# E = numpy.sort(E)		

				Es[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = numpy.sort([E[e] for e=1:nE])
				diffs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = [E[e+1]-E[e] for e=1:nE-1]
				degens[vals[i]["lam"].==lam, vals[i]["theta"].==theta] .= length(E[[E[e]-E[1]<epsDegen for e=1:nE]])
			
			end

			if plot_entanglement

				for j=1:k
					for b in 1:N-1
					
						# b = Int(N//2)+2
						# b = N-2
						num = min(20, Int(length(data["entanglement"][j, b])))

						ax.scatter(range(1, num), -2 .* numpy.log(data["entanglement"][j, b])[1:num], marker=".", s=5, color="mediumvioletred", zorder=100)
						# println(-2 .* numpy.log(data["entanglement"][j, b]))

						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)

						plt.xlabel(latexstring(L"i"))
						plt.ylabel(latexstring(L"$s^{\,}_{i}$"))

						plt.draw()

						if save_plot
							py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "entanglement/"))
							fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "entanglement/", "$(chop(data_filenames[i], tail=5))_b=$(b)_energy=$(data["energies"][j])" * ".png"))
							ax.cla()
						end
					end
				end
			end

			if plot_bulkenergies

				for j=1:k

					ax.plot(range(1, length(data["bulkEnergies"][j])), data["bulkEnergies"][j], marker="none", color="mediumvioletred", zorder=100)

					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)

					plt.xlabel(latexstring(L"iteration"))
					plt.ylabel(latexstring(L"$E$"))

					plt.draw()

					if save_plot
						py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "bulkEnergies/"))
						fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "bulkEnergies/", "$(chop(data_filenames[i], tail=5))_E%$(j-1)" * ".png"))
						ax.cla()
					end
				end
			end

			if plot_entropies | plot_lines_c

				Ss = data["entropies"]

				for j=1:k

					sites = range(1, size(Ss, 2))

					win = Int(4*N/5-4*(N%5)/5-4)
					# win = Int(N/2+(N/2)%2+4)
					# win = N-50
					# win = N//4

					sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
					S = Ss[j, Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

					# ind = numpy.where(sites .% 2 .== 1)[1] .+ 1
					# sites = sites[ind] .+ 0
					# S = S[ind]
					# println(sites)

					if PBCs
						py"""
						def f(l, c, const):
							import numpy

							N = int(l[len(l)//2]*2)

							return c/3 * numpy.log(N/numpy.pi * numpy.sin(numpy.pi*l/N)) + const

						"""
					else
						py"""
						def f(l, c, const):
							import numpy

							N = int(l[len(l)//2]*2)
							return c/6 * numpy.log(2*N/numpy.pi * numpy.sin(numpy.pi*l/N)) + const
						"""	
						# py"""
						# def f(l, c, a, b, const):
						# 	import numpy

						# 	N = int(l[len(l)//2]*2)
						# 	return c/6 * numpy.log(2*N/numpy.pi * numpy.sin(numpy.pi*l/N)) + a*numpy.cos(l * 2*numpy.pi*3/10 + b) + const
						# """
					end

					popt, pcov = scipy.optimize.curve_fit(py"f", collect(sites), S)
					perr = numpy.sqrt(numpy.diag(pcov))
					println(popt)

					cs[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= popt[1]
					cs[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= perr[1]

					if PBCs
						label=latexstring(L"$S = \frac{(%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3)))}{3} \ln\left[\frac{N}{\pi}\sin\frac{\pi \ell}{N} \right] + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3)))$")
					else	
						label=latexstring(L"$S = \frac{(%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3)))}{6} \ln\left[\frac{2N}{\pi}\sin\frac{\pi \ell}{N} \right] + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3)))$")
					end

					if plot_entropies

						ax.scatter(sites, S, marker="x", color="mediumvioletred", zorder=100)
						ax.plot(sites, py"f"(collect(sites), popt...), linestyle=":", marker=".", color="steelblue", label=label, zorder=100)

						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)
						ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0).set_zorder(101)

						plt.xlabel(latexstring(L"$\ell$"))
						plt.ylabel(latexstring(L"$S$"))

						plt.draw()
					
					end

					if save_plot
						py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "entropies/"))
						fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "entropies/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
						ax.cla()
					end
				end
			end

			if plot_binder

				for j=1:k

					mXsta = data["binder"][j, 1]

					binder[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.abs(mXsta)
				end
			end

			if plot_spins | plot_lines_mag | plot_spins_ft | plot_lines_spins_ft

				for j=1:k

					mX = data["mags"][j, 1, :]
					mY = data["mags"][j, 2, :]
					mZ = data["mags"][j, 3, :]

					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l=1:length(mX)])
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l=1:length(mY)])
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l=1:length(mZ)])

					println(numpy.mean([mX[l]*(-1)^l for l=1:length(mX)]))
					println(numpy.mean(mZ))

					if plot_spins
						ax.plot(range(1, length(mX)), mX, label=latexstring(L"$\langle \sigma^x_\ell \rangle$"), zorder=100)
						ax.plot(range(1, length(mY)), mY, label=latexstring(L"$\langle \sigma^y_\ell \rangle$"), zorder=100)
						ax.plot(range(1, length(mZ)), mZ, label=latexstring(L"$\langle \sigma^z_\ell \rangle$"), zorder=100)
						
						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)
						ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

						plt.xlabel(latexstring(L"$\ell$"))

						plt.draw()
					end

					py"""
					def f(x, a, b):
						import numpy

						N = int(len(x))

						return (1-(-1)**(numpy.floor(x/2)))/2 * ( a * (numpy.floor((x-2)/4)-numpy.floor((x-3)/4)) * numpy.sin(numpy.pi * (x - 1) / (N)) + b * (numpy.floor((x-3)/4)-numpy.floor((x-4)/4)) * numpy.sin(numpy.pi * (x - 2) / (N)) )
						# return a * numpy.sin(numpy.pi * (x-2) / (N - 1)) * (1+(-1)**(numpy.ceil(x/2)))/2 * numpy.sin(numpy.pi/2 * x + b) 

					"""
					py"""
					def g(x, a):
						import numpy

						N = int(len(x))

						# return a * (-1)**(x+1) * numpy.cos(numpy.pi * 4*(numpy.floor((x+2)/4)) / (N+2))
						return a * (-1)**(x+1) * ( (1-(-1)**x)/2 * numpy.cos(numpy.pi * (x-1) / (N)) + (1+(-1)**x)/2 * numpy.cos(numpy.pi * (x) / (N)) )
					"""

					# if lam[i]==0.1
					# # try
					# 	popt, pcov = scipy.optimize.curve_fit(py"g", collect(range(1, length(mY))), mX, [0.1])
					# 	perr = numpy.sqrt(numpy.diag(pcov))
					# 	println(popt, perr)

					# 	ax.plot(range(1, length(mY)), py"g"(collect(range(1, length(mY))), popt...), linestyle=":", marker=".", color="k", zorder=100)
					# 	# ax.plot(x, 0.02 .* numpy.sin(numpy.pi .* (x .- 2) ./ (N - 2)), linestyle=":", marker=".", color="red", zorder=100)
					# # catch
					# elseif lam[i]==0
					# 	print("mX at 0 ", j, mX[1])
					# 	popt, pcov = scipy.optimize.curve_fit(py"g", collect(range(1, length(mX))), mX, [1])
					# 	perr = numpy.sqrt(numpy.diag(pcov))
					# 	println(popt, perr)

					# 	ax.plot(range(1, length(mX)), py"g"(collect(range(1, length(mX))), popt...), linestyle=":", marker=".", color="k", zorder=100)
					# end

					if save_plot & plot_spins
						py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "spins/"))
						fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "spins/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
						ax.cla()
					end

					if plot_spins_ft | plot_lines_spins_ft

						mX = mX[start_:end+1-end_]
						mY = mY[start_:end+1-end_]
						mZ = mZ[start_:end+1-end_]

						ft = numpy.fft.fft(mX)
						ftfreq = numpy.fft.fftfreq(length(mX))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$x$"))

						ft = numpy.fft.fft(mY)
						ftfreq = numpy.fft.fftfreq(length(mY))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$y$"))

						ft = numpy.fft.fft(mZ)
						ftfreq = numpy.fft.fftfreq(length(mZ))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$z$"))
						
						if plot_lines_spins_ft

							ft = numpy.abs(ft[sort])
							ftfreq = ftfreq[sort]
							cond = (ftfreq.>0.2) .& (ftfreq.<0.45)
							ft = ft[cond]
							ftfreq = ftfreq[cond]
							ind_max_freq = numpy.argmax(ft)+1
							max_freq = ftfreq[ind_max_freq]
							max_amp = ft[ind_max_freq]

							if (max_freq<0.25) | (max_freq>0.4)
								max_freq = 0
							end

							ft_maxs_spins_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= max_freq
							ft_maxs_spins_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= max_amp

						end

						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)
						ax.legend(loc="center right").set_zorder(101)

						plt.xlabel("frequency")
						plt.ylabel("amplitude")

						ax.set_yscale("log")

						plt.draw()

						if save_plot
							py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "spins_ft/"))
							fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "spins_ft/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
							ax.cla()
						end
					end
				end
			end

			if plot_spins_spt

				for j=1:k

					mX = data["opSPT"][j, 1, :]
					mY = data["opSPT"][j, 2, :]
					mZ = data["opSPT"][j, 3, :]

					start = 1
					end_ = 0
					if !PBCs
						start = 2
						end_ = -1
					end

					ax.plot(range(start, length(mX)-end_), mX, label=latexstring(L"$\langle \sigma^x_\ell \rangle$"), zorder=100)
					ax.plot(range(start, length(mY)-end_), mY, label=latexstring(L"$\langle \sigma^y_\ell \rangle$"), zorder=100)
					ax.plot(range(start, length(mZ)-end_), mZ, label=latexstring(L"$\langle \sigma^z_\ell \rangle$"), zorder=100)

					py"""
					def f(x, a, b):
						import numpy

						N = int(len(x))

						return (1+(-1)**(numpy.floor(x/2)))/2 * ( a * (numpy.floor(x/4)-numpy.floor((x-1)/4)) * numpy.sin(numpy.pi * (x - 2) / (N - 2)) + b * (numpy.floor((x-1)/4)-numpy.floor((x-2)/4)) * numpy.sin(numpy.pi * (x - 3) / (N - 2)) )
						# return a * numpy.sin(numpy.pi * (x-2) / (N - 1)) * (1+(-1)**(numpy.ceil(x/2)))/2 * numpy.sin(numpy.pi/2 * x + b) 

					"""

					# if lam[i]==0.1
					# # try
					# 	popt, pcov = scipy.optimize.curve_fit(py"f", collect(range(start, length(mY)-end_)), mY, [0.1, 1])
					# 	perr = numpy.sqrt(numpy.diag(pcov))
					# 	println(popt, perr)

					# 	ax.plot(range(start, length(mY)-end_), py"f"(collect(range(start, length(mY)-end_)), popt...), linestyle=":", marker=".", color="k", zorder=100)
					# 	x = range(start, length(mY)-end_)
					# 	# ax.plot(x, 0.02 .* numpy.sin(numpy.pi .* (x .- 2) ./ (N - 2)), linestyle=":", marker=".", color="red", zorder=100)
					# # catch
					# end
					
					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

					plt.xlabel(latexstring(L"$\ell$"))

					plt.draw()

					if save_plot & plot_spins_spt
						py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "spinsSPT/"))
						fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "spinsSPT/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
						ax.cla()
					end
				end
			end

			if plot_correlations | plot_correlations_ft | plot_lines_correlations_ft | plot_correlations_ends

				for j=1:k

					corrX = data["corrs"][j, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
					corrY = data["corrs"][j, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
					corrZ = data["corrs"][j, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

					# if plot_correlations_ends
					# 	println("X_1, X_2N ", data["corrs"][j, 1, 1, N])
					# 	println("Y_1, Y_2N ", data["corrs"][j, 2, 1, N])
					# 	println("Z_1, Z_2N ", data["corrs"][j, 3, 1, N])
					# 	println("X_1 Z_2, Z_2N-1 X_2N ", data["ev_ends"][j, 1])
					# 	println("Y_1 Z_2, Z_2N-1 Y_2N ", data["ev_ends"][j, 2])
					# end
					
					corrX = corrX - data["mags"][j, 1, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)].*data["mags"][j, 1, Int(N/2)-Int(r/2)]
					corrY = corrY - data["mags"][j, 2, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)].*data["mags"][j, 2, Int(N/2)-Int(r/2)]
					corrZ = corrZ - data["mags"][j, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)].*data["mags"][j, 3, Int(N/2)-Int(r/2)]

					# corrZ = corrZ .* [numpy.exp(s / 2.45) for s in range(1, length(corrZ))]
					# corrZ = corrZ[1:60]


					plt.plot(range(1, r), numpy.abs(corrX), marker=4, color="mediumvioletred", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{{\,}}_{{x}}$"), zorder=100)
					plt.plot(range(1, r), numpy.abs(corrY), marker=5, color="steelblue", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{{\,}}_{{y}}$"), zorder=100)
					plt.plot(range(1, r), numpy.abs(corrZ), marker=6, color="salmon", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{{\,}}_{{z}}$"), zorder=100)

					# plt.plot(range(1, length(corrZ)), corrZ, marker=6, color="salmon", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{{\,}}_{{z}}$"), zorder=100)

					# sites = range(1, r)[start_:end+1-end_]
					# sites_z = range(1, r)[start_z:end+1-end_]

					# corrX = corrX[start_:end+1-end_]
					# corrY = corrY[start_:end+1-end_]
					
					py"""
					import numpy 

					def f(l, a, b, c):
						return a*(l**(-b)) + c
					def g(l, a, b, c):
						return a*numpy.exp(-l/b) + c
					def oz(l, a, b, c, d):
						return a*numpy.cos(2*numpy.pi*b*l + c) + d
					def oz_(l, a, b, c, d, e):
						return a*numpy.cos(2*numpy.pi*b*l + c)*(l**(-d)) + e 
					"""

					try
						ind = numpy.where(sites_z .% 4 .== 0)[1] .+ 1
						sites_ = sites_z[ind]
						corrZ_ = corrZ[ind]

						poptfz, pcov = scipy.optimize.curve_fit(py"f", collect(sites_), numpy.abs(corrZ_))
						perrfz = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrZ_), py"f"(collect(sites_), poptfz...), length(sites_)-4)
						pfz = chi2[2]

						plt.plot(sites_, py"f"(collect(sites_), poptfz...), ls="--", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\,}_{z}| = %$(round(poptfz[1]; digits=2)) \ell^{- %$(round(poptfz[2]; digits=2))} + %$(round(poptfz[3]; digits=2)), \ p = %$(round(pfz[1]; digits=3))$"))
					catch
						poptfz = 0
						perrfz = 0
						pfz = 0
					end

					try
						poptfx, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrX))
						perrfx = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"f"(collect(sites), poptfx...), length(sites)-4)
						pfx = chi2[2]

						plt.plot(sites, py"f"(collect(sites), poptfx...), ls="--", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\,}_{x}| = %$(round(poptfx[1]; digits=2)) \ell^{- %$(round(poptfx[2]; digits=2))} + %$(round(poptfx[3]; digits=2)), \ p = %$(round(pfx[1]; digits=3))$"))
					catch
						poptfx = 0
						perrfx = 0
						pfx = 0
					end

					try
						poptgx, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrX))
						perrgx = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"g"(collect(sites), poptgx...), length(sites)-4)
						pgx = chi2[2]

						plt.plot(sites, py"g"(collect(sites), poptgx...), ls=":", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\,}_{x}| = %$(round(poptgx[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgx[2]; digits=2))}} + %$(round(poptgx[3]; digits=2)), \ p = %$(round(pgx[1]; digits=3))$"))
					catch
						poptgx = 0
						perrgx = 0
						pgx = 0
					end

					try
						poptfy, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrY))
						perrfy = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrY), py"f"(collect(sites), poptfy...), length(sites)-4)
						pfy = chi2[2]

						plt.plot(sites, py"f"(collect(sites), poptfy...), ls="--", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\,}_{y}| = %$(round(poptfy[1]; digits=2)) \ell^{- %$(round(poptfy[2]; digits=2))} + %$(round(poptfy[3]; digits=2)), \ p = %$(round(pfy[1]; digits=3))$"))
					catch
						poptfy = 0
						perrfy = 0
						pfy = 0
					end

					try
						poptgy, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrY))
						perrgy = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrY), py"g"(collect(sites), poptgy...), length(sites)-4)
						pgy = chi2[2]

						plt.plot(sites, py"g"(collect(sites), poptgy...), ls=":", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\,}_{y}| = %$(round(poptgy[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgy[2]; digits=2))}} + %$(round(poptgy[3]; digits=2)), \ p = %$(round(pgy[1]; digits=3))$"))
					catch
						poptgy = 0
						perrgy = 0
						pgy = 0
					end

					try
						sites = range(1, r)[start_z:end+1-end_]
						corrZ_ = corrZ[sites]

						poptoz, pcov = scipy.optimize.curve_fit(py"oz_", collect(sites), corrZ_, [0.4, 0.4, 1, 2, 0])
						# poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(sites), corrZ_, [0.4, 0.2, 1, 0])
						perroz = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrZ_), py"oz_"(collect(sites), poptoz...), length(sites)-4)
						poz = chi2[2]

						println(poptoz)

						plt.plot(sites, py"oz_"(collect(sites), poptoz...), ls="-", lw=0.7, color="k", zorder=102, label=latexstring(L"$|C^{\,}_{z}| = %$(round(poptoz[1]; digits=2)) \cos(2 \pi \cdot %$(round(poptoz[2]; digits=4)) + %$(round(poptoz[3]; digits=2))) + %$(round(poptoz[4]; digits=2)), \ p = %$(round(poz[1]; digits=3))$"))
					catch
						poptoz = 0
						perroz = 0
						poz = 0
					end

					ax.set_xscale("log")
					ax.set_yscale("log")
							
					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

					plt.xlabel(latexstring(L"$\ell$"))

					# ax.set_yscale("log")

					plt.draw()

					if save_plot
						py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations/"))
						fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
						ax.cla()
					end

					if plot_correlations_ft | plot_lines_correlations_ft

						# ft = numpy.fft.fft(corrX)
						# ftfreq = numpy.fft.fftfreq(length(corrX))
						# sort = numpy.argsort(ftfreq).+1
						# plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$x$"))

						# ft = numpy.fft.fft(corrY)
						# ftfreq = numpy.fft.fftfreq(length(corrY))
						# sort = numpy.argsort(ftfreq).+1
						# plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$y$"))

						# corrZ = numpy.abs(corrZ) .* [numpy.exp(s / 2.5) for s in range(1, length(corrZ))]
						# println(corrZ)
						# corrZ = corrZ[10:60]

						ft = numpy.fft.fft(corrZ)
						ftfreq = numpy.fft.fftfreq(length(corrZ))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$z$"))
						
						if plot_lines_correlations_ft

							ft = numpy.abs(ft[sort])
							ftfreq = ftfreq[sort]
							cond = (ftfreq.>0.2) .& (ftfreq.<0.45)
							ft = ft[cond]
							ftfreq = ftfreq[cond]
							ind_max_freq = numpy.argmax(ft)+1
							max_freq = ftfreq[ind_max_freq]
							max_amp = ft[ind_max_freq]

							if max_freq<0.25
								max_freq = 0
							end

							ft_maxs_corrs_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= max_freq
							ft_maxs_corrs_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= max_amp

						end

						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)
						ax.legend(loc="center right").set_zorder(101)

						plt.xlabel("frequency")
						plt.ylabel("amplitude")

						ax.set_yscale("log")

						plt.draw()

						if save_plot
							py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations_ft/"))
							fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations_ft/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
							ax.cla()
						end
					end
				end
			end

			if plot_spt_correlations
				
				for j=1:k	

					py"""
					import numpy

					def f(l, a, b, c):
						return a*(l**(-b)) + c
					def g(l, a, b, c):
						return a*numpy.exp(-l/b) + c
					def oz(l, a, b, c, d):
						return a*numpy.cos(2*numpy.pi*b*l + c) + d
					"""

					corrX = data["corrsSPT"][j, 1, :]
					# plt.plot(range(1, length(corrX)), corrX, marker=4, color="mediumvioletred", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{\,}_{x, \mathrm{SPT}}$"), zorder=100)

					corrY = data["corrsSPT"][j, 2, :]
					# plt.plot(range(1, length(corrY)), corrY, marker=5, color="steelblue", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{\,}_{y, \mathrm{SPT}}$"), zorder=100)
					
					corrZ = data["corrsSPT"][j, 3, :]
					# plt.plot(range(1, length(corrZ)), corrZ, marker=5, color="salmon", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{\,}_{z, \mathrm{SPT}}$"), zorder=100)

					sites = range(1, length(corrX))[start_:end+1-end_]

					corrX = corrX[start_:end+1-end_]
					corrY = corrY[start_:end+1-end_]

					corrZ = corrZ - data["opSPT"][j, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)].*data["opSPT"][j, 3, Int(N/2)-Int(r/2)]

					corrZ = corrZ .* [numpy.exp(s / 2.45) for s in range(1, length(corrZ))]
					corrZ = corrZ[1:60]

					plt.plot(range(1, length(corrZ)), corrZ, marker=5, color="salmon", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{\,}_{z, \mathrm{SPT}}$"), zorder=100)

					try
						poptfx, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrX))
						perrfx = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"f"(collect(sites), poptfx...), length(sites)-4)
						pfx = chi2[2]

						plt.plot(sites, py"f"(collect(sites), poptfx...), ls="--", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\mathrm{SPT}}_{x}| = %$(round(poptfx[1]; digits=2)) \ell^{- %$(round(poptfx[2]; digits=2))} + %$(round(poptfx[3]; digits=2)), \ p = %$(round(pfx[1]; digits=3))$"))
					catch
						poptfx = 0
						perrfx = 0
						pfx = 0
					end

					try
						poptgx, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrX))
						perrgx = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"g"(collect(sites), poptgx...), length(sites)-4)
						pgx = chi2[2]

						plt.plot(sites, py"g"(collect(sites), poptgx...), ls=":", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\mathrm{SPT}}_{x}| = %$(round(poptgx[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgx[2]; digits=2))}} + %$(round(poptgx[3]; digits=2)), \ p = %$(round(pgx[1]; digits=3))$"))
					catch
						poptgx = 0
						perrgx = 0
						pgx = 0
					end

					try
						poptfy, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrY))
						perrfy = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrY), py"f"(collect(sites), poptfy...), length(sites)-4)
						pfy = chi2[2]

						plt.plot(sites, py"f"(collect(sites), poptfy...), ls="--", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\mathrm{SPT}}_{y}| = %$(round(poptfy[1]; digits=2)) \ell^{- %$(round(poptfy[2]; digits=2))} + %$(round(poptfy[3]; digits=2)), \ p = %$(round(pfy[1]; digits=3))$"))
					catch
						poptfy = 0
						perrfy = 0
						pfy = 0
					end

					try
						poptgy, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrY))
						perrgy = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrY), py"g"(collect(sites), poptgy...), length(sites)-4)
						pgy = chi2[2]

						plt.plot(sites, py"g"(collect(sites), poptgy...), ls=":", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\mathrm{SPT}}_{y}| = %$(round(poptgy[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgy[2]; digits=2))}} + %$(round(poptgy[3]; digits=2)), \ p = %$(round(pgy[1]; digits=3))$"))
					catch
						poptgy = 0
						perrgy = 0
						pgy = 0
					end
					
					# try
						sites = range(1, r)[start_z:end+1-end_]
						corrZ = corrZ[sites]

						poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(sites), corrZ, [0.4, 0.2, 1, 0])
						perroz = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrZ), py"oz"(collect(sites), poptoz...), length(sites)-4)
						poz = chi2[2]

						plt.plot(sites, py"oz"(collect(sites), poptoz...), ls="-", lw=0.7, color="k", zorder=102, label=latexstring(L"$|C^{\,}_{z}| = %$(round(poptoz[1]; digits=2)) \cos(2 \pi \cdot %$(round(poptoz[2]; digits=4)) + %$(round(poptoz[3]; digits=2))) + %$(round(poptoz[4]; digits=2)), \ p = %$(round(poz[1]; digits=3))$"))
					# catch
					# 	poptoz = 0
					# 	perroz = 0
					# 	poz = 0
					# end

					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

					plt.xlabel(latexstring(L"$\ell$"))
					plt.ylabel(latexstring(L"$C^{\,}_{\alpha, \mathrm{SPT}} = \langle O^z_{%$(Int(N/2-r/2-2))} O^\alpha_{%$(Int(N/2-r/2-1))} O^z_{%$(Int(N/2-r/2))} O^z_{%$(Int(N/2-r/2))+\ell} O^\alpha_{%$(Int(N/2-r/2+1))+\ell} O^z_{%$(Int(N/2-r/2+2))+\ell} \rangle$"))

					plt.draw()

					if save_plot
						py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "spt_correlations/"))
						fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "spt_correlations/", "$(chop(data_filenames[i], tail=5))_energy=$(data["energies"][j])" * ".png"))
						ax.cla()
					end
				end
			end

			if plot_grid_evs

				for j=1:nE

					evs_2[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(data["evs_2"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					evs_2[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean(data["evs_2"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					evs_2[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_2"][j, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

					evs_4[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 4*numpy.mean(data["evs_4"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					evs_4[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 4*numpy.mean(data["evs_4"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					evs_4[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 4*numpy.mean(data["evs_4"][j, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

					# evs_6[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*numpy.mean(data["evs_6"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_6[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*numpy.mean(data["evs_6"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_6[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*numpy.mean(data["evs_6"][j, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

					# evs_8[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*4*numpy.mean(data["evs_8"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_8[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*4*numpy.mean(data["evs_8"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_8[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*4*numpy.mean(data["evs_8"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

					# evs_10[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*16*numpy.mean(data["evs_10"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_10[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*16*numpy.mean(data["evs_10"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_10[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*16*numpy.mean(data["evs_10"][j, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

				end
			end

		end

	end

	if plot_grid_evs

		for j=1:nE

			data = pandas.DataFrame(evs_2[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{2}_{x}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if save_plot
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "x2_k=$j.png"))
				fig.clf()
			end

			data = pandas.DataFrame(evs_2[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{2}_{y}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if save_plot
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "y2_k=$j.png"))
				fig.clf()
			end

			data = pandas.DataFrame(evs_2[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{2}_{z}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if save_plot
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "z2_k=$j.png"))
				fig.clf()
			end

			data = pandas.DataFrame(evs_4[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{4}_{x}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if save_plot
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "x4_k=$j.png"))
				fig.clf()
			end

			data = pandas.DataFrame(evs_4[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{4}_{y}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if save_plot
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "y4_k=$j.png"))
				fig.clf()
			end

			data = pandas.DataFrame(evs_4[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{4}_{z}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if save_plot
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "z4_k=$j.png"))
				fig.clf()
			end

			# data = pandas.DataFrame(evs_6[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{6}_{x}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "x6_k=$j.png"))
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_6[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{6}_{y}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "y6_k=$j.png"))
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_6[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{6}_{z}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "z6_k=$j.png"))
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_8[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{8}_{x}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "x8_k=$j.png"))
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_8[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{8}_{y}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "y8_k=$j.png"))
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_8[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{8}_{z}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "z8_k=$j.png"))
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_10[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{10}_{x}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "x10_k=$j.png"))
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_10[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{10}_{y}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "y10_k=$j.png"))
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_10[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{10}_{z}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if save_plot
			# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/"))
			# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "grid_evs/" * "z10_k=$j.png"))
			# 	fig.clf()
			# end
		end
	end

	if plot_degeneracies

		data = pandas.DataFrame(degens[end:-1:1, :], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		seaborn.heatmap(data, cbar_kws=Dict("label" => latexstring(L"$n \, | \, E^{\,}_{m} - E^{\,}_{0} < \Delta \varepsilon, \quad 0 \leq m < n$")))

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.title(latexstring(L"$\Delta \varepsilon = %$(epsDegen)$"))

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "degeneracies/"))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "degeneracies/" * "eps=$(epsDegen).png"))
			ax.cla()
		end
	end

	if plot_lines_c

		for j=1:nE

			for l=1:length(lam)

				ax.errorbar(theta, cs[j, l, :, 1], cs[j, l, :, 2], ecolor="steelblue", elinewidth=0.5, color="mediumvioletred", marker=".", linewidth=0.5, markersize=5, zorder=105)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.xlabel(latexstring(L"$\theta$"))

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_c/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_c/" * "lam=$(lam[l])_k=$j.png"))
					ax.cla()
				end
			end
		end

		for j=1:nE

			for t=1:length(theta)

				ax.errorbar(lam, cs[j, :, t, 1], cs[j, :, t, 2], ecolor="steelblue", elinewidth=0.5, color="mediumvioletred", marker=".", linewidth=0.5, markersize=5, zorder=105)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.xlabel(latexstring(L"$\theta$"))

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_c/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_c/" * "theta=$(theta[t])_k=$j.png"))
					ax.cla()
				end
			end
		end
	end
	
	if plot_binder

		for j=1:nE

			for l=1:length(lam)
				
				ax.plot(theta, numpy.abs(binder[j, l, :, 1]), marker=".", zorder=105, label=latexstring(L"$|m^{x}_{\mathrm{sta}}|$"))

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.2), borderaxespad=0, ncol=3).set_zorder(101)

				plt.xlabel(latexstring(L"$\theta$"))

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "binder/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "binder/" * "lam=$(lam[l])_k=$j.png"))
					ax.cla()
				end
			end
		end
	end

	if plot_lines_mag

		for j=1:nE

			for l=1:length(lam)
				
				py"""
				def f(theta, a, b, c):
					import numpy

					return a*numpy.power(numpy.abs(theta-b), c)*numpy.heaviside(b-theta, 1)
				"""

				ax.plot(theta, numpy.abs(mags[j, l, :, 1]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{x}|$"))
				ax.plot(theta, numpy.abs(mags[j, l, :, 2]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{x}|$"))
				ax.plot(theta, numpy.abs(mags[j, l, :, 3]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{y}|$"))
				ax.plot(theta, numpy.abs(mags[j, l, :, 4]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{y}|$"))
				ax.plot(theta, numpy.abs(mags[j, l, :, 5]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{z}|$"))
				ax.plot(theta, numpy.abs(mags[j, l, :, 6]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{z}|$"))

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.2), borderaxespad=0, ncol=3).set_zorder(101)

				plt.xlabel(latexstring(L"$\theta$"))

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_mag/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_mag/" * "lam=$(lam[l])_k=$j.png"))
					ax.cla()
				end
			end

			# for t=1:length(theta)
				
			# 	py"""
			# 	def f(lam, a, b, c):
			# 		import numpy

			# 		return a*numpy.power(numpy.abs(lam-b), c)*numpy.heaviside(b-lam, 1)
			# 	"""

			# 	ax.plot(lam, numpy.abs(mags[j, :, t, 1]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{x}|$"))
			# 	ax.plot(lam, numpy.abs(mags[j, :, t, 2]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{x}|$"))
			# 	ax.plot(lam, numpy.abs(mags[j, :, t, 3]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{y}|$"))
			# 	ax.plot(lam, numpy.abs(mags[j, :, t, 4]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{y}|$"))
			# 	ax.plot(lam, numpy.abs(mags[j, :, t, 5]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{z}|$"))
			# 	ax.plot(lam, numpy.abs(mags[j, :, t, 6]), marker=".", zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{z}|$"))

			# 	ax.minorticks_on()
			# 	ax.grid(which="minor", linewidth=0.2)
			# 	ax.grid(which="major", linewidth=0.6)
			# 	ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.2), borderaxespad=0, ncol=3).set_zorder(101)

			# 	plt.xlabel(latexstring(L"$\lambda$"))

			# 	plt.draw()

			# 	if save_plot
			# 		py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_mag/"))
			# 		fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_mag/" * "theta=$(theta[t])_k=$j.png"))
			# 		ax.cla()
			# 	end
			# end
		end
	end

	if plot_lines_correlations_ft
		
		for j=1:nE

			# for l=1:length(lam)

			# 	color = "tab:blue"
			# 	ax.plot(theta, ft_maxs_corrs_z[j, l, :, 1], marker="x", markersize=2, zorder=100, color=color)
			# 	ax.set_xlabel(latexstring(L"$\theta$"))
			# 	ax.set_ylabel(latexstring(L"$\mathrm{frequency}$"), color=color)
			# 	ax.tick_params(axis="y", labelcolor=color)
				
			# 	colortwin = "tab:orange"
			# 	axtwin = ax.twinx()
			# 	axtwin.plot(theta, ft_maxs_corrs_z[j, l, :, 2], marker="x", markersize=3, zorder=101, color=colortwin)
			# 	axtwin.set_ylabel(latexstring(L"$\mathrm{ampltitude}$"), color=colortwin)
			# 	axtwin.tick_params(axis="y", labelcolor=colortwin)

			# 	ax.minorticks_on()
			# 	ax.grid(which="minor", linewidth=0.2)
			# 	ax.grid(which="major", linewidth=0.6)

			# 	plt.draw()

			# 	if save_plot
			# 		py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_correlations_ft/"))
			# 		fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_correlations_ft/" * "lam=$(lam[l])_k=$j.png"))
			# 		ax.cla()
			# 	end
			# end

			for t=1:length(theta)

				color = "tab:blue"
				ax.plot(lam, ft_maxs_corrs_z[j, :, t, 1], marker="x", markersize=2, zorder=100, color=color)
				ax.set_xlabel(latexstring(L"$\lambda$"))
				ax.set_ylabel(latexstring(L"$\mathrm{frequency}$"), color=color)
				ax.tick_params(axis="y", labelcolor=color)
				
				colortwin = "tab:orange"
				axtwin = ax.twinx()
				axtwin.plot(lam, ft_maxs_corrs_z[j, :, t, 2], marker="x", markersize=3, zorder=101, color=colortwin)
				axtwin.set_ylabel(latexstring(L"$\mathrm{ampltitude}$"), color=colortwin)
				axtwin.tick_params(axis="y", labelcolor=colortwin)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_correlations_ft/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_correlations_ft/" * "theta=$(theta[t])_k=$j.png"))
					ax.cla()
					axtwin.cla()
				end
			end
		end
	end

	if plot_lines_spins_ft

		for j=1:nE

			for l=1:length(lam)

				color = "tab:blue"
				ax.plot(theta, ft_maxs_spins_z[j, l, :, 1], marker="x", markersize=5, zorder=100, color=color)
				ax.set_xlabel(latexstring(L"$\theta$"))
				ax.set_ylabel(latexstring(L"$\mathrm{frequency}$"), color=color)
				ax.tick_params(axis="y", labelcolor=color)
				
				colortwin = "tab:orange"
				axtwin = ax.twinx()
				axtwin.plot(theta, ft_maxs_spins_z[j, l, :, 2], marker="x", markersize=5, zorder=101, color=colortwin)
				axtwin.set_ylabel(latexstring(L"$\mathrm{ampltitude}$"), color=colortwin)
				axtwin.tick_params(axis="y", labelcolor=colortwin)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_spins_ft/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_spins_ft/" * "lam=$(lam[l])_k=$j.png"))
					ax.cla()
				end
			end

			for t=1:length(theta)

				color = "tab:blue"
				ax.plot(lam, ft_maxs_spins_z[j, :, t, 1], marker="x", markersize=5, zorder=100, color=color)
				ax.set_xlabel(latexstring(L"$\lambda$"))
				ax.set_ylabel(latexstring(L"$\mathrm{frequency}$"), color=color)
				ax.tick_params(axis="y", labelcolor=color)
				
				colortwin = "tab:orange"
				axtwin = ax.twinx()
				axtwin.plot(lam, ft_maxs_spins_z[j, :, t, 2], marker="x", markersize=5, zorder=101, color=colortwin)
				axtwin.set_ylabel(latexstring(L"$\mathrm{ampltitude}$"), color=colortwin)
				axtwin.tick_params(axis="y", labelcolor=colortwin)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_spins_ft/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lines_spins_ft/" * "theta=$(theta[t])_k=$j.png"))
					ax.cla()
					axtwin.cla()
				end
			end
		end
	end

	if plot_scaling_gaps | plot_lines_gaps
		push!(diffsNs, diffs)
	end
	if plot_scaling_en
		push!(ENs, Es)
	end

end

if plot_scaling_gaps

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

	for l=1:length(lam)
		for t=1:length(theta)

			for k=1:nE-1
								
				py"""
				def f(N, a, b):
					import numpy 

					return a/N + b
				"""

				py"""
				def r2(y, fit):
					import numpy

					ss_res = numpy.sum((y-fit)**2)
					ss_tot = numpy.sum((y-numpy.mean(y))**2)

					return (1 - ss_res/ss_tot)
				"""

				popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffsNs[:, l, t, k])
				perr = numpy.sqrt(numpy.diag(pcov))

				r2_ = py"r2"(diffsNs[:, l, t, k], py"f"(Ns, popt...))

				ax.plot(1 ./Ns, py"f"(Ns, popt...), color="mediumvioletred", linestyle="--", zorder=99, label=latexstring(L"$\frac{%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3))}{N} + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3))),\ R^2 = %$(round(r2_; digits=3))$"))

				ax.scatter(1 ./Ns, diffsNs[:, l, t, k], marker="x", color="steelblue", zorder=100)

				ax.set_xlabel(L"$1/2N$")
				ax.set_ylabel(L"$E_1-E_0$")

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0).set_zorder(101)

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, "scaling/" * "lam=$(lam[l])_theta=$(theta[t])/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "scaling/" * "lam=$(lam[l])_theta=$(theta[t])/" * "E$(k)-E$(k-1).png"))
					ax.cla()
				end
			end
		end
	end
end

if plot_lines_gaps

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

	for n=1:length(Ns)
		for t=1:length(theta)

			for k=1:nE-1
								
				py"""
				def f(N, a, b):
					import numpy 

					return a/N + b
				"""

				py"""
				def r2(y, fit):
					import numpy

					ss_res = numpy.sum((y-fit)**2)
					ss_tot = numpy.sum((y-numpy.mean(y))**2)

					return (1 - ss_res/ss_tot)
				"""

				# popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffsNs[n, :, t, k])
				# perr = numpy.sqrt(numpy.diag(pcov))

				# r2_ = py"r2"(diffsNs[n, :, t, k], py"f"(Ns, popt...))

				# ax.plot(lam, py"f"(Ns, popt...), color="mediumvioletred", linestyle="--", zorder=99, label=latexstring(L"$\frac{%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3))}{N} + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3))),\ R^2 = %$(round(r2_; digits=3))$"))

				ax.scatter(lam, diffsNs[n, :, t, k], marker="x", color="steelblue", zorder=100)

				ax.set_xlabel(L"$\lambda$")
				ax.set_ylabel(L"$E_1-E_0$")

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0).set_zorder(101)

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, "scaling/" * "N=$(Ns[n])_theta=$(theta[t])/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "scaling/" * "N=$(Ns[n])_theta=$(theta[t])/" * "E$(k)-E$(k-1).png"))
					ax.cla()
				end
			end
		end
	end
end

if plot_scaling_en

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	ENs = numpy.array(ENs)[indSort, :, :, :]

	for l=1:length(lam)
		for t=1:length(theta)

			for k=1:nE
								
				py"""
				def f(N, a, b):
					import numpy 

					return a/N + b
				"""								
				py"""
				def g(N, a, b):
					import numpy 

					return a*N - b*0.713*numpy.pi/(6*N)
				"""

				py"""
				def r2(y, fit):
					import numpy

					ss_res = numpy.sum((y-fit)**2)
					ss_tot = numpy.sum((y-numpy.mean(y))**2)

					return (1 - ss_res/ss_tot)
				"""

				popt, pcov = scipy.optimize.curve_fit(py"g", Ns, ENs[:, l, t, k])
				perr = numpy.sqrt(numpy.diag(pcov))

				println(Ns,  ENs[:, l, t, k])
				println(popt, perr)

				r2_ = py"r2"(ENs[:, l, t, k], py"g"(Ns, popt...))

				ax.plot(Ns, py"g"(Ns, popt...), color="mediumvioletred", linestyle="--", zorder=99, label=latexstring(L"$(%$(round(popt[1]; digits=2)) \pm %$(round(perr[1]; digits=2))) \, (2N) - \frac{\pi \, (%$(round(popt[2]; digits=2)) \pm %$(round(perr[2]; digits=2)))}{6 \, (2N)},\ R^2 = %$(round(r2_; digits=3))$"))

				ax.scatter(Ns, ENs[:, l, t, k], marker="x", color="steelblue", zorder=100)

				ax.set_xlabel(latexstring(L"$2N$"))
				ax.set_ylabel(latexstring(L"$E_{%$(k-1)}$"))

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0).set_zorder(101)

				plt.draw()

				if save_plot
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, "energies/" * "lam=$(lam[l])_theta=$(theta[t])/"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "energies/" * "lam=$(lam[l])_theta=$(theta[t])/" * "E$(k-1).png"))
					ax.cla()
				end
			end
		end
	end
end
