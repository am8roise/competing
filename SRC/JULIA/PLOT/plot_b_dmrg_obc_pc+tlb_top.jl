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
simul_name = "PC+TLB/TOP"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

n_e = 5

save_plot = true

Ns = []
folders = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])

	push!(Ns, N)
	push!(folders, folder)
end

ind = numpy.argsort(Ns) .+ 1

Ns = Ns[ind]
folders = folders[ind]

corrs_ends_Ns = []
diffs_Ns = []

for folder in folders

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))
	
	corrs_ends = numpy.zeros((3, length(lam), length(theta), n_e))
	diffs = numpy.zeros((length(lam), length(theta), n_e-1))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		ind = numpy.argsort(data["energies"]) .+ 1
	
		E = data["energies"][ind]

		corr_X = data["boundary"][ind, 1]
		corrs_ends[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = corr_X[1:n_e]

		corr_Y = data["boundary"][ind, 2]
		corrs_ends[2, vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = corr_Y[1:n_e]

		corr_Z = data["corrs"][ind, 3, 1, N]
		corrs_ends[3, vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = corr_Z[1:n_e]

		diffs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = E[2:n_e] .- E[1:n_e-1]
	end

	push!(corrs_ends_Ns, corrs_ends)
	push!(diffs_Ns, diffs)
end

corrs_ends_Ns = numpy.array(corrs_ends_Ns)
diffs_Ns = numpy.array(diffs_Ns)

for l=1:length(lam)	
	for t=1:length(theta)

		cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=repeat([COLORS[1],COLORS[3],COLORS[4],COLORS[6]], 2), N=256)
		PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=repeat(["-", ":", "-", ":"], 2), marker=repeat([L"$/$", L"$\setminus$", "o", "o"], 2), markersize=repeat([10, 10, 7, 3] .*1.2, 2), markeredgewidth=repeat([1.5, 1.5, 1.5, 1.5], 2), color=cmap(numpy.linspace(0, 1, 2*(n_e-1)))))

		PyPlot.rc("axes.formatter", limits=[-3, 3])
		PyPlot.rc("axes.formatter", offset_threshold=3)

		fig, ax = plt.subplots(figsize=(6, 4))
		
		py"""
		def f(N, a):
			return a * N
		"""

		popt, pcov = scipy.optimize.curve_fit(py"f", 1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 1] ./ corrs_ends_Ns[:, 3, l, t, 1])
		perr = numpy.sqrt(numpy.diag(pcov))
		
		p1, = ax.plot(1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 1] ./ corrs_ends_Ns[:, 3, l, t, 1], lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
		p2, = ax.plot(1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 2] ./ corrs_ends_Ns[:, 3, l, t, 2], lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
		p3, = ax.plot(1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 3] ./ corrs_ends_Ns[:, 3, l, t, 3], lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
		p4, = ax.plot(1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 4] ./ corrs_ends_Ns[:, 3, l, t, 4], lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
		
		ax.minorticks_on()		
					
		ax.set_xlabel(latexstring(L"$1/(2N)$"))
		ax.set_ylabel(latexstring(L"$C^{\mathrm{SPT},x}_{1,2N}[\Psi]/C^{z}_{1,2N}[\Psi]$"))

		ax.set_xlim(0, 0.07)
		ax.set_ylim(-0.0008, 0)

		ax.set_xticklabels([L"$0$", L"$0.02$", L"$0.04$", L"$0.06$"])
		ax.set_yticks([-0.0008, 0])

		axins = ax.inset_axes([0.16, 0.16, 0.55, 0.4])
		axins.set_zorder(102)

		axins.plot(1 ./ Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 1]), lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
		axins.plot(1 ./ Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 2]), lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
		axins.plot(1 ./ Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 3]), lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
		axins.plot(1 ./ Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 4]), lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))

		extra = numpy.linspace(0, 1 ./ Ns[1], 100)
		p, = ax.plot(extra, py"f"(extra, popt...), lw=1, zorder=10, color="k", marker="none", linestyle="--")	

		axins.minorticks_on()	

		axins.set_ylabel(latexstring(L"$\big|C^{z}_{1,2N}[\Psi]\big|$"))

		axins.set_xlim(0, 0.07)
		axins.set_ylim(0.9913, 0.992)

		axins.set_xticks([0, 0.02, 0.04, 0.06])
		axins.set_xticklabels([L"$0$", L"$0.02$", L"$0.04$", L"$0.06$"])
		axins.set_yticks([0.9914, 0.9918])

		plt.draw()

		ax.legend(handles=[p1, p2, p3, p4], loc="lower right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)
		axins.legend(handles=[p], labels=[latexstring(L"%$(format_value(popt[1], perr[1]))\times(2N)^{-1}")], loc="center right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1.5, 1.9)).set_zorder(110)

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "corr.pdf"))
			fig.clf()
		end

		plt.close()

	end
end