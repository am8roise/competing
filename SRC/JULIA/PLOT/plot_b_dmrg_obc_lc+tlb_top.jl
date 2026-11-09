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
simul_name = "LC+TLB/TOP"

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
		println(corr_Z)
		corrs_ends[3, vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = corr_Z[1:n_e]

		diffs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = E[2:n_e] .- E[1:n_e-1]
	end

	push!(corrs_ends_Ns, corrs_ends)
	push!(diffs_Ns, diffs)
end

corrs_ends_Ns = numpy.array(corrs_ends_Ns)
diffs_Ns = numpy.array(diffs_Ns)

for n in 1:length(Ns)
	for t in 1:length(theta)

		fig = plt.figure(figsize=(4, 4))
		ax1 = plt.subplot2grid((2, 1), (0, 0))
		ax2 = plt.subplot2grid((2, 1), (1, 0))

		p1, = ax2.plot(lam .- pi/2, corrs_ends_Ns[n, 3, :, t, 1], color=COLORS[1], markeredgewidth=1.5, markersize=10*1.2, marker=L"$/$",  ls="-", lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
		p2, = ax2.plot(lam .- pi/2, corrs_ends_Ns[n, 3, :, t, 2], color=COLORS[3], markeredgewidth=1.5, markersize=10*1.2, marker=L"$\setminus$",  ls=":", lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
		p3, = ax1.plot(lam .- pi/2, corrs_ends_Ns[n, 3, :, t, 3], color=COLORS[4], markeredgewidth=1.5, markersize=7*1.2,  marker="o", ls="-", lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
		p4, = ax1.plot(lam .- pi/2, corrs_ends_Ns[n, 3, :, t, 4], color=COLORS[6], markeredgewidth=1.5, markersize=3*1.2,  marker="o", ls=":", lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))

		py"""
		def f(x, a, b):
			return a * x ** b - 1
		"""

		popt, pcov = scipy.optimize.curve_fit(py"f", lam[1:5] .- pi/2, corrs_ends_Ns[n, 3, 1:5, t, 1], [1, 2])
		perr = numpy.sqrt(numpy.diag(pcov))

		println(popt, perr)

		extra = numpy.linspace(0, lam[1:5] .- pi/2, 100)
		p, = ax2.plot(extra, py"f"(extra, popt...), lw=1, zorder=10, color="k", marker="none", linestyle="--")
					
		ax1.minorticks_on()
		ax2.minorticks_on()

		ax2.set_xlabel(latexstring(L"$\lambda-\pi/2$"))
		ax1.set_ylabel(latexstring(L"$C^{z}_{1,2N}[\Psi]$"))
		ax2.set_ylabel(latexstring(L"$C^{z}_{1,2N}[\Psi]$"))
		
		ax1.set_xticklabels([])
		ax2.set_xticklabels([L"$0$", L"$0.05$", L"$0.1$", L"$0.15$"])

		ax2.set_yticks([-1, -0.95])
		ax2.set_yticklabels([L"$-1$", L"$-0.95$"])
		ax1.set_yticks([0.95, 1])
		ax1.set_yticklabels([L"$+0.95$", L"$+1$"])

		ax1.set_xlim(0, 0.17)
		ax2.set_xlim(0, 0.17)

		leg1 = ax1.legend(handles=[p3, p4], loc="lower left", ncol=2, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8)
		leg1.set_zorder(110)

		ax2.legend(handles=[p1, p2], loc="upper left", ncol=2, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)
		leg2 = ax1.legend(handles=[p], labels=[latexstring(L"(%$(format_value(popt[1], perr[1])))\times(\lambda-\pi/2)^{%$(format_value(popt[2], perr[2]))}")], loc="lower right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1, 1.05)).set_zorder(110)

		ax1.add_artist(leg1)

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "corr.pdf"))
			fig.clf()
		end

		plt.close()
	end
end
