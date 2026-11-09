ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")
include("../utils/math.jl")

model_name = "A"
simul_name = "PC+TLT/LEFT"

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

		PyPlot.rc("axes.formatter", limits=[-3, 3])
		PyPlot.rc("axes.formatter", offset_threshold=3)

		fig, ax = plt.subplots(figsize=(2.7, 3))

		py"""
		def f(N, a):
			return a * N
		"""

		popt, pcov = scipy.optimize.curve_fit(py"f", 1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 1] ./ corrs_ends_Ns[:, 3, l, t, 1])
		perr = numpy.sqrt(numpy.diag(pcov))

		print(popt, perr)
		
		p1, = ax.plot(1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 1] ./ corrs_ends_Ns[:, 3, l, t, 1], marker="o", color=COLORS[5], lw=0, markersize=12, label=latexstring(L"$\Psi^{\,}_{0}$"), zorder=101)
		p2, = ax.plot(1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 2] ./ corrs_ends_Ns[:, 3, l, t, 2], marker="o", color=COLORS[6], lw=0, markersize=9, label=latexstring(L"$\Psi^{\,}_{1}$"), zorder=102)
		p3, = ax.plot(1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 3] ./ corrs_ends_Ns[:, 3, l, t, 3], marker="o", color=COLORS[3], lw=0, markersize=6, label=latexstring(L"$\Psi^{\,}_{2}$"), zorder=103)
		p4, = ax.plot(1 ./ Ns, corrs_ends_Ns[:, 1, l, t, 4] ./ corrs_ends_Ns[:, 3, l, t, 4], marker="o", color=COLORS[4], lw=0, markersize=3, label=latexstring(L"$\Psi^{\,}_{3}$"), zorder=104)
		
		ax.set_xlabel(latexstring(L"$(2N)^{-1}$"))
		ax.set_ylabel(latexstring(L"$C^{\mathrm{SPT},x}_{1,2N}[\Psi]\,/\,C^{z}_{1,2N}[\Psi]$"))

		ax.set_xlim(0, 0.07)
		ax.set_ylim(-0.0025, 0)

		ax.set_xticklabels([L"$0$", L"$0.02$", L"$0.04$", L"$0.06$"])
		ax.set_yticks([-0.002, -0.001, 0])

		axin = ax.inset_axes([0.21, 0.12, 0.6, 0.43])
		axin.set_zorder(102)

		axin.plot(1 ./ Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 1]), marker="o", color=COLORS[5], lw=0, markersize=12, label=latexstring(L"$\Psi^{\,}_{0}$"), zorder=101)
		axin.plot(1 ./ Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 2]), marker="o", color=COLORS[6], lw=0, markersize=9, label=latexstring(L"$\Psi^{\,}_{1}$"), zorder=102)
		axin.plot(1 ./ Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 3]), marker="o", color=COLORS[3], lw=0, markersize=6, label=latexstring(L"$\Psi^{\,}_{2}$"), zorder=103)
		axin.plot(1 ./ Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 4]), marker="o", color=COLORS[4], lw=0, markersize=3, label=latexstring(L"$\Psi^{\,}_{3}$"), zorder=104)

		extra = numpy.linspace(0, 1 ./ Ns[1], 100)
		p, = ax.plot(extra, py"f"(extra, popt...), lw=2, zorder=90, color=COLORS[2], marker="none", linestyle="-", label=latexstring(L"B^{\,}_{\Psi}\,(2N)^{-1}"))	

		axin.set_ylabel(latexstring(L"$\big|C^{z}_{1,2N}[\Psi]\big|$"), fontsize=10, labelpad=3)

		axin.set_xlim(0, 0.07)
		axin.set_ylim(0.9891, 0.9899)

		axin.set_xticks([0, 0.02, 0.04, 0.06])
		axin.set_xticklabels([L"$0$", L"$0.02$", L"$0.04$", L"$0.06$"])
		axin.set_yticks([0.9894, 0.9898])

		axin.tick_params(labelsize=10)
		
		axin.yaxis.get_offset_text().set_fontsize(10)

		# axin.yaxis.get_offset_text().set_visible(false)
		# axin.text(0.0, 1.02, raw"$\times 10^{-4} + 0.9890$", transform=axin.transAxes, ha="left", va="bottom", fontsize=10)

		for spine in ax.spines.values()
			spine.set_zorder(999)
		end
		
		ax.text(
			0.98, 0.97,
			"(d)",
			transform=ax.transAxes,
			ha="right",
			va="top",
			fontsize=12)

		leg = ax.legend(handles=[p],
			loc="upper center",
			ncol=1,
			bbox_to_anchor=(0.6, 1.05))
		leg.set_zorder(1001)

		ax.legend(handles=[p1, p2, p3, p4],
			loc="lower right",
			ncol=1,
			bbox_to_anchor=(1.07, 0.12)).set_zorder(1002)
		ax.add_artist(leg)

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "corr.pdf"))
			fig.clf()
		end

		plt.close()

	end
end