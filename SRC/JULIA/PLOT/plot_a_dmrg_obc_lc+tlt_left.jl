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
simul_name = "LC+TLT/LEFT"

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

for n in 1:length(Ns)
	for t in 1:length(theta)

		fig, ax = plt.subplots(2, 1, figsize=(2.7, 3))
		plt.subplots_adjust(hspace=0.05)

		p1, = ax[2].plot(pi/2 .- lam[end:-1:1], corrs_ends_Ns[n, 3, end:-1:1, t, 1], marker="o", color=COLORS[5], lw=0, markersize=10, label=latexstring(L"$\Psi^{\,}_{0}$"), zorder=101)
		p2, = ax[2].plot(pi/2 .- lam[end:-1:1], corrs_ends_Ns[n, 3, end:-1:1, t, 2], marker="o", color=COLORS[6], lw=0, markersize=5, label=latexstring(L"$\Psi^{\,}_{1}$"), zorder=102)
		p3, = ax[1].plot(pi/2 .- lam[end:-1:1], corrs_ends_Ns[n, 3, end:-1:1, t, 3], marker="o", color=COLORS[3], lw=0, markersize=10, label=latexstring(L"$\Psi^{\,}_{2}$"), zorder=103)
		p4, = ax[1].plot(pi/2 .- lam[end:-1:1], corrs_ends_Ns[n, 3, end:-1:1, t, 4], marker="o", color=COLORS[4], lw=0, markersize=5, label=latexstring(L"$\Psi^{\,}_{3}$"), zorder=104)

		py"""
		def f(x, a):
			return a * x ** 2 - 1
		"""
		py"""
		def g(x, a):
			return a * x ** 2 + 1
		"""

		popt, pcov = scipy.optimize.curve_fit(py"g", pi/2 .- lam[end:-1:end-7], corrs_ends_Ns[n, 3, end:-1:end-7, t, 3], [-1])
		perr = numpy.sqrt(numpy.diag(pcov))

		println(popt, perr)

		extra = numpy.linspace(0, pi/2 .- lam[1], 100)
		p, = ax[1].plot(extra, py"g"(extra, popt...), lw=2, zorder=90, color=COLORS[2], marker="none", linestyle="-", label=latexstring(L"-A^{\,}_{\Psi}\,\bigl(\frac{\pi}{2}-\lambda\bigr)^{2}+1"))

		popt, pcov = scipy.optimize.curve_fit(py"f", pi/2 .- lam[end:-1:end-7], corrs_ends_Ns[n, 3, end:-1:end-7, t, 1], [1])
		perr = numpy.sqrt(numpy.diag(pcov))

		println(popt, perr)

		extra = numpy.linspace(0, pi/2 .- lam[1], 100)
		p_, = ax[2].plot(extra, py"f"(extra, popt...), lw=2, zorder=90, color=COLORS[2], marker="none", linestyle=(0.5, (0, 1.5)), label=latexstring(L"+A^{\,}_{\Psi}\,\bigl(\frac{\pi}{2}-\lambda\bigr)^{2}-1"))

		leg = ax[2].legend(handles=[p_, p],
			loc="upper center",
			ncol=1,
			labelspacing=0.4,
			bbox_to_anchor=(0.42, 1.23))
		leg.set_zorder(1001)

		ax[2].legend(handles=[p1, p3, p2, p4],
			loc="lower left",
			ncol=2,
			borderpad=0.6,
			bbox_to_anchor=(-0.2, 0.28)).set_zorder(1000)

		ax[2].add_artist(leg)

		# ========== BROKEN Y AXIS
		ax[1].spines.bottom.set_visible(false)
		ax[2].spines.top.set_visible(false)

		d = 0.02
		ax[1].plot([-d, d], [-d,d], transform=ax[1].transAxes, lw=0.5, color="k", marker="none", clip_on=false, zorder=1000)
		ax[1].plot([1-d, 1+d], [-d,d], transform=ax[1].transAxes, lw=0.5, color="k", marker="none", clip_on=false, zorder=1000)
		ax[2].plot([-d, d], [1-d,1+d], transform=ax[2].transAxes, lw=0.5, color="k", marker="none", clip_on=false, zorder=1000)
		ax[2].plot([1-d, 1+d], [1-d,1+d], transform=ax[2].transAxes, lw=0.5, color="k", marker="none", clip_on=false, zorder=1000)
		# ==========

		ax[1].text(
			0.99, 0.97,
			"(c)",
			transform=ax[1].transAxes,
			ha="right",
			va="top",
			fontsize=12)

		ax[2].set_xlabel(latexstring(L"$\frac{\pi}{2}-\lambda$"))
		ax[1].set_ylabel(latexstring(L"$C^{z}_{1,2N}[\Psi]$"), y=0)
		
		ax[1].set_xticks([])
		ax[2].set_xticklabels([L"$0$", L"$0.05$", L"$0.1$", L"$0.15$"])

		ax[1].set_yticks([0.95, 1])
		ax[1].set_yticklabels([L"$+0.95$", L"$+1$"])
		ax[2].set_yticks([-1, -0.95])
		ax[2].set_yticklabels([L"$-1$", L"$-0.95$"])

		ax[1].set_xlim(0, 0.17)
		ax[2].set_xlim(0, 0.17)
		ax[1].set_ylim(0.941, 1.01)
		ax[2].set_ylim(-1.01, -0.941)
	
		for spine in ax[1].spines.values()
			spine.set_zorder(999)
		end
		for spine in ax[2].spines.values()
			spine.set_zorder(999)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "corr.pdf"))
			fig.clf()
		end

		plt.close()
	end
end
