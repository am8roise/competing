ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PC+PDC/TOP"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

r = 96

start_x = 20
end_x = 10

start_z = 10
end_z = 1

fig, ax = plt.subplots(1, 2, figsize=(5, 2))
plt.subplots_adjust(wspace=0.3)

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		mZ = data["mags"][1, 3, :]
		println("mz = ", numpy.mean(mZ))

		corrX = data["corrs"][1, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrZ = data["corrs"][1, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

		corrZ = corrZ - data["mags"][1, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)] * data["mags"][1, 3, Int(N/2)-Int(r/2)]

		sites_x = range(1, r)
		sites_z = range(1, r)

		select_x = numpy.where(numpy.abs(corrX) .> 2e-13)[1] .+ 1
		select_z = numpy.where(numpy.abs(corrZ) .> 2e-13)[1] .+ 1

		corrX = corrX[select_x]
		corrZ = corrZ[select_z]

		sites_x = sites_x[select_x]
		sites_z = sites_z[select_z]

		for spine in ax[1].spines.values()
			spine.set_zorder(1000)
		end
		for spine in ax[2].spines.values()
			spine.set_zorder(1000)
		end

		_sites_x = sites_x[start_x:end+1-end_x]
		_corrX = corrX[start_x:end+1-end_x]

		_sites_z = sites_z[start_z:end+1-end_z]
		_corrZ = corrZ[start_z:end+1-end_z]

		py"""
		import numpy
		
		def f(x, a, b):

			return a * x + b
		"""

		py"""
		import numpy
		
		def g(x, a, b, c):

			return a * numpy.cos(x * b * 2*numpy.pi) + c
		"""

		poptx, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_x), numpy.log(numpy.abs(_corrX)), [-2.5, 0])
		perrx = numpy.sqrt(numpy.diag(pcov))

		a = 1 / poptx[1]
		da = a ^ 2 * perrx[1]

		println("x : ", a, " ", da)

		p, = ax[1].plot(sites_x, corrX, marker="o", markerfacecolor=COLORS[4], linewidth=0, markersize=7, label=latexstring(L"$|C^{x}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=103)

		poptz, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_z), numpy.log(numpy.abs(_corrZ)), [-2.5, 0])
		perrz = numpy.sqrt(numpy.diag(pcov))

		a = 1 / poptz[1]
		da = a ^ 2 * perrz[1]

		println("z : ", a, " ", da)

		p, = ax[1].plot(sites_z, corrZ, marker="o", markerfacecolor=COLORS[1], linewidth=0, markersize=7, label=latexstring(L"$|C^{z,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=102)

		p, = ax[2].plot(sites_z, corrZ .* numpy.exp(-sites_z / a), marker="o", markerfacecolor=COLORS[1], linewidth=0, markersize=7, label=latexstring(L"$e^{+s/\xi^{z,\mathrm{c}}}\, C^{z,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s}$"), zorder=102)

		poptz, pcov = scipy.optimize.curve_fit(py"g", collect(_sites_z), _corrZ .* numpy.exp(-_sites_z / a), [0.3, 0.22, 0])
		perrz = numpy.sqrt(numpy.diag(pcov))

		println(poptz, perrz)

		p, = ax[2].plot(_sites_z, py"g"(_sites_z, poptz...), marker="none", linewidth=0.5, linestyle="-", color=COLORS[5], label=latexstring(L"A^{z,\mathrm{c}}_{s^{\,}_{0}}\,\cos\big(q^{z,\mathrm{c}}\,s+\phi^{z,\mathrm{c}}_{s}\big)"), zorder=103)

		ax[1].set_xlabel(L"$s$")
		ax[2].set_xlabel(L"$s$")

		ax[1].set_yscale("log")

		ax[2].set_ylim(-0.5, 0.7)
			
		ax[1].set_yticks([1e-14, 1e-10, 1e-6, 1e-2])
		ax[2].set_yticks([-0.3, 0, 0.3])

		ax[1].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([10.0 ^ k for k in -16:0]))
		ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

		ax[1].text(
			0.97, 0.97,
			"(a)",
			transform=ax[1].transAxes,
			ha="right",
			va="top",
			fontsize=12)
		ax[2].text(
			0.97, 0.97,
			"(b)",
			transform=ax[2].transAxes,
			ha="right",
			va="top",
			fontsize=12)
		
		ax[1].legend(
			loc="upper center",
			ncol=1,
			handletextpad=0.3,
			borderpad=0.3,
			labelspacing=0.7,
			bbox_to_anchor=(0.55, 1.1)).set_zorder(1000)			
		ax[2].legend(
			loc="upper left",
			ncol=1,
			handletextpad=0.3,
			borderpad=0.3,
			labelspacing=0.7,
			bbox_to_anchor=(-0.2, 1.1)).set_zorder(1000)		
		
		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "corr.pdf"))
		end

		plt.close()
	end
end

