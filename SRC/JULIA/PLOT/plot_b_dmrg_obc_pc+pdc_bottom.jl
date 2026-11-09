ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PC+PDC/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

r = 128

start_x = 5
end_x = 64

start_y = 5
end_y = 64

start_z = 10
end_z = 10

fig, ax = plt.subplots(1, 2, figsize=(5, 2))
plt.subplots_adjust(wspace=0.15)

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
		corrY = data["corrs"][1, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrZ = data["corrs"][1, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

		corrZ = corrZ - data["mags"][1, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)] * data["mags"][1, 3, Int(N/2)-Int(r/2)]

		sites_x = range(1, r)
		sites_y = range(1, r)
		sites_z = range(1, r)

		select_x = numpy.where(numpy.abs(corrX) .> 1e-15)[1] .+ 1
		select_y = numpy.where(numpy.abs(corrY) .> 1e-15)[1] .+ 1

		corrX = corrX[select_x]
		corrY = corrY[select_y]

		sites_x = sites_x[select_x]
		sites_y = sites_y[select_y]

		for spine in ax[1].spines.values()
			spine.set_zorder(1000)
		end
		for spine in ax[2].spines.values()
			spine.set_zorder(1000)
		end
			
		p, = ax[1].plot(sites_x, corrX, marker="o", markerfacecolor=COLORS[4], linewidth=0, markersize=5, label=latexstring(L"$C^{x}_{s^{\,}_{0},s^{\,}_{0}+s}$"), zorder=103)

		p, = ax[1].plot(sites_y, corrY, marker="o", markerfacecolor=COLORS[3], linewidth=0, markersize=5, label=latexstring(L"$C^{y}_{s^{\,}_{0},s^{\,}_{0}+s}$"), zorder=102)

		p, = ax[2].plot(sites_z, corrZ, marker="o", markerfacecolor=COLORS[1], linewidth=0, markersize=5, label=latexstring(L"$C^{z,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s}$"), zorder=101)

		_sites_x = sites_x[start_x:end+1-end_x]
		_corrX = corrX[start_x:end+1-end_x]

		_sites_y = sites_y[start_y:end+1-end_y]
		_corrY = corrY[start_y:end+1-end_y]

		_sites_z = sites_z[start_z:end+1-end_z]
		_corrZ = corrZ[start_z:end+1-end_z]

		py"""
		import numpy
		
		def f(x, a, b):

			return a * x + b
		"""

		py"""
		import numpy
		
		def g(x, a, b, c, d):

			return a * x ** b * numpy.cos(x * c * 2*numpy.pi) + d
		"""

		axin = ax[1].inset_axes([0.55, 0.17, 0.4, 0.4])
		axin.set_zorder(110)

		p, = axin.plot(_sites_x, numpy.abs(_corrX), marker="o", markerfacecolor=COLORS[4], linewidth=0, markersize=3, zorder=113)

		poptx, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_x)), numpy.log(numpy.abs(_corrX)), [-1, 0])
		perrx = numpy.sqrt(numpy.diag(pcov))

		println("x : ", poptx, perrx)

		p, = axin.plot(_sites_y, numpy.abs(_corrY), marker="o", markerfacecolor=COLORS[3], linewidth=0, markersize=3, zorder=111)

		popty, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_y)), numpy.log(numpy.abs(_corrY)), [-1, 0])
		perry = numpy.sqrt(numpy.diag(pcov))

		println("y : ", popty, perry)

		poptz, pcov = scipy.optimize.curve_fit(py"g", collect(_sites_z), _corrZ, [0.3, -1, 0.3, 0])
		perrz = numpy.sqrt(numpy.diag(pcov))

		println("z : ", poptz, perrz)

		x = numpy.linspace(0, 100, 1000)
		env = numpy.abs(poptz[1]) .* x .^ poptz[2]

		p, = ax[2].plot(x, env, color=COLORS[5], linewidth=0.6, linestyle="-", marker="none", label=L"{}\pm A^{z,\mathrm{c}}_{s^{\,}_{0}}\, s^{-\eta^{z,\mathrm{c}}}", zorder=91)
		p, = ax[2].plot(x, -env, color=COLORS[5], linewidth=0.6, linestyle="-", marker="none", zorder=91)

		ax[2].fill_between(
			x,
			-env,
			env,
			facecolor=COLORS[5],
			alpha=0.2,
			linewidth=0,
			zorder=90
		)

		axin.set_xscale("log")
		axin.set_yscale("log")

		ax[1].set_xlabel(L"$s$")
		ax[2].set_xlabel(L"$s$")
		
		ax[1].set_xlim(0, 85)
		ax[2].set_xlim(0, 85)
		ax[1].set_ylim(-0.05, 0.02)
		ax[2].set_ylim(-0.05, 0.02)
		
		ax[1].set_yticks([-0.05, 0])
		ax[2].set_yticks([])
		axin.set_xticks([10])
		axin.set_yticks([0.01])

		axin.tick_params(labelsize=10)

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
			loc="lower left",
			ncol=1,
			handletextpad=0.3,
			borderpad=0.3,
			labelspacing=0.5,
			bbox_to_anchor=(-0.13, 0.13)).set_zorder(1000)		
		ax[2].legend(
			loc="lower right",
			ncol=1,
			handletextpad=0.7,
			borderpad=0.3,
			labelspacing=0.7,
			bbox_to_anchor=(1.05, 0.1)).set_zorder(1000)
		
		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "corr.pdf"))
		end

		plt.close()
	end
end

