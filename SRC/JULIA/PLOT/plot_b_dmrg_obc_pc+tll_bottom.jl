ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PC+TLL/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

r = 128

start_x = 5
end_x = 100

start_y = 20
end_y = 40

start_z = 18
end_z = 40

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
plt.subplots_adjust(wspace=0.4)

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

		corrX = data["corrs"][1, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrY = data["corrs"][1, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrZ = data["corrs"][1, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

		corrZ = corrZ - data["mags"][1, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)] * data["mags"][1, 3, Int(N/2)-Int(r/2)]

		corrX = corrX .* [(-1)^s for s in range(1, length(corrX))]

		sites_x = range(1, r)
		sites_y = range(1, r)
		sites_z = range(1, r)

		select_x = numpy.where(corrX .> 1e-15)[1] .+ 1

		corrX = corrX[select_x]

		sites_x = sites_x[select_x]
		
		p, = ax[1].plot(sites_x, corrX, marker="o", color=COLORS[1], lw=0, markersize=5, label=latexstring(L"$(-1)^{s}\, C^{x}$"), zorder=101)
		p, = ax[1].plot(sites_y, numpy.abs(corrY), marker="o", color=COLORS[4], lw=0, markersize=5, label=latexstring(L"$(-1)^{s}\, C^{x}$"), zorder=101)
		p, = ax[1].plot(sites_z, numpy.abs(corrZ), marker="o", color=COLORS[7], lw=0, markersize=5, label=latexstring(L"$(-1)^{s}\, C^{x}$"), zorder=101)

		p, = ax[2].plot(sites_y, corrY, marker="o", color=COLORS[4], lw=0, markersize=5, label=latexstring(L"$(-1)^{s}\, C^{y}$"), zorder=103)

		p, = ax[3].plot(sites_z, corrZ, marker="o", color=COLORS[7], lw=0, markersize=5, label=latexstring(L"$C^{z,\mathrm{c}}$"), zorder=103)

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

		poptx, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_x)), numpy.log(_corrX))
		perrx = numpy.sqrt(numpy.diag(pcov))

		a = poptx[1]
		da = perrx[1]

		println("x : ", a, " ", da)

		p, = ax[1].plot(_sites_x, numpy.exp(py"f"(collect(numpy.log(_sites_x)), poptx...)), color=COLORS[4], linewidth=1, linestyle="-", marker="none", zorder=110, label=latexstring(L"fit$^{\,}_{x}$"))

		py"""
		import numpy
		
		def g(x, a, b, c, d, e):

			return a * x ** b * numpy.cos(x * 2*numpy.pi * c + d) + e
		"""

		popty, pcov = scipy.optimize.curve_fit(py"g", collect(_sites_y), _corrY, [0.1, -1, 0.26, 1, 0])
		perry = numpy.sqrt(numpy.diag(pcov))

		println("y : ", popty, perry)

		p, = ax[2].plot(_sites_y, py"g"(collect(_sites_y), popty...), color="k", linewidth=1, linestyle="-", marker="none", zorder=110, label=latexstring(L"fit$^{\,}_{\mathrm{q}}$"))

		py"""
		import numpy
		
		def h(x, a, b, c, d, e):

			return a * x ** b * numpy.cos(x * 2*numpy.pi * c + d) + e
		"""

		poptz, pcov = scipy.optimize.curve_fit(py"h", collect(_sites_z), _corrZ, [0.5, -1, 0.259, 1, 0])
		perrz = numpy.sqrt(numpy.diag(pcov))

		println("z : ", poptz, perrz)

		p, = ax[3].plot(_sites_z, py"h"(collect(_sites_z), poptz...), color="k", linewidth=1, linestyle="-", marker="none", zorder=110, label=latexstring(L"fit$^{\,}_{\mathrm{q}}$"))

		ax[1].set_xscale("log")
		ax[1].set_yscale("log")

		ax[2].set_ylim([-0.005, 0.005])
		
		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "corr.pdf"))
		end

		plt.close()
	end
end

