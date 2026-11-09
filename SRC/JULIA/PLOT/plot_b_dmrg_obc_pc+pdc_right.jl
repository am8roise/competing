ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PC+PDC/RIGHT"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

r = 96

start_x = 10
end_x = 10

start_y = 20
end_y = 10

start_z = 4
end_z = 20

fig, ax = plt.subplots(1, 2, figsize=(7, 3))
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

		mZ = data["mags"][1, 3, :]
		println("mz = ", numpy.mean(mZ))

		corrX = data["corrs"][1, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrY = data["corrs"][1, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrZ = data["corrs"][1, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

		corrZ = corrZ - data["mags"][1, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)] * data["mags"][1, 3, Int(N/2)-Int(r/2)]

		corrX = corrX .* [(-1)^s for s in range(1, length(corrX))]
		corrY = corrY .* [(-1)^s for s in range(1, length(corrY))]

		sites_x = range(1, r)
		sites_y = range(1, r)
		sites_z = range(1, r)

		select_x = numpy.where(corrX .> 1e-15)[1] .+ 1
		select_y = numpy.where(corrY .> 1e-15)[1] .+ 1

		corrX = corrX[select_x]
		corrY = corrY[select_y]

		sites_x = sites_x[select_x]
		sites_y = sites_y[select_y]
		
		p, = ax[1].plot(sites_x, corrX, marker="+", color=COLORS[1], lw=0, markersize=5, markeredgewidth=1.5, label=latexstring(L"$(-1)^{s}\, C^{x}$"), zorder=101)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		p, = ax[1].plot(sites_y, corrY, marker="x", color=COLORS[4], lw=0, markersize=4, markeredgewidth=1.5, label=latexstring(L"$(-1)^{s}\, C^{y}$"), zorder=103)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		p, = ax[2].plot(sites_z, corrZ, marker=".", color=COLORS[7], lw=0, markersize=5, markeredgewidth=1.5, label=latexstring(L"$C^{z,\mathrm{c}}$"), zorder=103)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

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

		ax[1].text(28, 2.4e-2, latexstring(L"$\eta^{x}=%$(format_value(abs(a), da))$"), color=COLORS[1], fontsize=14, rotation=-37, ha="center", va="center")

		popty, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_y)), numpy.log(_corrY))
		perry = numpy.sqrt(numpy.diag(pcov))

		a = popty[1]
		da = perry[1]

		println("y : ", a, " ", da)

		ax[1].text(20, 0.5e-2, latexstring(L"$\eta^{y}=%$(format_value(abs(a), da))$"), color=COLORS[4], fontsize=14, rotation=-40, ha="center", va="center")

		py"""
		import numpy
		
		def g(x, a, b, c, d, e):

			return a * x ** b * numpy.cos(x * c * 2*numpy.pi + d) + e
		"""

		poptz, pcov = scipy.optimize.curve_fit(py"g", collect(_sites_z), _corrZ, [0.1, -1, 0.3, 1, 0])
		perrz = numpy.sqrt(numpy.diag(pcov))

		println("z : ", poptz, perrz)

		p, = ax[2].plot(_sites_z, py"g"(collect(_sites_z), poptz...), color="k", linewidth=1, linestyle="-", marker="none", zorder=102, label=latexstring(L"fit$^{\,}_{\mathrm{q}}$"))

		ax[2].text(58, -0.012, latexstring(L"$\eta^{z,\mathrm{c}}=%$(format_value(abs(poptz[2]), perrz[2]))$"), color=COLORS[7], fontsize=14, rotation=12, ha="center", va="center")

		ax[1].set_xlabel(L"$s$")
		ax[2].set_xlabel(L"$s$")
		
		ax[1].set_xscale("log")
		ax[1].set_yscale("log")
		
		ax[1].minorticks_on()
		ax[2].minorticks_on()
		
		ax[1].set_ylim(5e-4, 0.5)
		ax[2].set_xlim(0, 100)
		ax[2].set_ylim(-0.05, 0.02)
		
		ax[2].set_yticks([-0.05, 0])
		
		ax[1].legend(loc="lower left", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.5, handlelength=1).set_zorder(110)
		ax[2].legend(loc="lower right", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.5, handlelength=1).set_zorder(110)
		
		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "corr.pdf"))
		end

		plt.close()
	end
end

