ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PC+TEL/BOTTOM"

PBCs = true

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

r = 48
start_x = 8
end_x = 21
start_z = 8
end_z = 20

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	for i in 1:length(data_filenames)

		fig, ax = plt.subplots(1, figsize=(8, 3))

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		corrX = data["corrs"][1, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrY = data["corrs"][1, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrZ = data["corrs"][1, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

		corrX = corrX .* [(-1)^l for l in 1:length(corrX)]

		sites_x = range(1, r)
		sites_z = range(1, r)
		
		ind = numpy.where(sites_z .% 4 .== 0)[1] .+ 1
		_sites_z = sites_z[ind]
		_corrZ = corrZ[ind]
		
		p, = ax.plot(sites_x, corrX, marker=".", color=COLORS[1], lw=0, markersize=5, markeredgewidth=1.5, label=latexstring(L"$(-1)^{s}\,C^{x}$"), zorder=101)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		
		p, = ax.plot(_sites_z, _corrZ, marker="x", color=COLORS[7], lw=0, markersize=6, markeredgewidth=1.5, label=latexstring(L"$C^{z}$"), zorder=102)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		_sites_x = sites_x[start_x:end+1-end_x]
		_corrX = corrX[_sites_x]

		_sites_z = sites_z[start_z:end+1-end_z]
		_corrZ = corrZ[_sites_z]

		ind = numpy.where(_sites_z .% 4 .== 0)[1] .+ 1
		_sites_z = _sites_z[ind]
		_corrZ = _corrZ[ind]

		py"""
		import numpy 

		def f(l, a, b):
			return a*l + b  
		"""

		poptx, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_x)), numpy.log(_corrX))
		perrx = numpy.sqrt(numpy.diag(pcov))

		println(poptx, perrx)

		ax.plot(_sites_x, numpy.exp(py"f"(collect(numpy.log(_sites_x)), poptx...)), ls="-", lw=1.5, color=COLORS[3], zorder=103, label=latexstring(L"$(-1)^{s}\,C^{x}\sim s^{-(%$(numpy.abs(round(poptx[1]; digits=3)))\pm %$(round(perrx[1]; digits=3)))}$"))

		poptz, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_z)), numpy.log(_corrZ))
		perrz = numpy.sqrt(numpy.diag(pcov))

		println(poptz, perrz)

		ax.plot(_sites_z, numpy.exp(py"f"(collect(numpy.log(_sites_z)), poptz...)), ls=":", lw=1.5, color=COLORS[7], zorder=104, label=latexstring(L"$C^{z}\sim s^{-(%$(numpy.abs(round(poptz[1]; digits=3)))\pm %$(round(perrz[1]; digits=3)))}$"))

		plt.xlabel(L"$s$")

		ax.set_xscale("log")
		ax.set_yscale("log")

		ax.legend(loc="upper right", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.6, handlelength=1.8).set_zorder(110)

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "corr.pdf"))
		end

		plt.close()
		
	end
end