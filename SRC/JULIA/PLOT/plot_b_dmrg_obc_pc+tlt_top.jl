ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PC+TLT/TOP"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

r = 96

start_x = 10
end_x = 40

start_z = 10
end_z = 40

fig, ax = plt.subplots(1, figsize=(2, 3))

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

		corrZ = numpy.abs(corrZ)

		sites_x = range(1, r)
		sites_z = range(1, r)

		select_x = numpy.where(corrX .> 1e-15)[1] .+ 1
		select_z = numpy.where(corrZ .> 1e-15)[1] .+ 1

		corrX = corrX[select_x]
		corrZ = corrZ[select_z]

		sites_x = sites_x[select_x]
		sites_z = sites_z[select_z]
		
		p, = ax.plot(sites_x, corrX, marker="o", markerfacecolor=COLORS[4], lw=0, markersize=7, label=latexstring(L"$|C^{x}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=103)

		p, = ax.plot(sites_z, corrZ, marker="o", markerfacecolor=COLORS[1], lw=0, markersize=7, label=latexstring(L"$|C^{z}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=103)

		_sites_x = sites_x[start_x:end+1-end_x]
		_corrX = corrX[start_x:end+1-end_x]

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
		
		poptz, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_z)), numpy.log(_corrZ))
		perrz = numpy.sqrt(numpy.diag(pcov))

		a = poptz[1]
		da = perrz[1]

		println("z : ", a, " ", da)

		ax.set_xlabel(L"$s$")
		
		ax.set_xscale("log")
		ax.set_yscale("log")
		
		ax.set_ylim(0.0005, 3)

		for spine in ax.spines.values()
			spine.set_zorder(1000)
		end

		ax.text(
			0.03, 0.97,
			"(e)",
			transform=ax.transAxes,
			ha="left",
			va="top",
			fontsize=12)

		ax.legend(
			loc="upper right",
			ncol=1,
			bbox_to_anchor=(0.9, 1.085)).set_zorder(1000)
		
		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "corr.pdf"))
		end

		plt.close()
	end
end

