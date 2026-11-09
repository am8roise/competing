ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PC+PBL/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

r = 128

start_x = 1
end_x = 5

start_y = 10
end_y = 1

start_z = 10
end_z = 1

fig, ax = plt.subplots(figsize=(2, 2))

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

		corrX = corrX - data["mags"][1, 1, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)] * data["mags"][1, 1, Int(N/2)-Int(r/2)]
		corrX = corrX .* [(-1)^j for j in range(1, length(corrX))]

		corrY = numpy.abs(corrY)
		corrZ = numpy.abs(corrZ)

		sites_x = range(1, r)
		sites_y = range(1, r)
		sites_z = range(1, r)

		select_x = numpy.where(corrX .> 1e-15)[1] .+ 1
		select_y = numpy.where(corrY .> 1e-15)[1] .+ 1
		select_z = numpy.where(corrZ .> 1e-15)[1] .+ 1

		corrX = corrX[select_x]
		corrY = corrY[select_y]
		corrZ = corrZ[select_z]

		sites_x = sites_x[select_x]
		sites_y = sites_y[select_y]
		sites_z = sites_z[select_z]
		
		p, = ax.plot(sites_x, corrX, marker="o", markerfacecolor=COLORS[2], linewidth=0, markersize=4, label=latexstring(L"$|C^{x,\mathrm{c}}_{j^{\,}_{0},j^{\,}_{0}+s}|$"), zorder=101)

		p, = ax.plot(sites_y, corrY, marker="D", markerfacecolor=COLORS[5], linewidth=0, markersize=4, label=latexstring(L"$|C^{y}_{j^{\,}_{0},j^{\,}_{0}+s}|$"), zorder=103)
		
		p, = ax.plot(sites_z, corrZ, marker="s", markerfacecolor=COLORS[8], linewidth=0, markersize=4, label=latexstring(L"$|C^{z}_{j^{\,}_{0},j^{\,}_{0}+s}|$"), zorder=104)

		_sites_x = sites_x[start_x:end+1-end_x]
		_corrX = corrX[start_x:end+1-end_x]

		_sites_y = sites_y[start_y:end+1-end_y]
		_corrY = corrY[start_y:end+1-end_y]

		_sites_z = sites_z[start_z:end+1-end_z]
		_corrZ = corrZ[start_z:end+1-end_z]

		py"""
		import numpy 

		def f(l, a, b):
			return a*l + b  
		"""

		poptx, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_x), numpy.log(_corrX))
		perrx = numpy.sqrt(numpy.diag(pcov))

		a = 1 / poptx[1]
		da = a^2 * perrx[1]

		println("x : ", a, " ", da)

		# plt.text(23, 1e-10, latexstring(L"$\xi^{x,\mathrm{c}}=%$(format_value(abs(a), da))$"), color=COLORS[1+1], fontsize=10, rotation=-51, ha="center", va="center")

		popty, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_y), numpy.log(_corrY))
		perry = numpy.sqrt(numpy.diag(pcov))

		a = 1 / popty[1]
		da = a^2 * perry[1]

		println("y : ", a, " ", da)

		# plt.text(19, 5e-12, latexstring(L"$\xi^{y}=%$(format_value(abs(a), da))$"), color=COLORS[4+1], fontsize=10, rotation=-50, ha="center", va="center")

		poptz, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_z), numpy.log(_corrZ))
		perrz = numpy.sqrt(numpy.diag(pcov))

		a = 1 / poptz[1]
		da = a^2 * perrz[1]

		println("z : ", a, " ", da)

		# plt.text(48, 2e-11, latexstring(L"$\xi^{z}=%$(format_value(abs(a), da))$"), color=COLORS[7+1], fontsize=10, rotation=-45, ha="center", va="center")

		ax.set_xlabel(L"$s$")

		ax.set_yscale("log")
		ax.set_yticks([1e-2, 1e-6, 1e-10, 1e-14])

		ax.set_xlim(-5, 65)

		seaborn.despine(offset=5)

		ax.legend(loc="upper right", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.2, handlelength=1, borderaxespad=0.2).set_zorder(0)

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "corr"))
		end

		plt.close()
	end
end

