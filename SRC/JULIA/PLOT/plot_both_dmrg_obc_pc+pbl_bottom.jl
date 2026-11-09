ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "A"
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

fig, ax = plt.subplots(1, 2, figsize=(4, 2))
plt.subplots_adjust(wspace=0.15)

handles = []

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
		println("x", corrX)
		println("y", corrY)
		println("z", corrZ)
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
		
		p, = ax[1].plot(sites_x, corrX, marker="o", markerfacecolor=COLORS[4], lw=0, markersize=7, label=latexstring(L"$|C^{x,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=103)

		p, = ax[1].plot(sites_y, corrY, marker="o", markerfacecolor=COLORS[3], lw=0, markersize=7, label=latexstring(L"$|C^{y}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=102)
		
		p, = ax[1].plot(sites_z, corrZ, marker="o", markerfacecolor=COLORS[1], lw=0, markersize=7, label=latexstring(L"$|C^{z}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=101)

		_sites_x = sites_x[start_x:end+1-end_x]
		_corrX = corrX[start_x:end+1-end_x]

		_sites_y = sites_y[start_y:end+1-end_y]
		_corrY = corrY[start_y:end+1-end_y]

		_sites_z = sites_z[start_z:end+1-end_z]
		_corrZ = corrZ[start_z:end+1-end_z]

		py"""
		import numpy 

		def f(l, a, b):
			return a * l + b  
		"""

		poptx, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_x), numpy.log(_corrX))
		perrx = numpy.sqrt(numpy.diag(pcov))

		a = 1 / poptx[1]
		da = a^2 * perrx[1]

		println("x : ", a, " ", da)

		popty, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_y), numpy.log(_corrY))
		perry = numpy.sqrt(numpy.diag(pcov))

		a = 1 / popty[1]
		da = a^2 * perry[1]

		println("y : ", a, " ", da)

		poptz, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_z), numpy.log(_corrZ))
		perrz = numpy.sqrt(numpy.diag(pcov))

		a = 1 / poptz[1]
		da = a^2 * perrz[1]

		println("z : ", a, " ", da)

		ax[1].set_xlabel(L"$s$")
		ax[1].set_xlim(-5, 65)

		ax[1].set_yscale("log")
		ax[1].set_ylim(1e-16, 1e0)

		ax[1].set_xticks([0, 25, 50])
		ax[1].set_yticks([1e-14, 1e-10, 1e-6, 1e-2])

		ax[1].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([10.0 ^ k for k in -16:0]))
		ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

		ax[1].text(
			0.97, 0.97,
			"(a)",
			transform=ax[1].transAxes,
			ha="right",
			va="top",
			fontsize=12)
	end
end

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
		
		p, = ax[2].plot(sites_x, corrX, marker="o", markerfacecolor=COLORS[4], lw=0, markersize=7, label=latexstring(L"$|C^{x,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=103)
		push!(handles, p)

		p, = ax[2].plot(sites_y, corrY, marker="o", markerfacecolor=COLORS[3], lw=0, markersize=7, label=latexstring(L"$|C^{y}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=102)
		push!(handles, p)
		
		p, = ax[2].plot(sites_z, corrZ, marker="o", markerfacecolor=COLORS[1], lw=0, markersize=7, label=latexstring(L"$|C^{z}_{s^{\,}_{0},s^{\,}_{0}+s}|$"), zorder=101)
		push!(handles, p)

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

		popty, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_y), numpy.log(_corrY))
		perry = numpy.sqrt(numpy.diag(pcov))

		a = 1 / popty[1]
		da = a^2 * perry[1]

		println("y : ", a, " ", da)

		poptz, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_z), numpy.log(_corrZ))
		perrz = numpy.sqrt(numpy.diag(pcov))

		a = 1 / poptz[1]
		da = a^2 * perrz[1]

		println("z : ", a, " ", da)

		ax[2].set_xlabel(L"$s$")
		ax[2].set_xlim(-5, 65)

		ax[2].set_yscale("log")
		ax[2].set_ylim(1e-16, 1e0)

		ax[2].text(
			0.97, 0.97,
			"(b)",
			transform=ax[2].transAxes,
			ha="right",
			va="top",
			fontsize=12)

		ax[2].set_xticks([0, 25, 50])
		ax[2].set_yticks([])
	end
end

ax[2].legend(
	handles=handles,
	loc="center left",
	ncol=1,
	bbox_to_anchor=(0.9, 0.5)).set_zorder(1000)

plt.draw()

if save_plot
	model_name = "BOTH"
	simul_name = "PC+PBL/BOTTOM"

	PBCs = false

	boundary_name = PBCs ? "PBC" : "OBC"
	method_name = "DMRG"

	base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

	folder = "N=256_chi=128"

	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlations.pdf"))
end

plt.close()