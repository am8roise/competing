ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "LC+TMT/TOP"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

plot_each_point = true

save_plot = true

r = 400

start_z_xi = 10
end_z_xi = 10

start_z_eta = 5
end_z_eta = 5

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	xi = numpy.zeros((length(lam), length(theta), 1))
	eta = numpy.zeros((length(lam), length(theta), 1))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		fig, ax = plt.subplots(1, figsize=(8, 3))

		corrX = data["corrs"][1, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrY = data["corrs"][1, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrZ = data["corrs"][1, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

		corrZ = corrZ - data["mags"][1, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)] * data["mags"][1, 3, Int(N/2)-Int(r/2)]
		corrZ = numpy.abs(corrZ)

		if (vals[i]["lam"] < 2.74)

			sites_z = range(1, length(corrZ))
			
			p, = ax.plot(sites_z, corrZ, marker=".", color=COLORS[1], lw=0, markersize=5, markeredgewidth=1.5, label=latexstring(L"$C^{z,\mathrm{c}}$"), zorder=101)
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			ind = numpy.where((corrZ .< 1E-3) .& (corrZ .> 1E-12))[1] .+ 1

			if (vals[i]["lam"] >= 2.68) .& (vals[i]["lam"] < 2.72)
				ind = numpy.where((corrZ .< 1E-4) .& (corrZ .> 1E-7))[1] .+ 1

			elseif (vals[i]["lam"] >= 2.72) .& (vals[i]["lam"] < 2.74)
				ind = numpy.where((corrZ .< 1E-3) .& (corrZ .> 1E-5))[1] .+ 1

			elseif (vals[i]["lam"] >= 2.74) .& (vals[i]["lam"] < 2.75)
				ind = numpy.where((corrZ .< 1E-1) .& (corrZ .> 1E-3))[1] .+ 1
			end


			_sites_z = sites_z[ind]
			_corrZ = corrZ[ind]

			_sites_z = _sites_z[start_z_xi:end+1-end_z_xi]
			_corrZ = _corrZ[start_z_xi:end+1-end_z_xi]

			py"""
			import numpy 

			def f(x, a, b):
				return a * x + b  
			"""

			poptz, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_z), numpy.log(numpy.abs(_corrZ)))
			perrz = numpy.sqrt(numpy.diag(pcov))

			a = 1 / poptz[1]
			da = a^2 * perrz[1]

			println(a, " ", da)

			ax.plot(_sites_z, numpy.exp(py"f"(collect(_sites_z), poptz...)), ls="-", lw=1.5, color=COLORS[3], zorder=103, label=latexstring(L"$C^{z,\mathrm{c}}\sim e^{-s/(%$(format_value(abs(a), da)))}$"))

			xi[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= a

			plt.xlabel(L"$s$")

			ax.set_yscale("log")

			ax.minorticks_on()

			ax.legend(loc="upper right", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.6, handlelength=1.8).set_zorder(110)

			plt.draw()

			if save_plot & plot_each_point
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lam=$(vals[i]["lam"])_exp.png"))
				ax.cla()
			end

			plt.close()

		elseif (vals[i]["lam"] > 2.75)

			sites_z = range(1, length(corrZ))
			
			p, = ax.plot(sites_z, corrZ, marker=".", color=COLORS[1], lw=0, markersize=5, markeredgewidth=1.5, label=latexstring(L"$C^{x}$"), zorder=101)
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			ind = numpy.where((corrZ .< 1) .& (corrZ .> 1E-3))[1] .+ 1

			if (vals[i]["lam"] >= 2.75) .& (vals[i]["lam"] < 2.77)
				ind = numpy.where((corrZ .< 1.1E-1) .& (corrZ .> 5E-2))[1] .+ 1

			elseif (vals[i]["lam"] >= 2.77) .& (vals[i]["lam"] < 2.8)
				ind = numpy.where((corrZ .< 1E-1) .& (corrZ .> 3E-2))[1] .+ 1

			elseif (vals[i]["lam"] >= 2.8) .& (vals[i]["lam"] < 2.84)
				ind = numpy.where((corrZ .< 1E-1) .& (corrZ .> 1E-2))[1] .+ 1

			elseif (vals[i]["lam"] >= 2.84) .& (vals[i]["lam"] < 2.9)
				ind = numpy.where((corrZ .< 1E-1) .& (corrZ .> 5E-3))[1] .+ 1

			elseif (vals[i]["lam"] >= 3.05)
				ind = numpy.where((corrZ .< 1E-1) .& (corrZ .> 1E-5))[1] .+ 1
			end

			_sites_z = sites_z[ind]
			_corrZ = corrZ[ind]

			print(length(_corrZ))

			_sites_z = _sites_z[start_z_eta:end+1-end_z_eta]
			_corrZ = _corrZ[start_z_eta:end+1-end_z_eta]

			py"""
			import numpy 

			def f(x, a, b):
				return a * x + b  
			"""

			poptz, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_z)), numpy.log(numpy.abs(_corrZ)))
			perrz = numpy.sqrt(numpy.diag(pcov))

			a = poptz[1]
			da = perrz[1]

			println(a, " ", da)

			ax.plot(_sites_z, numpy.exp(py"f"(collect(numpy.log(_sites_z)), poptz...)), ls="-", lw=1.5, color=COLORS[3], zorder=103, label=latexstring(L"$C^{z,\mathrm{c}}\sim s^{-%$(format_value(abs(a), da))}$"))

			eta[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= poptz[1]

			plt.xlabel(L"$s$")

			ax.set_xscale("log")
			ax.set_yscale("log")

			ax.minorticks_on()

			ax.legend(loc="upper right", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.6, handlelength=1.8).set_zorder(110)

			plt.draw()

			if save_plot & plot_each_point
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lam=$(vals[i]["lam"])_alg.png"))
				ax.cla()
			end

			plt.close()
		end
		
	end

	fig, ax = plt.subplots(1, figsize=(8, 3))

	ind = numpy.where(xi[:, 1, 1])[1] .+ 1

	p, = ax.plot(lam[ind], -xi[ind, 1, 1], marker=".", color=COLORS[1], lw=1, markersize=5, markeredgewidth=1.5, zorder=101, label=latexstring(L"$\xi^{z,\mathrm{c}}$"))
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

	ind = numpy.where(eta[:, 1, 1])[1] .+ 1

	p, = ax.plot(lam[ind], -eta[ind, 1, 1], marker="+", color=COLORS[4], lw=1, markersize=5, markeredgewidth=1.5, zorder=101, label=latexstring(L"$\eta^{z}}$"))
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

	ax.axvline(pi-0.2849, color="k", linestyle="--", linewidth=1)

	plt.xlabel(L"$\lambda$")
	
	ax.minorticks_on()

	plt.draw()

	if save_plot
		py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
		fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlation_length.pdf"))
		ax.cla()
	end

	plt.close()

end