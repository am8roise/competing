ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "LC+PBL/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

plot_each_point = false

save_plot = true

r = 200
start_x = 1
end_x = 20

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

		corrX = corrX - data["mags"][1, 1, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)] * data["mags"][1, 1, Int(N/2)-Int(r/2)]

		sites_x = range(1, r)
		
		p, = ax.plot(sites_x, corrX, marker=".", color=COLORS[1], lw=0, markersize=5, markeredgewidth=1.5, label=latexstring(L"$C^{x}$"), zorder=101)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		_sites_x = sites_x[start_x:end+1-end_x]
		_corrX = corrX[_sites_x]

		ind = numpy.where((_corrX .> 1E-12) .& (_corrX .< 1E-2))[1] .+ 1
		_sites_x = _sites_x[ind]
		_corrX = _corrX[ind]

		py"""
		import numpy 

		def f(l, a, b):
			return a*l + b  
		"""

		poptx, pcov = scipy.optimize.curve_fit(py"f", collect(_sites_x), numpy.log(numpy.abs(_corrX)))
		perrx = numpy.sqrt(numpy.diag(pcov))

		println(poptx, perrx)

		ax.plot(_sites_x, numpy.exp(py"f"(collect(_sites_x), poptx...)), ls="-", lw=1.5, color=COLORS[3], zorder=103, label=latexstring(L"$C^{x}\sim s^{-(%$(numpy.abs(round(poptx[1]; digits=3)))\pm %$(round(perrx[1]; digits=3)))}$"))

		xi[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 1 / poptx[1]

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

		if (vals[i]["lam"] >= 0.78) & (vals[i]["lam"] <= 0.79)

			sites_x = range(1, r)
			
			p, = ax.plot(sites_x, corrX, marker=".", color=COLORS[1], lw=0, markersize=5, markeredgewidth=1.5, label=latexstring(L"$C^{x}$"), zorder=101)
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			_sites_x = sites_x[start_x:end+1-end_x]
			_corrX = corrX[_sites_x]

			ind = numpy.where((_corrX .> 1E-2) .& (_corrX .< 1E-0))[1] .+ 1
			_sites_x = _sites_x[ind]
			_corrX = _corrX[ind]

			py"""
			import numpy 

			def f(l, a, b):
				return a*l + b  
			"""

			poptx, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_x)), numpy.log(numpy.abs(_corrX)))
			perrx = numpy.sqrt(numpy.diag(pcov))

			println(poptx, perrx)

			ax.plot(_sites_x, numpy.exp(py"f"(collect(numpy.log(_sites_x)), poptx...)), ls="-", lw=1.5, color=COLORS[3], zorder=103, label=latexstring(L"$C^{x}\sim s^{-(%$(numpy.abs(round(poptx[1]; digits=3)))\pm %$(round(perrx[1]; digits=3)))}$"))

			eta[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= poptx[1]

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
	
	p, = ax.plot(lam, -xi[:, 1, 1], marker=".", color=COLORS[1], lw=1, markersize=5, markeredgewidth=1.5, zorder=101)
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

	ax.axvline(pi/4, color="k", linestyle="--", linewidth=1)

	plt.xlabel(L"$\lambda$")
	plt.ylabel(L"$\xi^{x,\mathrm{c}}$")
	
	ax.minorticks_on()

	axin = ax.inset_axes([0.15, 0.4, 0.2, 0.45])

	p, = axin.plot(lam, -xi[:, 1, 1], marker=".", color=COLORS[1], lw=1, markersize=5, markeredgewidth=1.5, zorder=101)
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

	axin.axvline(pi/4, color="k", linestyle="--", linewidth=1)

	axin.set_xlim(0.78, 0.79)
	axin.set_ylim(0, 50)

	axin.set_ylabel(L"$\xi^{x,\mathrm{c}}$")

	axin.minorticks_on()

	plt.draw()

	axin = ax.inset_axes([0.5, 0.4, 0.2, 0.45])

	p, = axin.plot(lam, -eta[:, 1, 1], marker=".", color=COLORS[1], lw=1, markersize=5, markeredgewidth=1.5, zorder=101)
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

	axin.axvline(pi/4, color="k", linestyle="--", linewidth=1)

	axin.set_xlim(0.78, 0.79)
	axin.set_ylim(0.9, 1.3)

	axin.set_ylabel(L"$\eta^{x,\mathrm{c}}$")

	axin.minorticks_on()

	plt.draw()

	if save_plot
		py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
		fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "correlation_length.pdf"))
		ax.cla()
	end

	plt.close()

end