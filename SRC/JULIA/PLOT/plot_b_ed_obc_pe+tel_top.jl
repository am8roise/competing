ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PE+TEL/TOP"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "ED"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

n_E = 5

Ns = []
gaps_Ns = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	gaps = numpy.zeros((length(lam), length(theta), n_E - 1))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		E = numpy.sort(data["energies"])[1:n_E]

		println(E)

		gaps[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = E[2:end] .- E[1:end-1]
	end

	push!(Ns, N)
	push!(gaps_Ns, gaps)
end

ind = numpy.argsort(Ns) .+ 1

Ns = numpy.array(Ns)[ind]
gaps_Ns = numpy.array(gaps_Ns)[ind, :, :, :]

for l in 1:length(lam)
	for t in 1:length(theta)

		fig, ax = plt.subplots(figsize=(4, 4))

		py"""
		def f(N, a, b):
			return a*1/N + b
		"""

		ind = numpy.where(Ns .% 4 .== 0)[1] .+ 1
		ind_fit = numpy.where((Ns .% 4 .== 0) .& (Ns .> 8))[1] .+ 1

		p, = ax.plot(1 ./ Ns[ind], gaps_Ns[ind, l, t, 1],  marker="s", color=COLORS[1], linewidth=0, markeredgewidth=2, markersize=8, zorder=100, label=latexstring(L"$E^{\,}_{1}-E^{\,}_{0},\quad 2M\equiv 0\ \mathrm{mod}\ 4$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_fit]), gaps_Ns[ind_fit, l, t, 1])
		perr = numpy.sqrt(numpy.diag(pcov))

		println(popt, perr)

		p, = ax.plot(1 ./ Ns[ind_fit], py"f"(collect(Ns[ind_fit]), popt...),  linestyle="-", color=COLORS[1], linewidth=2, zorder=99, marker="none", label=latexstring(L"$v^{(0\ \mathrm{mod}\ 4)}_{1,0} = %$(format_value(popt[1], perr[1]))$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		ind = numpy.where(Ns .% 4 .== 0)[1] .+ 1
		ind_fit = numpy.where((Ns .% 4 .== 0) .& (Ns .> 8))[1] .+ 1

		p, = ax.plot(1 ./ Ns[ind], gaps_Ns[ind, l, t, 2],  marker="v", color=COLORS[3], linewidth=0, markeredgewidth=2, markersize=8, zorder=100, label=latexstring(L"$E^{\,}_{2}-E^{\,}_{1},\quad 2M\equiv 0\ \mathrm{mod}\ 4$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		
		popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_fit]), gaps_Ns[ind_fit, l, t, 2])
		perr = numpy.sqrt(numpy.diag(pcov))

		println(popt, perr)

		p, = ax.plot(1 ./ Ns[ind_fit], py"f"(collect(Ns[ind_fit]), popt...),  linestyle="-", color=COLORS[3], linewidth=2, zorder=99, marker="none", label=latexstring(L"$v^{(0\ \mathrm{mod}\ 4)}_{2,1} = %$(format_value(popt[1], perr[1]))$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		ind = numpy.where(Ns .% 4 .== 2)[1] .+ 1
		ind_fit = numpy.where((Ns .% 4 .== 2) .& (Ns .> 8))[1] .+ 1

		p, = ax.plot(1 ./ Ns[ind], gaps_Ns[ind, l, t, 1],  marker="^", color=COLORS[4], linewidth=0, markeredgewidth=2, markersize=8, zorder=100, label=latexstring(L"$E^{\,}_{1}-E^{\,}_{0},\quad 2M\equiv 2\ \mathrm{mod}\ 4$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_fit]), gaps_Ns[ind_fit, l, t, 1])
		perr = numpy.sqrt(numpy.diag(pcov))

		println(popt, perr)

		p, = ax.plot(1 ./ Ns[ind_fit], py"f"(collect(Ns[ind_fit]), popt...),  linestyle="--", color=COLORS[4], linewidth=2, zorder=99, marker="none", label=latexstring(L"$v^{(2\ \mathrm{mod}\ 4)}_{1,0} = %$(format_value(popt[1], perr[1]))$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		ind = numpy.where(Ns .% 4 .== 2)[1] .+ 1
		ind_fit = numpy.where((Ns .% 4 .== 2) .& (Ns .> 8))[1] .+ 1

		p, = ax.plot(1 ./ Ns[ind], gaps_Ns[ind, l, t, 2],  marker="D", color=COLORS[6], linewidth=0, markeredgewidth=2, markersize=8, zorder=100, label=latexstring(L"$E^{\,}_{2}-E^{\,}_{1},\quad 2M\equiv 2\ \mathrm{mod}\ 4$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_fit]), gaps_Ns[ind_fit, l, t, 2])
		perr = numpy.sqrt(numpy.diag(pcov))

		println(popt, perr)

		p, = ax.plot(1 ./ Ns[ind_fit], py"f"(collect(Ns[ind_fit]), popt...),  linestyle="--", color=COLORS[6], linewidth=2, zorder=99, marker="none", label=latexstring(L"$v^{(2\ \mathrm{mod}\ 4)}_{2,1} = %$(format_value(popt[1], perr[1]))$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		ax.minorticks_on()

		plt.xlabel(latexstring(L"$(2M)^{-1}$"))

		ax.legend(loc="center left", ncol=1, labelspacing=0.8, columnspacing=2, handletextpad=0.8, handlelength=1.6, borderaxespad=0, bbox_to_anchor=(1.01, 0.5)).set_zorder(200)

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "lam=$(lam[l])_theta=$(theta[t]).pdf"))
			fig.clf()
		end

		plt.close()

	end
end