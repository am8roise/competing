ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")
include("../utils/math.jl")

model_name = "B"
simul_name = "PS+DTL/TOP"

PBCs = true

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

plot_entropies_combined = true

save_plot = true

ps = []
Ns = []
folders = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])

	push!(Ns, N)
	push!(folders, folder)
end

sort = numpy.argsort(Ns).+1
Ns = Ns[sort]
folders = folders[sort]

cs = []
cs_err = []

fig, ax = plt.subplots(figsize=(8, 3))

linestyles = repeat(["-", "-"], outer=length(Ns))[1:length(Ns)]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[3],COLORS[4],COLORS[6],COLORS[7],COLORS[8]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(marker=repeat(["x", "."], outer=length(Ns))[1:length(Ns)], markersize=repeat([4, 6], outer=length(Ns))[1:length(Ns)], markeredgewidth=repeat([1.25], outer=length(Ns))[1:length(Ns)], color=cmap(numpy.linspace(0, 1, length(Ns)))))

for folder in folders

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])
	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	for i=1:length(data_filenames)
		
		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))
		
		k = length(data["energies"])

		println("··· ", data_filenames[i])

		if plot_entropies_combined
			
			Ss = data["entropies"]

			# win = Int(4*N/5-4*(N%5)/5-4)
			# win = Int(N/2+(N/2)%2+4)
			win = N-10

			sites = range(1, size(Ss, 2))

			sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
			S = Ss[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

			py"""
			def g(x, a, b):
				import numpy

				return a*x + b
			"""

			x = 1/3 * numpy.log(N/numpy.pi .* numpy.sin(numpy.pi .*sites ./N))

			popt, pcov = scipy.optimize.curve_fit(py"g", x, S)
			perr = numpy.sqrt(numpy.diag(pcov))

			println(popt, perr)

			push!(cs, popt[1])
			push!(cs_err, perr[1])

		end
	end
end

plots, caps, errs = ax.errorbar(Ns, numpy.abs(cs .- 1), cs_err, ecolor=COLORS[1], elinewidth=1.5, marker="x", markersize=5, markeredgewidth=1.5, linewidth=0.5, color=COLORS[7], zorder=110)
plots.set_marker(matplotlib.markers.MarkerStyle(plots.get_marker(), capstyle="round"))

for err in errs
	err.set_capstyle("round")
end

ax.set_xlabel(L"$2N$")
ax.set_ylabel(L"$\mathsf{c}^{\,}_{2N}-1$")

ax.set_yscale("log")

ax.set_yticks([0.01, 0.1])

ax.minorticks_on()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "chi=$(chi).pdf"))
	ax.cla()
end

plt.close()