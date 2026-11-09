ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")
include("../utils/math.jl")

model_name = "A"
simul_name = "LS+TMB/RIGHT"

PBCs = true

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

Ns = []
folders = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])

	push!(Ns, N)
	push!(folders, folder)
end

ind = numpy.argsort(Ns) .+ 1

Ns = Ns[ind]
folders = folders[ind]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[3],COLORS[4]], N=length(Ns))
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=repeat(["-", "--", ":"], outer=length(Ns))[1:length(Ns)], marker=repeat([".", 2, 3], outer=length(Ns))[1:length(Ns)], markersize=repeat([6], outer=length(Ns))[1:length(Ns)], color=cmap(numpy.linspace(0, 1, length(Ns)))))

fig, ax = plt.subplots(figsize=(8, 3))

for folder in folders

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	cs = numpy.zeros((length(lam), length(theta), 2))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))
		
		# win = Int(N/2+(N/2)%2+2)
		# win = Int(3*N/4)
		win = N-20

		println(data_filenames[i])

		Ss = data["entropies"]

		sites = range(1, size(Ss, 2))

		sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
		S = Ss[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

		py"""
		def f(l, c, const):
			import numpy

			N = int(l[len(l)//2]*2)

			return c/3 * numpy.log(N/numpy.pi * numpy.sin(numpy.pi*l/N)) + const
		"""

		popt, pcov = scipy.optimize.curve_fit(py"f", collect(sites), S)
		perr = numpy.sqrt(numpy.diag(pcov))

		cs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= popt[1]
		cs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= perr[1]
	end

	p, = ax.plot(lam, cs[:, 1, 1], linewidth=0.5, zorder=105, label=L"$2N=%$(N)$")
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

end

ax.minorticks_on()

ax.axvline(0.2849, color="k", linewidth=1, linestyle="--")

plt.xlabel(latexstring(L"$\lambda$"))
plt.ylabel(latexstring(L"$\mathsf{c}$"))

ax.legend(loc="lower left", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.6, handlelength=1.8).set_zorder(110)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "theta=$(theta[1]).pdf"))
	ax.cla()
end

plt.close()
