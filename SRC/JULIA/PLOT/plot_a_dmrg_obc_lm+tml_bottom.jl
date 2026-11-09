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
simul_name = "LM+TML/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

Ns = []
binder_Ns = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))
	
	binder = numpy.ones((length(lam), length(theta), 2, 3, 2))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		binder[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :, :, :] = data["binder"][1, :, :, :]
	end

	push!(Ns, N)
	push!(binder_Ns, binder)
end

ind = numpy.argsort(Ns) .+ 1

Ns = numpy.array(Ns)[ind]
binder_Ns = numpy.array(binder_Ns)[ind, :, :, :, :, :]

len = length(Ns)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[3],COLORS[4],COLORS[6],COLORS[7],COLORS[8]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=repeat(["-", "--", ":"], outer=len)[1:len], marker=repeat(["o"], outer=len)[1:len], markersize=repeat([3], outer=len)[1:len], color=cmap(numpy.linspace(0, 1, len)), markerfacecolor=cmap(numpy.linspace(0, 1, len))))

fig, ax = plt.subplots(figsize=(4, 3))

for n in 1:length(Ns)

	p, = ax.plot(theta, binder_Ns[n, 1, :, 1, 3, 1], linewidth=0.6, markeredgewidth=0.2, markeredgecolor="0.", label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101)
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

end

ax.minorticks_on()

ax.legend(loc="lower center", ncol=3, labelspacing=0.4, columnspacing=2, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1.25, 1.02)).set_zorder(110)

ax.set_xlabel(latexstring(L"$\theta$"))
ax.set_ylabel(latexstring(L"$U\left[m^{z}_{\mathrm{uni}}\right]$"))

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "binder.pdf"))
end

plt.close()

