ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "LE+TEL/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "ED"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

n_E = 5

Ns = []
gaps_Ns = []
lam_Ns = []

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

		gaps[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = E[2:end] .- E[1:end-1]
	end

	push!(Ns, N)
	push!(gaps_Ns, gaps)
	push!(lam_Ns, lam)
end

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
				
ind = numpy.argsort(Ns) .+ 1

Ns = Ns[ind]
gaps_Ns = gaps_Ns[ind]
lam_Ns = lam_Ns[ind]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[7],COLORS[8],COLORS[1],COLORS[3],COLORS[4],COLORS[6]], N=length(Ns))
PyPlot.rc("axes", prop_cycle=cycler.cycler(marker=repeat([".", "|", "+"], outer=length(Ns))[1:length(Ns)], color=cmap(numpy.linspace(0, 1, length(Ns)))))

for i in 1:n_E-1

	for n in 1:length(Ns)
		p, = ax[i].plot(lam_Ns[n], Ns[n] .* gaps_Ns[n][:, 1, i], alpha=1, linestyle=":", markeredgewidth=1.4, markersize=6, linewidth=0.6, zorder=100, label=latexstring(L"$2N=%$(Ns[n])$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	end

	ax[i].minorticks_on()
	
	ax[i].set_xlabel(L"$\lambda$")

	if i==3
		ax[i].set_title(L"$2N(E^{\,}_{%$(i)} - E^{\,}_{%$(i-1)})$", loc="right", pad=10, fontsize=18)
	else
		ax[i].set_title(L"$2N(E^{\,}_{%$(i)} - E^{\,}_{%$(i-1)})$", pad=10, fontsize=18)
	end

	ax[i].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins="auto", integer=true))

	ax[i].axvline(pi/4, color="k", linestyle="--", linewidth=1)
end

ax[end].legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, ncol=1, labelspacing=0.8).set_zorder(200)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "theta=$(round(theta[1]; digits=3)).pdf"))
end

plt.close()
