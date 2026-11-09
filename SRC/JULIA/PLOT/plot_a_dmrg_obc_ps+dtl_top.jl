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
simul_name = "PS+DTL/TOP"

PBCs = false

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

fig, ax = plt.subplots(figsize=(2, 3))

colors = [COLORS[4],COLORS[3],COLORS[1]]
zorders = [101, 102, 103]
sizes = [5, 3, 5]

axin = ax.inset_axes([0.34, 0.23, 0.5, 0.26])
axin.set_zorder(110)

for (a, folder) in enumerate(folders)

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])
	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	for i in 1:length(data_filenames)
		
		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))
		
		println("··· ", data_filenames[i])

		if plot_entropies_combined
			
			entropies = data["entropies"]

			if N == 128
				win = N - 10
			elseif N == 256 
				win = N - 10
			else
				win = 4 * N // 5 - 4 * (N % 5) // 5 - 4
			end

			sites = range(1, size(entropies, 2))

			sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
			S = entropies[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

			py"""
			import numpy
			
			def g(x, a, b):

				return a * x + b
			"""

			x = 1 / 6 * numpy.log(2*N / numpy.pi .* numpy.sin(numpy.pi .* sites ./ N))

			popt, pcov = scipy.optimize.curve_fit(py"g", x, S)
			perr = numpy.sqrt(numpy.diag(pcov))

			println(popt, perr)

			push!(cs, popt[1])
			push!(cs_err, perr[1])

			p, = axin.plot(x, S, marker="o", markerfacecolor=colors[a], linewidth=0, markersize=sizes[a], label=latexstring(L"$2N=%$(N)$"), zorder=zorders[a])
			push!(ps, p)
		end
	end
end

p, = ax.plot(Ns, numpy.abs(cs .- 1), color=COLORS[5], marker="o", linewidth=0.5, markersize=10, zorder=112)

axin.set_xlabel(latexstring(L"$\frac{1}{6} \ln \left[ \frac{4N}{\pi} \sin\frac{\pi l}{2N} \right]$"), fontsize=10)
axin.set_ylabel(latexstring(L"$S(l)$"), fontsize=10)

ax.minorticks_off()

ax.set_xlabel(L"2N")
ax.set_ylabel(L"\mathsf{c}^{\,}_{2N}-1")

ax.set_yscale("log")

ax.set_ylim([0.006, 0.13])

ax.set_xticks([128, 256, 512])

axin.tick_params(labelsize=10)

ax.text(
	0.03, 0.97,
	"(a)",
	transform=ax.transAxes,
	ha="left",
	va="top",
	fontsize=12)

ax.legend(
	handles=ps,
	loc="upper right",
	ncol=1,
	borderpad=0.3,
	labelspacing=0.7,
	bbox_to_anchor=(0.9, 1.085)).set_zorder(1000)		

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "central_charge.pdf"))
end

plt.close()