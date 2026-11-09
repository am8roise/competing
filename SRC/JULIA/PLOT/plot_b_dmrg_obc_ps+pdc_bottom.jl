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
simul_name = "PS+PDC/BOTTOM"

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

fig, ax = plt.subplots(1, 2, figsize=(5, 2))

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
	"...",
	colors=[COLORS[7],COLORS[6],COLORS[5],COLORS[4]],
	N=256)

PyPlot.rc(
	"axes",
	prop_cycle=cycler.cycler(
		color=cmap(numpy.linspace(0, 1, length(Ns)))))

for folder in folders

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

			win = N - N // 8

			sites = range(1, size(entropies, 2))

			sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
			S = entropies[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

			py"""
			import numpy
			
			def g(x, a, b):

				return a * x + b
			"""

			x = 1 / 6 * numpy.log(2 * N / numpy.pi .* numpy.sin(numpy.pi .* sites ./ N))

			popt, pcov = scipy.optimize.curve_fit(py"g", x, S)
			perr = numpy.sqrt(numpy.diag(pcov))

			println(popt, perr)

			push!(cs, popt[1])
			push!(cs_err, perr[1])

			p, = ax[1].plot(x, S, marker="o", linewidth=0, markersize=5, label=latexstring(L"$2N=%$(N)$"), zorder=100-i)
			push!(ps, p)
		end
	end
end

axin = ax[1].inset_axes([0.15, 0.3, 0.25, 0.25])
axin.set_zorder(110)

p, = axin.plot(Ns, numpy.abs(cs), color=COLORS[3], marker="o", linewidth=0.5, markersize=3, zorder=112)

axin.fill_between(
	Ns,
	numpy.abs(cs) - cs_err,
	numpy.abs(cs) + cs_err,
	facecolor=COLORS[3],
	alpha=0.2,
	linewidth=0,
	zorder=111)

println(numpy.abs(cs) - cs_err, numpy.abs(cs) + cs_err)

ax[1].set_xlabel(latexstring(L"$\frac{1}{6} \ln \left[ \frac{4N}{\pi} \sin\frac{\pi l}{2N} \right]$"))
ax[1].set_ylabel(latexstring(L"$S(l)$"))

axin.set_xlabel(L"$2N$", fontsize=8)
axin.set_ylabel(L"$\mathsf{c}$", fontsize=8)

ax[1].set_xlim(0.4, 1.2)
ax[1].set_ylim(0, 1.4)

axin.set_xticks([0, 500, 1000])
axin.set_yticks([0.8, 1])

axin.tick_params(labelsize=8)

ax[1].minorticks_on()
axin.minorticks_on()

ax[1].text(
	0.97, 0.97,
	"(a)",
	transform=ax[1].transAxes,
	ha="right",
	va="top",
	fontsize=12)

ax[1].legend(
	handles=ps,
	loc="lower right",
	ncol=1,
	labelspacing=0.5,
	handlelength=0.5,
	handletextpad=0.5,
	handleheight=0,
	borderpad=0,
	borderaxespad=0.3,
	bbox_to_anchor=(1, 0)).set_zorder(1000)	

model_name = "B"
simul_name = "PS+PDC/BOTTOM"

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

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
	"...",
	colors=[COLORS[7],COLORS[6],COLORS[4]],
	N=256)

PyPlot.rc(
	"axes",
	prop_cycle=cycler.cycler(
		color=cmap(numpy.linspace(0, 1, length(Ns)))))

for folder in folders

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

			win = N - N // 8

			sites = range(1, size(entropies, 2))

			sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
			S = entropies[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

			py"""
			import numpy
			
			def g(x, a, b):

				return a * x + b
			"""

			x = 1 / 3 * numpy.log(N / numpy.pi .* numpy.sin(numpy.pi .* sites ./ N))

			popt, pcov = scipy.optimize.curve_fit(py"g", x, S)
			perr = numpy.sqrt(numpy.diag(pcov))

			println(popt, perr)

			push!(cs, popt[1])
			push!(cs_err, perr[1])

			p, = ax[2].plot(x, S, marker="o", linewidth=0, markersize=5, label=latexstring(L"$2N=%$(N)$"), zorder=100-i)
			push!(ps, p)
		end
	end
end

axin_ = ax[2].inset_axes([0.15, 0.3, 0.25, 0.25])
axin_.set_zorder(110)

p, = axin_.plot(Ns, numpy.abs(cs), color=COLORS[3], marker="o", linewidth=0.5, markersize=3, zorder=112)

axin_.fill_between(
	Ns,
	numpy.abs(cs) - cs_err,
	numpy.abs(cs) + cs_err,
	facecolor=COLORS[3],
	alpha=0.2,
	linewidth=0,
	zorder=111)

println(numpy.abs(cs) - cs_err, numpy.abs(cs) + cs_err)

ax[2].set_xlabel(latexstring(L"$\frac{1}{3} \ln \left[ \frac{2N}{\pi} \sin\frac{\pi l}{2N} \right]$"))

axin_.set_xlabel(L"$2N$", fontsize=8)
axin_.set_ylabel(L"$\mathsf{c}$", fontsize=8)

# ax[2].set_xlim(0.4, 1.2)
# ax[2].set_ylim(0, 1.4)

axin_.set_xticks([0, 50])
# axin_.set_yticks([0.8, 1])

axin_.set_yscale("log")

axin_.tick_params(labelsize=8)

ax[2].minorticks_on()
axin_.minorticks_on()

ax[2].text(
	0.97, 0.97,
	"(a)",
	transform=ax[1].transAxes,
	ha="right",
	va="top",
	fontsize=12)

ax[2].legend(
	handles=ps,
	loc="lower right",
	ncol=1,
	labelspacing=0.5,
	handlelength=0.5,
	handletextpad=0.5,
	handleheight=0,
	borderpad=0,
	borderaxespad=0.3,
	bbox_to_anchor=(1, 0)).set_zorder(1000)	

if save_plot
	model_name = "B"
	simul_name = "PS+PDC/BOTTOM"

	boundary_name = "BOTH"
	method_name = "DMRG"

	base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "central_charge.pdf"))
end

plt.close()