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

fig, ax = plt.subplots(1, 2, figsize=(5.5, 3))
plt.subplots_adjust(wspace=0.4)

colors = [COLORS[4],COLORS[3],COLORS[7],COLORS[1]]

axin = ax[1].inset_axes([0.44, 0.22, 0.52, 0.55])
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

			p, = axin.plot(x, S, marker="o", markerfacecolor=colors[a], linewidth=0, markersize=5, label=latexstring(L"$2N=%$(N)$"), zorder=100-i)
			push!(ps, p)
		end
	end
end

p, = ax[1].plot(Ns, numpy.abs(cs), color=COLORS[5], marker="o", linewidth=0.5, markersize=7, zorder=112)

ax[1].fill_between(
	Ns,
	numpy.abs(cs) - cs_err,
	numpy.abs(cs) + cs_err,
	facecolor=COLORS[5],
	alpha=0.2,
	linewidth=0,
	zorder=111)

axin.set_xlabel(latexstring(L"$\frac{1}{6} \ln \left[ \frac{4N}{\pi} \sin\frac{\pi l}{2N} \right]$"), fontsize=10)
axin.set_ylabel(latexstring(L"$S(l)$"), fontsize=10)

ax[1].set_xlabel(L"$2N$")
ax[1].set_ylabel(L"$\mathsf{c}$")

ax[1].set_ylim(0.5, 1)

ax[1].set_xticks([128, 512, 1024])
ax[1].set_yticks([0.6, 0.8, 1])

ax[1].xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([256]))
ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

axin.set_xticks([0.5, 1])
axin.set_yticks([0.7, 1, 1.3])

axin.tick_params(labelsize=10)

ax[1].text(
	0.03, 0.97,
	"(a)",
	transform=ax[1].transAxes,
	ha="left",
	va="top",
	fontsize=12)

ax[1].legend(
	handles=ps,
	loc="lower left",
	ncol=1,
	handletextpad=0.3,
	borderpad=0.3,
	labelspacing=0.7,
	bbox_to_anchor=(-0.27, 0.05)).set_zorder(1000)	

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

colors = [COLORS[4],COLORS[3],COLORS[1]]

axin_ = ax[2].inset_axes([0.24, 0.22, 0.5, 0.25])
axin_.set_zorder(110)

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

			p, = axin_.plot(x, S, marker="o", markerfacecolor=colors[a], linewidth=0, markersize=5, label=latexstring(L"$2N=%$(N)$"), zorder=100-i)
			push!(ps, p)
		end
	end
end

p, = ax[2].plot(Ns, numpy.abs(cs .- 1), color=COLORS[5], marker="o", linewidth=0.5, markersize=7, zorder=112)

axin_.set_xlabel(latexstring(L"$\frac{1}{3} \ln \left[ \frac{2N}{\pi} \sin\frac{\pi l}{2N} \right]$"), fontsize=10)
axin_.set_ylabel(latexstring(L"$S(l)$"), fontsize=10)

ax[2].set_xlabel(L"$2N$")
ax[2].set_ylabel(L"$\mathsf{c-1}$")

ax[2].set_yscale("log")

ax[2].set_xlim([5, 140])
ax[2].set_ylim([0.00005, 0.15])

ax[2].set_xticks([32, 64, 128])

axin_.set_yticks([1, 2])

axin_.tick_params(labelsize=10)

ax[2].minorticks_off()
axin_.minorticks_on()

ax[2].text(
	0.03, 0.97,
	"(b)",
	transform=ax[2].transAxes,
	ha="left",
	va="top",
	fontsize=12)

ax[2].legend(
	handles=ps,
	loc="upper right",
	ncol=1,
	handletextpad=0.3,
	borderpad=0.3,
	labelspacing=0.7,
	bbox_to_anchor=(1.05, 0.95)).set_zorder(1000)	

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