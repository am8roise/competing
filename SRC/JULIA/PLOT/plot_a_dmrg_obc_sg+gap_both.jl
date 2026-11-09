ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "A"
simul_name = "SG+GAP/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

plot_grid_evs = true

r = 32

save_plot = true

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[5], COLORS[8], COLORS[1], COLORS[4], COLORS[7]], N=256)

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	global lam_bottom = lam

	global evs_2_bottom = numpy.zeros((length(lam), length(theta), 3))
	global evs_4_bottom = numpy.zeros((length(lam), length(theta), 3))
	global evs_6_bottom = numpy.zeros((length(lam), length(theta), 3))
	global evs_8_bottom = numpy.zeros((length(lam), length(theta), 3))
	global evs_10_bottom = numpy.zeros((length(lam), length(theta), 3))

	for i=1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))
	
		k = length(data["energies"])

		println(data_filenames[i])

		if plot_grid_evs
			
			evs_2_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(data["evs_2"][1, 1, Int(N/2-r/2):Int(N/2+r/2)])
			evs_2_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean(data["evs_2"][1, 2, Int(N/2-r/2):Int(N/2+r/2)])
			evs_2_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_2"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_4_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 4*numpy.mean(data["evs_4"][1, 1, Int(N/2-r/2):Int(N/2+r/2)])
			evs_4_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 4*numpy.mean(data["evs_4"][1, 2, Int(N/2-r/2):Int(N/2+r/2)])
			evs_4_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 4*numpy.mean(data["evs_4"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_6_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*numpy.mean(data["evs_6"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_8_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*4*numpy.mean(data["evs_8"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_10_bottom[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*16*numpy.mean(data["evs_10"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])
		end
	end
end

simul_name = "SG+GAP/TOP"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	global lam_top = lam

	global evs_2_top = numpy.zeros((length(lam), length(theta), 3))
	global evs_4_top = numpy.zeros((length(lam), length(theta), 3))
	global evs_6_top = numpy.zeros((length(lam), length(theta), 3))
	global evs_8_top = numpy.zeros((length(lam), length(theta), 3))
	global evs_10_top = numpy.zeros((length(lam), length(theta), 3))

	for i=1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))
	
		k = length(data["energies"])

		println(data_filenames[i])

		if plot_grid_evs
			
			evs_2_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(data["evs_2"][1, 1, Int(N/2-r/2):Int(N/2+r/2)])
			evs_2_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean(data["evs_2"][1, 2, Int(N/2-r/2):Int(N/2+r/2)])
			evs_2_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_2"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_4_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(data["evs_4"][1, 1, Int(N/2-r/2):Int(N/2+r/2)])
			evs_4_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean(data["evs_4"][1, 2, Int(N/2-r/2):Int(N/2+r/2)])
			evs_4_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_4"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_6_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_6"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_8_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_8"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_10_top[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_10"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])
		end
	end
end

folder = numpy.unique(py"folder_names"(joinpath(DATA_FOLDER, base_folder)))[1]

simul_name = "SG+GAP/BOTH"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

if plot_grid_evs

	evs_2 = cat(evs_2_bottom, evs_2_top[2:end, :, :]; dims=1)
	evs_4 = cat(evs_4_bottom, evs_4_top[2:end, :, :]; dims=1)
	evs_6 = cat(evs_6_bottom, evs_6_top[2:end, :, :]; dims=1)
	evs_8 = cat(evs_8_bottom, evs_8_top[2:end, :, :]; dims=1)
	evs_10 = cat(evs_10_bottom, evs_10_top[2:end, :, :]; dims=1)

	lam = cat(lam_bottom, lam_top[2:end]; dims=1)

	function formatter(L, labels)

		function _formatter(y, pos)
			eps = 1e-6
			for (frac, label) in labels
				if abs(y - frac * L) < eps
					return label
				end
			end
			return ""
		end

		return _formatter
	end

	fig, ax = plt.subplots(1, 6, figsize=(11, 4))

	plt.subplots_adjust(wspace=0.2)

	quantities = [evs_2[end:-1:1, :, 1],
				  evs_2[end:-1:1, :, 2],
				  evs_4[end:-1:1, :, 1],
				  evs_4[end:-1:1, :, 2],
				  evs_4[end:-1:1, :, 3],
				  evs_8[end:-1:1, :, 3]]

	panels = [
		"(a)",
		"(b)",
		"(c)",
		"(d)",
		"(e)",
		"(f)"]

	for i in 1:length(quantities)

		data = pandas.DataFrame(
			quantities[i],
			index=numpy.round(lam, 5)[end:-1:1],
			columns=numpy.round(theta, 4))
		global axs = seaborn.heatmap(
			data,
			ax=ax[i],
			square=true,
			cbar=false,
			rasterized=true,
			cmap=cmap)

		print("\n\n", panels[i], quantities[i])

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		cb = fig.colorbar(axs.collections[1],
			ax=axs,
			location="top",
			ticks=[-1, 0, 1],
			pad=0.02)
		cb.ax.tick_params(labelsize=12)

		ax[i].set_xlabel(latexstring(L"$\theta$"))

		ax[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(length(theta) / 2))
		ax[i].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(length(theta) / 10))

		labels = [
			(1.0,	L"\frac{\pi}{2}"),
			(0.5,	L"\frac{\pi}{4}"),
			(0.0,	L"0")]

		ax[i].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(formatter(length(theta), labels)))

		ax[i].set_yticks([])

		ax[i].tick_params(axis="x", labelrotation=0)
		ax[i].tick_params(axis="y", labelrotation=90)

		ax[i].text(
			0.97, 0.97,
			panels[i],
			transform=ax[i].transAxes,
			ha="right",
			va="top",
			fontsize=12)
		
		if i == 1
			ax[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(length(lam) / 4))
			ax[i].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(length(lam) / 20))

			labels = [
				(0.0,	L"\pi"),
				(0.25,	L"\frac{3\pi}{4}"),
				(0.5,	L"\frac{\pi}{2}"),
				(0.75,	L"\frac{\pi}{4}"),
				(1.0,	L"0")]

			ax[i].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(formatter(length(lam), labels)))

			ax[i].set_ylabel(latexstring(L"$\lambda$"))
		end
	end

	plt.draw()

	if save_plot
		py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
		fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "all.pdf"))
		fig.clf()
	end

	plt.close()
end