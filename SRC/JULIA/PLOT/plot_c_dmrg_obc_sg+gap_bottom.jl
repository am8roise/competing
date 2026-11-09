ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "C"
simul_name = "SG+GAP/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

plot_grid_evs = true

r = 32

save_plot = true

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[7],COLORS[9],COLORS[1],COLORS[6],COLORS[4]], N=256)

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	evs_2 = numpy.zeros((length(lam), length(theta), 3))
	evs_4 = numpy.zeros((length(lam), length(theta), 3))
	evs_6 = numpy.zeros((length(lam), length(theta), 3))
	evs_8 = numpy.zeros((length(lam), length(theta), 3))
	evs_10 = numpy.zeros((length(lam), length(theta), 3))
	evs_2_stag = numpy.zeros((length(lam), length(theta), 3))
	evs_4_stag = numpy.zeros((length(lam), length(theta), 3))
	evs_6_stag = numpy.zeros((length(lam), length(theta), 3))
	evs_8_stag = numpy.zeros((length(lam), length(theta), 3))
	evs_10_stag = numpy.zeros((length(lam), length(theta), 3))

	for i=1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))
	
		k = length(data["energies"])

		println(data_filenames[i])

		if plot_grid_evs
			
			evs_2[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(data["evs_2"][1, 1, Int(N/2-r/2):Int(N/2+r/2)])
			evs_2[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean(data["evs_2"][1, 2, Int(N/2-r/2):Int(N/2+r/2)])
			evs_2[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_2"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_4[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(data["evs_4"][1, 1, Int(N/2-r/2):Int(N/2+r/2)])
			evs_4[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean(data["evs_4"][1, 2, Int(N/2-r/2):Int(N/2+r/2)])
			evs_4[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_4"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_6[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_6"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_8[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_8"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

			evs_10[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_10"][1, 3, Int(N/2-r/2):Int(N/2+r/2)])

		end
	end

	if plot_grid_evs

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_2[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{2,x}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "x2.pdf"))
			fig.clf()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_2[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{2,y}}$"), fontsize=18, pad=10)
		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "y2.pdf"))
			fig.clf()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_2[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{2,z}}$"), fontsize=18, pad=10)
		
		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "z2.pdf"))
			fig.clf()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_4[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{4,x}}$"), fontsize=18, pad=10)
		
		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "x4.pdf"))
			fig.clf()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_4[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{4,y}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "y4.pdf"))
			fig.clf()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_4[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{4,z}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "z4.pdf"))
			fig.clf()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_6[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{6,z}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "z6.pdf"))
			fig.clf()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_8[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{8,z}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "z8.pdf"))
			fig.clf()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_10[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{10,z}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(theta), 0.5*length(theta), 1*length(theta)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "z10.pdf"))
			fig.clf()
		end

		plt.close()
	end
end


