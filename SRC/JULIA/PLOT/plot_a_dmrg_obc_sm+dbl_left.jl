ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "A"
simul_name = "SM+DBL/LEFT"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

plot_lines = true
plot_line_clean = true
plot_boundary = true

save_plot = true

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[3]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=["-", "--"], marker=[".", "."], markersize=[6, 6], color=cmap(numpy.linspace(0, 1, 2*length(2)))))

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	mags = numpy.ones((length(lam), length(theta), 6)) * 0.03
	windows = numpy.zeros((length(lam), 2))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		mX = data["mags"][1, 1, :]
		mY = data["mags"][1, 2, :]
		mZ = data["mags"][1, 3, :]

		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l in 1:length(mX)])
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l in 1:length(mY)])
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l in 1:length(mZ)])
	end

	for l in 1:length(lam)

		windows_x = numpy.where(numpy.abs(mags[l, :, 2]) .> 0.1)[1] .+ 1
		windows_z = numpy.where(numpy.abs(mags[l, :, 5]) .> 0.1)[1] .+ 1

		theta_x = numpy.max(theta[windows_x])
		theta_z = numpy.min(theta[windows_z])

		middle = (theta_z + theta_x) / 2
		size = (theta_z - theta_x) / 2

		windows[l, 1] = middle
		windows[l, 2] = size

		if plot_lines | (plot_line_clean & (l == 11))

			fig, ax = plt.subplots(figsize=(8, 3))

			if plot_lines

				ax.plot(theta, numpy.abs(mags[l, :, 2]), zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{x}|$"))
				ax.plot(theta, numpy.abs(mags[l, :, 5]), zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{z}|$"))

			elseif plot_line_clean

				ind = numpy.where((mags[l, :, 2] .!= 0.03) .& (mags[l, :, 5] .!= 0.03))[1] .+ 1
				theta_ = theta[ind]
				
				ax.plot(theta_, numpy.abs(mags[l, ind, 2]), zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{x}|$"))
				ax.plot(theta_, numpy.abs(mags[l, ind, 5]), zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{z}|$"))
			end

			ax.minorticks_on()

			ax.legend(loc="lower right", bbox_to_anchor=(0.49, 0.2), borderaxespad=0, ncol=3).set_zorder(101)

			plt.xlabel(latexstring(L"$\theta$"))

			plt.draw()

			if save_plot
				py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
				fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "lam=$(round(lam[l]; digits=3)).pdf"))
				ax.cla()
			end

			plt.close()
		end
	end

	if plot_boundary

		fig, ax = plt.subplots(figsize=(6, 5))

		ax.plot(pi/4, 0.299, markersize=8, marker="o", color=COLORS[1], linewidth=0, linestyle="none", zorder=200, label=L"$\bigl(\pi/4,\lambda^{{\,}}_{{\mathrm{{gFM}}^{{\,}}_{{z}}}}\bigr)$")
		ax.plot(pi/4, pi/2-0.299, markersize=8, marker="o", color=COLORS[3], linewidth=0, linestyle="none", zorder=200, label=L"$\bigl(\pi/4,\lambda^{{\mathrm{{SPT}}}}_{{\mathrm{{gFM}}^{{\,}}_{{z}}}}\bigr)$")

		lam = lam[[1:9; 11:12; 14:15; 17:end]]
		windows = windows[[1:9; 11:12; 14:15; 17:end], :]

		lams = numpy.concatenate((lam, pi/2 .- lam[end:-1:1]))
		thetas = numpy.concatenate((windows[:, 1], windows[end:-1:1, 1]))
		dts = numpy.concatenate((windows[:, 2], windows[end:-1:1, 2]))

		print(windows[:, 1], windows[:, 2])

		plt.fill_betweenx(lams, thetas - dts, thetas + dts, color=COLORS[1], alpha=0.2, label=L"$\partial\bigl(\mathrm{gFM}^{{\,}}_{{z}}\bigr)$")

		lams = numpy.concatenate((lam, pi/2 .- lam[end:-1:1]))
		thetas = numpy.concatenate((pi/2 .- windows[:, 1], pi/2 .- windows[end:-1:1, 1]))
		dts = numpy.concatenate((windows[:, 2], windows[end:-1:1, 2]))

		plt.fill_betweenx(lams, thetas - dts, thetas + dts, color=COLORS[1], alpha=0.2)

		ax.minorticks_on()

		ax.set_xlim(0, pi/2)
		ax.set_ylim(0, pi/2)

		plt.xticks([0, pi/4, pi/2], [L"$0$", L"$\pi/4$", L"$\pi/2$"])
		plt.yticks([0, pi/4, pi/2], [L"$0$", L"$\pi/4$", L"$\pi/2$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		ax.set_aspect("equal")

		ax.set_xlabel(L"$\theta$")
		ax.set_ylabel(L"$\lambda$")

		ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.01), borderaxespad=0.5, labelspacing=0.6, columnspacing=1, handletextpad=0.6, handlelength=1.6).set_zorder(200)

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "boundary_square.pdf"))
			ax.cla()
		end

		plt.close()
	end
end
