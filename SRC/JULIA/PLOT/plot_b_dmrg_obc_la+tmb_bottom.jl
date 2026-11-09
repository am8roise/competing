ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "LA+TMB/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-be\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

plot_x = false
plot_22 = false

start_x = 10
end_x = 50

start_22 = 20
end_22 = 60

py"""
def f(x, a, b):
	return a * x + b
"""

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	r = Int(N / 2)

	eta = numpy.zeros((length(lam), length(theta), 3))

	for i in 1:length(data_filenames)

		_lam = vals[i]["lam"]
		_theta = vals[i]["theta"]

		if _lam == 0.785398163397448
			continue
		end

		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		sign = model_name == "A" ? 1.0 : -1.0

		j₀ = Int(N / 2) - Int(r / 2)
		jᵣ = j₀+1 : j₀+r

		corr_x = abs.(data["corrs"][1, 1, j₀, jᵣ])

		_corr_22 =
			data["corrsCOS22"][1, 1, :] -
			data["corrsCOS22"][1, 2, :] -
			data["corrsCOS22"][1, 3, :] +
			data["corrsCOS22"][1, 4, :]

		corr_22ₗ =
			data["evs_2"][1, 1, j₀] -
			data["evs_2"][1, 2, j₀]

		corr_22ᵣ =
			data["evs_2"][1, 1, jᵣ] -
			data["evs_2"][1, 2, jᵣ]

		corr_22 = _corr_22 .- corr_22ₗ .* corr_22ᵣ

		fig, ax = plt.subplots(figsize=(2, 3))

		sites_x = range(1, r)

		_sites_x = sites_x[start_x:end_x]
		_corr_x = corr_x[_sites_x]
	
		popt_x, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_x)), numpy.log(abs.(_corr_x)))
		perr_x = numpy.sqrt(numpy.diag(pcov))

		eta[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= -popt_x[1]

		println(popt_x, perr_x)

		if plot_x

			ax.plot(sites_x, abs.(corr_x), marker="o", color=COLORS[1], lw=0, markersize=5, label=L"|C^{\,}_{j^{\,}_{0},j^{\,}_{0}+s}|", zorder=100)

			ax.plot(_sites_x, numpy.exp(py"f"(numpy.log(_sites_x), popt_x...)), color=COLORS[5], linewidth=2, linestyle="-", label=L"\propto s^{%$(format_value(popt_x[1], perr_x[1]))}", zorder=101)

			ax.set_xlabel(L"$s$")

			ax.set_xscale("log")
			ax.set_yscale("log")

			ax.legend(
				bbox_to_anchor=(1.05, 1.05),
				loc="upper right"
			).set_zorder(110)

			if save_plot
				plot_folder = joinpath(GRAPHS_FOLDER, base_folder, folder, "corr_x")

				py"create_folder_if_missing"(plot_folder)
				fig.savefig(joinpath(plot_folder, "lam=$(vals[i]["lam"])_theta=$(vals[i]["theta"]).png"))
			end
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(4, 3))

		sites_22 = range(1, r)

		_sites_22 = sites_22[start_22:end_22]
		_corr_22 = corr_22[_sites_22]
	
		popt_22, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_22)), numpy.log(abs.(_corr_22)))
		perr_22 = numpy.sqrt(numpy.diag(pcov))

		eta[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= -popt_22[1]

		println(popt_22, perr_22)

		if plot_22

			ax.plot(sites_22, abs.(corr_22), marker="o", color=COLORS[1], lw=0, markersize=5, label=L"|C^{\,}_{j^{\,}_{0},j^{\,}_{0}+s}|", zorder=100)

			ax.plot(_sites_22, numpy.exp(py"f"(numpy.log(_sites_22), popt_22...)), color=COLORS[5], linewidth=2, linestyle="-", label=L"\propto s^{%$(format_value(popt_22[1], perr_22[1]))}", zorder=101)

			ax.set_xlabel(L"$s$")

			ax.set_xscale("log")
			ax.set_yscale("log")

			ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([10.0 ^ k for k in -16:0]))
			ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

			ax.set_ylim(1e-10, 2)

			ax.legend(
				bbox_to_anchor=(0.1, 0.08),
				loc="lower left"
			).set_zorder(110)

			if save_plot
				plot_folder = joinpath(GRAPHS_FOLDER, base_folder, folder, "corr_22")

				py"create_folder_if_missing"(plot_folder)
				fig.savefig(joinpath(plot_folder, "lam=$(vals[i]["lam"])_theta=$(vals[i]["theta"]).png"))
			end
		end

		plt.close()

	end

	fig, ax = plt.subplots(figsize=(3, 3))
	
	ax.plot(
		lam[1:end-1],
		eta[1:end-1, 1, 1],
		marker="o",
		color=COLORS[1],
		markersize=5,
		linewidth=0,
		label=L"\eta^{x}",
		zorder=101
	)	

	ax.plot(
		lam[1:end-1],
		eta[1:end-1, 1, 2] ./ 4,
		marker="o",
		color=COLORS[3],
		markersize=8,
		linewidth=0,
		label=L"p^{\,}_{\mathrm{U}(1)}/4",
		zorder=100
	)

	ax.axvline(0.2993, color="k", linestyle="-", linewidth=0.5)

	ax.set_xticks([0, pi/8, pi/4])
	ax.set_xticklabels([L"0", L"\frac{\pi}{8}", L"\frac{\pi}{4}"])
	ax.set_yticks([0, 1/2, 1])
	ax.set_yticklabels([L"0", L"\frac{1}{2}", L"1"])

	ax.set_ylim([0, 1.1])

	ax.set_xlabel(L"$\lambda$")

	ax.legend(
		bbox_to_anchor=(0.1, 1.03),
		loc="upper left"
	).set_zorder(110)
	
	plt.draw()

	if save_plot
		plot_folder = joinpath(GRAPHS_FOLDER, base_folder, folder)

		py"create_folder_if_missing"(plot_folder)
		fig.savefig(joinpath(plot_folder, "exponents.pdf"))
	end

	plt.close()

end