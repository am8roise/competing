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
simul_name = "PA+PDC/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

start_x = 10
end_x = 100

start_22 = 20
end_22 = 300

start_44 = 20
end_44 = 300

nₖ = 5001
use_window_structure_factor = true
minₖ = 0.1 * pi

function get_window(n, use_window_structure_factor)

	if use_window_structure_factor
		return collect(numpy.hanning(n) .* 2 ./ n)
	else
		return ones(n)
	end

end

function structure_factor(sites, corr, nₖ, use_window_structure_factor, minₖ)

	sites = collect(Float64, sites)
	corr = collect(Float64, corr)

	window = get_window(length(corr), use_window_structure_factor)
	corr_window = corr .* window

	k = collect(range(0.0, stop = pi, length = nₖ))
	Cₖ = zeros(ComplexF64, length(k))

	for (i, kᵢ) in enumerate(k)
		Cₖ[i] = sum(corr_window .* exp.(-1im .* kᵢ .* sites))
	end

	_Cₖ = abs.(Cₖ)

	ind = findall(k .> minₖ)

	if length(ind) > 0
		maxₖ = ind[argmax(_Cₖ[ind])]
	else
		maxₖ = argmax(_Cₖ)
	end

	Q = k[maxₖ]

	return k, _Cₖ, Q

end

py"""
import numpy

def f(x, a, b):
	return a * x + b

def g(x, a, b, c, d, e, f):
	return a / x ** b * numpy.cos(x * 2 * numpy.pi * c + d) + e / x ** f

"""

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	if N != 1024
		continue
	end

	r = Int(N / 2)

	for i in 1:length(data_filenames)

		_lam = vals[i]["lam"]
		_theta = vals[i]["theta"]

		if _lam != 0.706858347057704
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

		_corr_24 =
			data["corrsCOS24"][1, 1, :] -
			data["corrsCOS24"][1, 2, :] -
			data["corrsCOS24"][1, 3, :] +
			data["corrsCOS24"][1, 4, :]

		_corr_42 =
			data["corrsCOS42"][1, 1, :] -
			data["corrsCOS42"][1, 2, :] -
			data["corrsCOS42"][1, 3, :] +
			data["corrsCOS42"][1, 4, :]

		_corr_44 =
			data["corrsCOS44"][1, 1, :] -
			data["corrsCOS44"][1, 2, :] -
			data["corrsCOS44"][1, 3, :] +
			data["corrsCOS44"][1, 4, :]

		corr_22ₗ =
			data["evs_2"][1, 1, j₀] -
			data["evs_2"][1, 2, j₀]

		corr_22ᵣ =
			data["evs_2"][1, 1, jᵣ] -
			data["evs_2"][1, 2, jᵣ]

		corr_44ₗ =
			data["evs_4"][1, 1, j₀] -
			data["evs_4"][1, 2, j₀]

		corr_44ᵣ =
			data["evs_4"][1, 1, jᵣ] -
			data["evs_4"][1, 2, jᵣ]

		corr_22 = _corr_22 .- corr_22ₗ .* corr_22ᵣ
		corr_44 = _corr_44 .- corr_44ₗ .* corr_44ᵣ
		
		fig, ax = plt.subplots(figsize=(2, 3))

		sites_x = range(1, r)

		_sites_x = sites_x[start_x:end_x]
		_corr_x = corr_x[_sites_x]

		k, Cₖ, Q = structure_factor(_sites_x, _corr_x, nₖ, use_window_structure_factor, minₖ)
	
		popt_x, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_x)), numpy.log(abs.(_corr_x)))
		perr_x = numpy.sqrt(numpy.diag(pcov))

		ax.plot(sites_x, abs.(corr_x), marker="o", color=COLORS[1], lw=0, markersize=5, label=L"\big|C^{x}_{s^{\,}_{0},s^{\,}_{0}+s}\big|", zorder=100)

		ax.plot(_sites_x, numpy.exp(py"f"(numpy.log(_sites_x), popt_x...)), color=COLORS[5], linewidth=2, linestyle="-", label=L"\propto s^{%$(format_value(popt_x[1], perr_x[1]))}", zorder=101)

		ax.set_xlabel(L"$s$")

		ax.set_xscale("log")
		ax.set_yscale("log")
		
		ax.text(
			0.03, 0.03,
			"(a)",
			transform=ax.transAxes,
			ha="left",
			va="bottom",
			fontsize=12)

		ax.legend(
			bbox_to_anchor=(1.05, 1.05),
			loc="upper right"
		).set_zorder(110)

		if save_plot
			plot_folder = joinpath(GRAPHS_FOLDER, base_folder, folder)

			py"create_folder_if_missing"(plot_folder)
			fig.savefig(joinpath(plot_folder, "x_lam=$(vals[i]["lam"])_theta=$(vals[i]["theta"]).pdf"))
		end

		plt.close()

		fig, ax = plt.subplots(1, 2, figsize=(4, 3))
		plt.subplots_adjust(wspace=0.25)

		sites_22 = range(1, r)

		_sites_22 = sites_22[start_22:end_22]
		_corr_22 = corr_22[_sites_22]

		k, Cₖ, Q = structure_factor(_sites_22, _corr_22, nₖ, use_window_structure_factor, minₖ)
	
		popt_22, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_22)), numpy.log(abs.(_corr_22)))
		perr_22 = numpy.sqrt(numpy.diag(pcov))

		_popt_22, pcov = scipy.optimize.curve_fit(py"g", collect(_sites_22), _corr_22, [1, 1, 0.3, 1, 1, 1])
		_perr_22 = numpy.sqrt(numpy.diag(pcov))

		println("----- XX-YY ------")
		println("R^2 uniform: \t", py"r2"(numpy.log(abs.(_corr_22)), py"f"(numpy.log(_sites_22), popt_22...)))
		println("R^2 oscillating: \t", py"r2"(_corr_22, py"g"(collect(_sites_22), _popt_22...)))
		println("params : \t", _popt_22)
		println("errors : \t", _perr_22)

		ax[1].plot(sites_22, abs.(corr_22), marker="o", color=COLORS[1], lw=0, markersize=5, label=L"\big|C^{2,2,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s}\big|", zorder=100)

		ax[1].plot(_sites_22, numpy.exp(py"f"(numpy.log(_sites_22), popt_22...)), color=COLORS[5], linewidth=2, linestyle="-", label=L"\propto s^{%$(format_value(popt_22[1], perr_22[1]))}", zorder=101)

		ax[2].plot(k, Cₖ, color=COLORS[1], linewidth=1, label=L"$\big|\widetilde{C}^{2,2,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s,q}\big|$", zorder=100)

		ax[2].plot(0, Cₖ[1], marker="o", markersize=5, color=COLORS[4], linewidth=0, label=L"q=0", zorder=101)
		ax[2].plot(Q, Cₖ[k .== Q], marker="o", markersize=8, color=COLORS[3], linewidth=0, label=L"q=%$(Q / pi)\times\pi", zorder=101)

		ax[1].set_xlabel(L"$s$")
		ax[2].set_xlabel(L"$q$")

		ax[1].set_xscale("log")
		ax[1].set_yscale("log")

		ax[1].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([10.0 ^ k for k in -16:0]))
		ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

		ax[2].set_xticks([0, pi/2, pi])
		ax[2].set_xticklabels([L"0", L"\frac{\pi}{2}", L"\pi"])
		ax[2].set_yticks([0, 4, 8] .* 1e-6)

		ax[1].set_ylim(1e-10, 2)
		ax[2].set_ylim(-1e-6, 9e-6)

		ax[1].text(
			0.97, 0.97,
			"(b)",
			transform=ax[1].transAxes,
			ha="right",
			va="top",
			fontsize=12)

		ax[2].text(
			0.97, 0.97,
			"(c)",
			transform=ax[2].transAxes,
			ha="right",
			va="top",
			fontsize=12)

		ax[1].legend(
			bbox_to_anchor=(-0.3, 0.08),
			loc="lower left"
		).set_zorder(110)

		ax[2].legend(
			bbox_to_anchor=(1.1, 0.4),
			loc="center right"
		).set_zorder(110)

		if save_plot
			plot_folder = joinpath(GRAPHS_FOLDER, base_folder, folder)

			py"create_folder_if_missing"(plot_folder)
			fig.savefig(joinpath(plot_folder, "22_lam=$(vals[i]["lam"])_theta=$(vals[i]["theta"]).pdf"))
		end

		plt.close()

		fig, ax = plt.subplots(1, 2, figsize=(4, 3))
		plt.subplots_adjust(wspace=0.25)

		sites_44 = range(1, r)

		_sites_44 = sites_44[start_44:end_44]
		_corr_44 = corr_44[_sites_44]

		k, Cₖ, Q = structure_factor(_sites_44, _corr_44, nₖ, use_window_structure_factor, minₖ)
	
		popt_44, pcov = scipy.optimize.curve_fit(py"f", collect(numpy.log(_sites_44)), numpy.log(abs.(_corr_44)))
		perr_44 = numpy.sqrt(numpy.diag(pcov))	
		
		_popt_44, pcov = scipy.optimize.curve_fit(py"g", collect(_sites_44), _corr_44, [1, 1, 0.3, 1, 1, 1])
		_perr_44 = numpy.sqrt(numpy.diag(pcov))

		println("----- Z(XX-YY)Z ------")
		println("R^2 uniform: \t", py"r2"(numpy.log(abs.(_corr_44)), py"f"(numpy.log(_sites_44), popt_44...)))
		println("R^2 oscillating: \t", py"r2"(_corr_44, py"g"(collect(_sites_44), _popt_44...)))
		println("params : \t", _popt_44)
		println("errors : \t", _perr_44)

		ax[1].plot(sites_44, abs.(corr_44), marker="o", color=COLORS[1], lw=0, markersize=5, label=L"\big|C^{4,4,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s}\big|", zorder=100)

		ax[1].plot(_sites_44, numpy.exp(py"f"(numpy.log(_sites_44), popt_44...)), color=COLORS[5], linewidth=2, linestyle="-", label=L"\propto s^{%$(format_value(popt_44[1], perr_44[1]))}", zorder=101)

		ax[2].plot(k, Cₖ, color=COLORS[1], linewidth=1, label=L"$\big|\widetilde{C}^{4,4,\mathrm{c}}_{s^{\,}_{0},s^{\,}_{0}+s,q}\big|$", zorder=100)

		ax[2].plot(0, Cₖ[1], marker="o", markersize=5, color=COLORS[4], linewidth=0, label=L"q=0", zorder=101)
		ax[2].plot(Q, Cₖ[k .== Q], marker="o", markersize=8, color=COLORS[3], linewidth=0, label=L"q=%$(Q / pi)\times\pi", zorder=101)

		ax[1].set_xlabel(L"$s$")
		ax[2].set_xlabel(L"$q$")

		ax[1].set_xscale("log")
		ax[1].set_yscale("log")

		ax[1].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([10.0 ^ k for k in -16:0]))
		ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

		ax[2].set_xticks([0, pi/2, pi])
		ax[2].set_xticklabels([L"0", L"\frac{\pi}{2}", L"\pi"])
		ax[2].set_yticks([0, 4, 8] .* 1e-6)

		ax[1].set_ylim(1e-10, 2)
		ax[2].set_ylim(-1e-6, 9e-6)

		ax[1].text(
			0.97, 0.97,
			"(d)",
			transform=ax[1].transAxes,
			ha="right",
			va="top",
			fontsize=12)

		ax[2].text(
			0.97, 0.97,
			"(e)",
			transform=ax[2].transAxes,
			ha="right",
			va="top",
			fontsize=12)

		ax[1].legend(
			bbox_to_anchor=(-0.3, 0.08),
			loc="lower left"
		).set_zorder(110)

		ax[2].legend(
			bbox_to_anchor=(1.1, 0.4),
			loc="center right"
		).set_zorder(110)

		if save_plot
			plot_folder = joinpath(GRAPHS_FOLDER, base_folder, folder)

			py"create_folder_if_missing"(plot_folder)
			fig.savefig(joinpath(plot_folder, "44_lam=$(vals[i]["lam"])_theta=$(vals[i]["theta"]).pdf"))
		end

		plt.close()
	end
end