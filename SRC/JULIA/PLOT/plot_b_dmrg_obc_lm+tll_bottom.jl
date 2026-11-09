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
simul_name = "LM+TLL/BOTTOM"

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

fig, ax = plt.subplots(1, 2, gridspec_kw=Dict("wspace" => 0.1), figsize=(5, 3))

for n in 1:length(Ns)

	len = length(numpy.where(Ns .% 4 .== 0)[1])

	colors = [COLORS[6], COLORS[5], COLORS[4], COLORS[3], COLORS[7], COLORS[1]]
	markersizes = [3, 4, 5, 6, 7, 8]

	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=colors, N=len)
	PyPlot.rc("axes",
		prop_cycle=cycler.cycler(
			linestyle=repeat([(1.5, (3, 3)), (0, (0, 2)), "-"], outer=len)[1:len],
			markersize=markersizes,
			color=cmap(numpy.linspace(0, 1, len))))

	if Ns[n] % 4 == 0

		p, = ax[1].plot(lam, binder_Ns[n, :, 1, 1, 1, 2], marker="o", lw=0.5, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101-n)

		p, = ax[2].plot(lam, binder_Ns[n, :, 1, 2, 1, 2],  marker="o", lw=0.5, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101-n)

		println("For N=", Ns[n], ", U < 0 at", lam[numpy.where(binder_Ns[n, :, 1, 1, 1, 2] .< 0)[1] .+ 1])

	end
end

ax[2].text(
	0.97, 0.97,
	"(a)",
	transform=ax[2].transAxes,
	ha="right",
	va="top",
	fontsize=12)

ax[2].legend(
	loc="lower center",
	ncol=3,
	handlelength=1.5,
	columnspacing=1.2,
	bbox_to_anchor=(-0.05, 0.9)).set_zorder(1000)

ax[1].set_xlabel(latexstring(L"$\lambda$"))
ax[2].set_xlabel(latexstring(L"$\lambda$"))

ax[1].set_ylabel(latexstring(L"$U^{\,}_{\hat{m}^{x}_{\mathrm{sta}}}$"))
ax[2].set_ylabel(latexstring(L"$U^{\,}_{\hat{m}^{\mathrm{SPT},x}_{\mathrm{sta}}}$"), rotation=-90, labelpad=20)
ax[2].yaxis.set_label_position("right")

ax[1].axvline(pi/4, color="k", linestyle="-", linewidth=0.5)
ax[2].axvline(pi/4, color="k", linestyle="-", linewidth=0.5)

ax[1].set_xticks([0.7, 0.8, 0.9])
ax[2].set_xticks([0.7, 0.8, 0.9])

ax[1].set_yticks([0, 0.5])
ax[1].set_yticklabels([L"0", L"\frac{1}{2}"])

ax[2].set_yticks([0, 0.5])
ax[2].set_yticklabels([L"0", L"\frac{1}{2}"])
ax[2].yaxis.tick_right()

ax[1].set_ylim(0, 0.6)
ax[2].set_ylim(0, 0.6)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "binder.pdf"))
end

plt.close()

fig, ax = plt.subplots(1, 2, gridspec_kw=Dict("wspace" => 0.02, "width_ratios" => [2, 1]), figsize=(5, 2))

handles = []
mean = 0
err = 0

ind_0 = numpy.where(Ns .% 4 .== 0)[1] .+ 1
ind_2 = numpy.where(Ns .% 4 .== 2)[1] .+ 1

binder_Ns_0 = binder_Ns[ind_0,:, 1, 1, 1, 2]
binder_Ns_2 = binder_Ns[ind_2,:, 1, 1, 1, 2]

cross_0 = []
for n in zip(1:length(Ns[ind_0])-1, 2:length(Ns[ind_0]))

	ind = numpy.where((lam .> 0.5) .& (lam .< 1))[1] .+ 1

	d = binder_Ns_0[n[2], ind] .- binder_Ns_0[n[1], ind]
	λ = lam[ind]

	for i in 1:length(λ)-1
		d1, d2 = d[i], d[i+1]

		if d1 * d2 < 0 || d2 == 0
			x = d1 / (d1 - d2)
			global cross_ = λ[i] + x * (λ[i+1] - λ[i])
			break
		end
	end

	push!(cross_0, cross_)
end

cross_2 = []
for n in zip(1:length(Ns[ind_2])-1, 2:length(Ns[ind_2]))

	ind = numpy.where((lam .> 0.5) .& (lam .< 1))[1] .+ 1

	d = binder_Ns_2[n[2], ind] .- binder_Ns_2[n[1], ind]
	λ = lam[ind]

	for i in 1:length(λ)-1
		d1, d2 = d[i], d[i+1]

		if d1 * d2 < 0 || d2 == 0
			x = d1 / (d1 - d2)
			global cross_ = λ[i] + x * (λ[i+1] - λ[i])
			break
		end
	end

	push!(cross_2, cross_)
end

py"""
def f(N, a, b, c):
	import numpy

	return a * 1 / (N * numpy.log(N)) ** b + c
"""

println(cross_0, cross_2)

p, = ax[1].plot(Ns[ind_0][1:end-2], cross_0[1:end-1], marker="o", linewidth=0, markerfacecolor=COLORS[3], linestyle="none", markersize=10, zorder=100, label=L"\lambda^{(0)}_{\hat{m}^{x}_{\mathrm{sta}},2N^{\,}_{\times}}")
push!(handles, p)

p, = ax[1].plot(Ns[ind_2][1:end-1], cross_2[1:end], marker="o", linewidth=0, markerfacecolor=COLORS[5], linestyle="none", markersize=10, zorder=101, label=L"\lambda^{(2)}_{\hat{m}^{x}_{\mathrm{sta}},2N^{\,}_{\times}}")
push!(handles, p)

popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_0][1:end-2]), cross_0[1:end-1])
perr = numpy.sqrt(numpy.diag(pcov))

println(popt, perr)
mean += popt[end]
err += perr[end]

x = numpy.linspace(numpy.min(Ns), numpy.max(Ns)+2, 100)
p, = ax[1].plot(x, py"f"(x, popt...), color=COLORS[3], linewidth=1, linestyle="-", marker="none", markersize=0)
lam_0 = py"f"(x, popt...)

x = numpy.logspace(numpy.log10(numpy.max(Ns)), numpy.log10(1000), 1000)
p, = ax[2].plot(x, py"f"(x, popt...), color=COLORS[3], linewidth=1, linestyle="-", marker="none", markersize=0)
lam_0_ = py"f"(x, popt...)

popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_2][1:end-1]), cross_2[1:end])
perr = numpy.sqrt(numpy.diag(pcov))

println(popt, perr)
mean += popt[end]
err += perr[end]

x = numpy.linspace(numpy.min(Ns), numpy.max(Ns)+2, 100)
p, = ax[1].plot(x, py"f"(x, popt...), color=COLORS[5], linewidth=1, linestyle="-", marker="none", markersize=0, zorder=98)
lam_2 = py"f"(x, popt...)

x = numpy.logspace(numpy.log10(numpy.max(Ns)), numpy.log10(1000), 1000)
p, = ax[2].plot(x, py"f"(x, popt...), color=COLORS[5], linewidth=1, linestyle="-", marker="none", markersize=0, zorder=98)
lam_2_ = py"f"(x, popt...)

ind_0 = numpy.where(Ns .% 4 .== 0)[1] .+ 1
ind_2 = numpy.where(Ns .% 4 .== 2)[1] .+ 1

binder_Ns_0 = binder_Ns[ind_0,:, 1, 2, 1, 2]
binder_Ns_2 = binder_Ns[ind_2,:, 1, 2, 1, 2]

cross_0 = []
for n in zip(1:length(Ns[ind_0])-1, 2:length(Ns[ind_0]))

	ind = numpy.where((lam .> 0.5) .& (lam .< 1))[1] .+ 1

	d = binder_Ns_0[n[2], ind] .- binder_Ns_0[n[1], ind]
	λ = lam[ind]

	for i in 1:length(λ)-1
		d1, d2 = d[i], d[i+1]

		if d1 * d2 < 0 || d2 == 0
			x = d1 / (d1 - d2)
			global cross_ = λ[i] + x * (λ[i+1] - λ[i])
			break
		end
	end

	push!(cross_0, cross_)
end

cross_2 = []
for n in zip(1:length(Ns[ind_2])-1, 2:length(Ns[ind_2]))

	ind = numpy.where((lam .> 0.5) .& (lam .< 1))[1] .+ 1

	d = binder_Ns_2[n[2], ind] .- binder_Ns_2[n[1], ind]
	λ = lam[ind]

	for i in 1:length(λ)-1
		d1, d2 = d[i], d[i+1]

		if d1 * d2 < 0 || d2 == 0
			x = d1 / (d1 - d2)
			global cross_ = λ[i] + x * (λ[i+1] - λ[i])
			break
		end
	end

	push!(cross_2, cross_)
end

println(cross_0, cross_2)

p, = ax[1].plot(Ns[ind_0][1:end-2], cross_0[1:end-1], marker="o", linewidth=0, markerfacecolor=COLORS[4], linestyle="none", markersize=10, zorder=100, label=L"\lambda^{(0)}_{\hat{m}^{\mathrm{SPT},x}_{\mathrm{sta}},2N^{\,}_{\times}}")
push!(handles, p)

p, = ax[1].plot(Ns[ind_2][1:end-1], cross_2[1:end], marker="o", linewidth=0, markerfacecolor=COLORS[6], linestyle="none", markersize=10, zorder=101, label=L"\lambda^{(2)}_{\hat{m}^{\mathrm{SPT},x}_{\mathrm{sta}},2N^{\,}_{\times}}")
push!(handles, p)

popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_0][1:end-2]), cross_0[1:end-1])
perr = numpy.sqrt(numpy.diag(pcov))

println(popt, perr)
mean += popt[end]
err += perr[end]

x = numpy.linspace(numpy.min(Ns), numpy.max(Ns)+2, 100)
p, = ax[1].plot(x, py"f"(x, popt...), color=COLORS[4], linewidth=1, linestyle="-", marker="none", markersize=0, zorder=98)
lam_0_SPT = py"f"(x, popt...)

x = numpy.logspace(numpy.log10(numpy.max(Ns)), numpy.log10(1000), 1000)
p, = ax[2].plot(x, py"f"(x, popt...), color=COLORS[4], linewidth=1, linestyle="-", marker="none", markersize=0, zorder=98)
lam_0_SPT_ = py"f"(x, popt...)

popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_2][1:end-1]), cross_2[1:end])
perr = numpy.sqrt(numpy.diag(pcov))

println(popt, perr)
mean += popt[end]
err += perr[end]

x = numpy.linspace(numpy.min(Ns), numpy.max(Ns)+2, 100)
p, = ax[1].plot(x, py"f"(x, popt...), color=COLORS[6], linewidth=1, linestyle="-", marker="none", markersize=0, zorder=98)
lam_2_SPT = py"f"(x, popt...)

x = numpy.logspace(numpy.log10(numpy.max(Ns)), numpy.log10(1000), 1000)
p, = ax[2].plot(x, py"f"(x, popt...), color=COLORS[6], linewidth=1, linestyle="-", marker="none", markersize=0, zorder=98)
lam_2_SPT_ = py"f"(x, popt...)

x = numpy.linspace(numpy.min(Ns), numpy.max(Ns)+2, 100)
x_ = numpy.logspace(numpy.log10(numpy.max(Ns)), numpy.log10(1000), 1000)

# ----- MEAN -----
m = numpy.mean(numpy.vstack((lam_0, lam_2, lam_0_SPT, lam_2_SPT)), axis=0)
p, = ax[1].plot(x, m, color=COLORS[1], linewidth=1, linestyle=(2, (4, 4)), marker="none", markersize=0, zorder=100, label=L"\lambda^{\,}_{\mathrm{c}}")
push!(handles, p)

m_ = numpy.mean(numpy.vstack((lam_0_, lam_2_, lam_0_SPT_, lam_2_SPT_)), axis=0)
p, = ax[2].plot(x_, m_, color=COLORS[1], linewidth=1, linestyle=(2, (4, 4)), marker="none", markersize=0, zorder=100, label=L"\lambda^{\,}_{\mathrm{c}}")

println(mean / 4, "±", err)
# ----------------

for spine in ax[1].spines.values()
	spine.set_zorder(999)
end
for spine in ax[2].spines.values()
	spine.set_zorder(999)
end

# ========== BROKEN X AXIS
ax[1].spines.right.set_visible(false)
ax[2].spines.left.set_visible(false)

d = 0.03
ax[1].plot([1-d/4, 1+d/4], [-d, d], ls="-", transform=ax[1].transAxes, lw=0.5, color="k", marker="none", clip_on=false, zorder=1000)
ax[1].plot([1-d/4, 1+d/4], [1-d, 1+d], ls="-", transform=ax[1].transAxes, lw=0.5, color="k", marker="none", clip_on=false, zorder=1000)
ax[2].plot([-d/2, d/2], [-d, d], ls="-", transform=ax[2].transAxes, lw=0.5, color="k", marker="none", clip_on=false, zorder=1000)
ax[2].plot([-d/2, d/2], [1-d, 1+d], ls="-", transform=ax[2].transAxes, lw=0.5, color="k", marker="none", clip_on=false, zorder=1000)
# ==========

ax[1].set_xlabel(L"2N^{\,}_{\times}", x=0.75)

ax[2].set_xscale("log")

ax[1].set_xlim(28, 51.99)
ax[2].set_xlim(52, numpy.max(x_))

ax[1].set_yticks([0.75, pi/4, 0.85])
ax[2].set_yticks([0.75, pi/4, 0.85])
ax[1].set_yticklabels([L"0.75", L"\frac{\pi}{4}", L"0.85"])
ax[2].set_yticklabels([L"0.75", L"\frac{\pi}{4}", L"0.85"])

ax[1].set_ylim(0.7, 0.9)
ax[2].set_ylim(0.7, 0.9)

ax[1].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([]))
ax[2].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([]))

ax[2].yaxis.tick_right()

ax[2].text(
	0.97, 0.97,
	"(b)",
	transform=ax[2].transAxes,
	ha="right",
	va="top",
	fontsize=12)

ax[2].legend(
	handles=handles,
	loc="lower center",
	ncol=3,
	handlelength=1,
	columnspacing=1.2,
	bbox_to_anchor=(-1.2, 0.9)).set_zorder(1000)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "cross.pdf"))
end

plt.close()