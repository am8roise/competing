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
simul_name = "SM+DBL/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

Ns = []
binder_Ns = []

lam = []
theta = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam_ = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta_ = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))
	
	push!(lam, lam_...)
	push!(theta, theta_...)
end

global lam = numpy.sort(numpy.unique(lam))
global theta = numpy.sort(numpy.unique(theta))

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	binder = numpy.ones((length(lam), length(theta), 2, 3, 2)) .* 1

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

len = length(Ns)

cross = numpy.ones((length(lam), 2)) .* 1

for l in range(1, length(lam))

	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[3],COLORS[4],COLORS[6],COLORS[7]], N=256)
	PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=repeat(["-", "--", ":"], outer=len)[1:len], marker=repeat(["x", "."], outer=len)[1:len], markersize=repeat([1], outer=len)[1:len], color=cmap(numpy.linspace(0, 1, len))))

	fig, ax = plt.subplots(1, 2, figsize=(7, 3))
	plt.subplots_adjust(wspace=0.5)

	for n in 1:length(Ns)

		p, = ax[1].plot(theta, binder_Ns[n, l, :, 1, 1, 2], linewidth=0.8, markeredgewidth=1.5, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		p, = ax[2].plot(theta, binder_Ns[n, l, :, 1, 3, 1], linewidth=0.8, markeredgewidth=1.5, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	end

	ax[1].minorticks_on()
	ax[2].minorticks_on()

	ax[1].legend(loc="lower center", ncol=3, labelspacing=0.4, columnspacing=2, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1.25, 1.02)).set_zorder(110)

	ax[1].set_xlabel(latexstring(L"$\theta$"))
	ax[2].set_xlabel(latexstring(L"$\theta$"))
	ax[1].set_ylabel(latexstring(L"$U\left[m^{x}_{\mathrm{sta}}\right]$"))
	ax[2].set_ylabel(latexstring(L"$U\left[m^{z}_{\mathrm{uni}}\right]$"))

	ax[1].axvline(pi/4, color="k", linestyle="--", linewidth=0.8)
	ax[2].axvline(pi/4, color="k", linestyle="--", linewidth=0.8)

	plt.draw()

	if save_plot
		py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
		fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "binder_lam=$(lam[l]).pdf"))
	end

	plt.close()

	ind = numpy.where(binder_Ns[1, l, :, 1, 3, 1] .!= 1)[1] .+ 1

	cross[l, 1] = theta[ind][py"find_nearest"(binder_Ns[1, l, ind, 1, 3, 1], binder_Ns[2, l, ind, 1, 3, 1]) + 1]
	cross[l, 2] = pi/600

	println(lam[l], theta[ind], cross[l, 1])
end

fig, ax = plt.subplots(figsize=(2, 2))

lams = numpy.concatenate((lam, pi/2 .- lam[end:-1:1]))
thetas = numpy.concatenate((cross[:, 1], cross[end:-1:1, 1]))
dts = numpy.concatenate((cross[:, 2], cross[end:-1:1, 2]))

plt.fill_betweenx(lams, thetas - dts, thetas + dts, color=COLORS[3], lw=0.1, alpha=1, label=L"$\partial\bigl(\mathrm{gFM}^{{\,}}_{{z}}\bigr)$")

lams = numpy.concatenate((lam, pi/2 .- lam[end:-1:1]))
thetas = numpy.concatenate((pi/2 .- cross[:, 1], pi/2 .- cross[end:-1:1, 1]))
dts = numpy.concatenate((cross[:, 2], cross[end:-1:1, 2]))

plt.fill_betweenx(lams, thetas - dts, thetas + dts, color=COLORS[3], lw=0.1, alpha=1)

ax.plot(pi/4, 0.299, markersize=8, marker="o", color=COLORS[5], linewidth=0, linestyle="none", zorder=200, label=L"$\Bigl(\frac{\pi}{4},\lambda^{{\,}}_{{\mathrm{{gFM}}^{{\,}}_{{z}}}}\Bigr)$")
ax.plot(pi/4, pi/2-0.299, markersize=8, marker="o", color=COLORS[6], linewidth=0, linestyle="none", zorder=200, label=L"$\Bigl(\frac{\pi}{4},\lambda^{{\mathrm{{SPT}}}}_{{\mathrm{{gFM}}^{{\,}}_{{z}}}}\Bigr)$")
ax.plot(0.272, pi/4, markersize=8, marker="o", color=COLORS[1], linewidth=0, linestyle="none", zorder=200, label=L"$\Bigl(\theta^{{\,}}_{{\mathrm{{gFM}}^{{\mathrm{{I}}}}_{{z}}}},\frac{\pi}{4}\Bigr)$")
ax.plot(pi/2-0.272, pi/4, markersize=8, marker="o", color=COLORS[2], linewidth=0, linestyle="none", zorder=200, label=L"$\Bigl(\theta^{{\prime}}_{{\mathrm{{gFM}}^{{\mathrm{{I}}}}_{{z}}}},\frac{\pi}{4}\Bigr)$")

xticks = [0, pi/4, pi/2]
xlabels = [L"0", L"\frac{\pi}{4}", L"\frac{\pi}{2}"]

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

yticks = [0, pi/4, pi/2]
ylabels = [L"0", L"\frac{\pi}{4}", L"\frac{\pi}{2}"]

ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)

ax.tick_params(axis="x", labelrotation=0)
ax.tick_params(axis="y", labelrotation=90)

ax.set_aspect("equal")

ax.set_xlabel(L"\theta")
ax.set_ylabel(L"\lambda")

ax.text(
	0.98, 0.03,
	"(a)",
	transform=ax.transAxes,
	ha="right",
	va="bottom",
	fontsize=12)

ax.legend(
	loc="lower center",
	ncol=1,
	handlelength=1.5,
	handletextpad=0.8,
	labelspacing=0.5,
	bbox_to_anchor=(0.5, 0.9)).set_zorder(1000)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "boundary_square.pdf"))
	ax.cla()
end

plt.close()