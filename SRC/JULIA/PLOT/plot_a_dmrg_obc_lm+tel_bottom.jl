ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "A"
simul_name = "LM+TEL/BOTTOM"

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

len = length(Ns)

colors = [COLORS[6], COLORS[5], COLORS[4], COLORS[3], COLORS[7], COLORS[1]]
markersizes = [3, 4, 5, 6, 7, 8]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=colors, N=256)
PyPlot.rc("axes",
	prop_cycle=cycler.cycler(
		linestyle=repeat([(1.5, (3, 3)), (0, (0, 2)), "-"], outer=len)[1:len],
		markersize=markersizes,
		color=cmap(numpy.linspace(0, 1, len))))

fig, ax = plt.subplots(2, 1, figsize=(5, 5.5))
plt.subplots_adjust(hspace=0.05)

axin1 = ax[1].inset_axes([0.18, 0.28, 0.28, 0.5])
axin2 = ax[2].inset_axes([0.2, 0.53, 0.25, 0.44])

for n in 1:length(Ns)

	p, = ax[1].plot(lam, binder_Ns[n, :, 1, 1, 1, 2], marker="o", lw=0.5, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101+n)

	p, = ax[2].plot(lam, binder_Ns[n, :, 1, 2, 2, 2],  marker="o", lw=0.5, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101-n)

	println("For N=", Ns[n], ", U < 0 at", lam[numpy.where(binder_Ns[n, :, 1, 1, 1, 2] .< 0)[1] .+ 1])

	ind = numpy.argmin(binder_Ns[n, :, 1, 1, 1, 2]) + 1
	p, = axin1.plot(1 / (lam[ind] - pi/4),  binder_Ns[n, ind, 1, 1, 1, 2], marker="o", color=colors[n], markersize=markersizes[n], lw=0, ls="none")

	ind = numpy.argmin(binder_Ns[n, :, 1, 2, 2, 2]) + 1
	p, = axin2.plot(lam[ind] - pi/4,  binder_Ns[n, ind, 1, 2, 2, 2], marker="o", color=colors[n], markersize=markersizes[n], lw=0, ls="none")
end

for spine in axin1.spines.values()
	spine.set_zorder(999)
end
for spine in axin2.spines.values()
	spine.set_zorder(999)
end

axin1.set_xlabel(latexstring(L"$\bigl(\lambda^{\,}_{\mathrm{min}}-\frac{\pi}{4}\bigr)^{-1}$"), fontsize=10)
axin2.set_xlabel(latexstring(L"$\lambda^{\,}_{\mathrm{min}}-\frac{\pi}{4}$"), fontsize=10)
axin1.set_ylabel(latexstring(L"$\min\limits_{\lambda}\,U^{\,}_{\hat{m}^{x}_{\mathrm{sta}}}$"), fontsize=10, labelpad=0)
axin2.set_ylabel(latexstring(L"$\min\limits_{\lambda}\,U^{\,}_{\hat{m}^{\mathrm{SPT},y}_{\mathrm{sta}}}$"), fontsize=10)

axin2.set_xticks([0, 0.02])
axin2.set_xticklabels([L"0", L"0.02"])

axin1.set_yticks([-0.1, -0.2])
axin2.set_yticks([0, -0.1, -0.2])

axin1.tick_params(labelsize=10)
axin2.tick_params(labelsize=10)

ax[2].legend(
	loc="lower center",
	ncol=3,
	handlelength=1.5,
	columnspacing=1.2,
	bbox_to_anchor=(0.5, 2)).set_zorder(1000)

ax[2].set_xlabel(latexstring(L"$\lambda$"))

ax[1].set_ylabel(latexstring(L"$U^{\,}_{\hat{m}^{x}_{\mathrm{sta}}}$"), rotation=-90, labelpad=25)
ax[1].yaxis.set_label_position("right")

ax[2].set_ylabel(latexstring(L"$U^{\,}_{\hat{m}^{\mathrm{SPT},y}_{\mathrm{sta}}}$"), rotation=-90, labelpad=25)
ax[2].yaxis.set_label_position("right")

ax[1].axvline(pi/4, color="k", linestyle="-", linewidth=0.5)
ax[2].axvline(pi/4, color="k", linestyle="-", linewidth=0.5)

ax[1].set_xticks([])
ax[2].set_xticks([0.7, 0.8, 0.9])

ax[1].set_yticks([0, 2/3])
ax[1].set_yticklabels([L"0", L"\frac{2}{3}"])
ax[1].yaxis.tick_right()

ax[2].set_yticks([0, 2/3])
ax[2].set_yticklabels([L"0", L"\frac{2}{3}"])
ax[2].yaxis.tick_right()

ax[1].set_ylim(-0.25, 0.8)
ax[2].set_ylim(-0.25, 0.8)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "op.pdf"))
end

plt.close()