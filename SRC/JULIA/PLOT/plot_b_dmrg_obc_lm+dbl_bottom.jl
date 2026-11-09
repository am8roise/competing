ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "LM+DBL/BOTTOM"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

Ns = []
folders = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])

	push!(Ns, N)
	push!(folders, folder)
end

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[2], COLORS[3], COLORS[1], COLORS[4]], N=2*length(Ns))
PyPlot.rc("axes", prop_cycle=cycler.cycler(
	linestyle=["-",  "-", (0.2, (0, 2)), (0, (0, 2))],
	markersize=[10, 10, 5, 5],
	color=cmap(numpy.linspace(0, 1, 2*length(Ns)))))

fig, ax = plt.subplots(figsize=(4, 2))

sort = numpy.argsort(Ns) .+ 1
folders = folders[sort]

binder_Ns = []

for folder in folders

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))
	
	mags = numpy.ones((length(lam), length(theta), 6))
	binder = numpy.ones((length(lam), length(theta), 2, 3, 2))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		binder[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :, :, :] = data["binder"][1, :, :, :]

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

		p1, = ax.plot(theta, numpy.abs(mags[l, :, 2]), marker="o", lw=1, label=latexstring(L"$|m^{\mathrm{sta}}_{x}|,\ 2N=%$(N)$"), zorder=105)
		p1.set_marker(matplotlib.markers.MarkerStyle(p1.get_marker(), capstyle="round"))

		p2, = ax.plot(theta, numpy.abs(mags[l, :, 5]), marker="o", lw=1, label=latexstring(L"$|m^{\mathrm{uni}}_{z}|,\ 2N=%$(N)$"), zorder=105)
		p2.set_marker(matplotlib.markers.MarkerStyle(p2.get_marker(), capstyle="round"))
	end

	push!(binder_Ns, binder)
end

ax.set_xlim(-0.04, 1)

ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels([L"0", L"0.25", L"0.5", L"0.75", L"1"])

ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([L"0", L"0.5", L"1"])

ax.legend(
	loc="center left",
	ncol=1,
	handlelength=2,
	bbox_to_anchor=(0.85, 0.4)).set_zorder(1001)

plt.xlabel(latexstring(L"$\theta$"))

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "op.pdf"))
	ax.cla()
end

plt.close()

ind = numpy.argsort(Ns) .+ 1

Ns = numpy.array(Ns)[ind]
binder_Ns = numpy.array(binder_Ns)[ind, :, :, :, :, :]

len = length(Ns)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[3],COLORS[4],COLORS[6],COLORS[7],COLORS[8]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=repeat(["-", "--", ":"], outer=len)[1:len], marker=repeat(["x", "."], outer=len)[1:len], markersize=repeat([1], outer=len)[1:len], color=cmap(numpy.linspace(0, 1, len))))

fig, ax = plt.subplots(1, 2, figsize=(7, 3))
plt.subplots_adjust(wspace=0.5)

for n in 1:length(Ns)

	p, = ax[1].plot(theta, binder_Ns[n, 1, :, 1, 1, 2], linewidth=0.8, markeredgewidth=1.5, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101)
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

	# println("For N=", Ns[n], ", U < 0 at", lam[numpy.where(binder_Ns[n, :, 1, 1, 1, 2] .< 0)[1] .+ 1])

end

for n in 1:length(Ns)

	p, = ax[2].plot(theta, 4 .* binder_Ns[n, 1, :, 1, 3, 1], linewidth=0.8, markeredgewidth=1.5, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101)
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
end

ax[1].minorticks_on()
ax[2].minorticks_on()

ax[1].legend(loc="lower center", ncol=3, labelspacing=0.4, columnspacing=2, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1.25, 1.02)).set_zorder(110)

ax[1].set_xlabel(latexstring(L"$\theta$"))
ax[2].set_xlabel(latexstring(L"$\theta$"))
ax[1].set_ylabel(latexstring(L"$U\left[m^{x}_{\mathrm{sta}}\right]$"))
ax[2].set_ylabel(latexstring(L"$U\left[m^{z}_{\mathrm{uni}}\right]$"))

# ax[1].axvline(pi/4, color="k", linestyle="--", linewidth=0.8)
# ax[2].axvline(pi/4, color="k", linestyle="--", linewidth=0.8)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "binder.pdf"))
end

plt.close()