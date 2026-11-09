ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "LM+TML/TOP"

PBCs = true

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

fig, ax = plt.subplots(1, 2, figsize=(16, 3))

sort = numpy.argsort(Ns) .+ 1
folders = folders[sort]

handles = []

for folder in folders

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))
	
	mags = numpy.ones((length(lam), length(theta), 6))
	magsSPT = numpy.ones((length(lam), length(theta), 6))

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

		mX = data["opSPT"][1, 1, :]
		mY = data["opSPT"][1, 2, :]
		mZ = data["opSPT"][1, 3, :]

		magsSPT[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
		magsSPT[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l in 1:length(mX)])
		magsSPT[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
		magsSPT[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l in 1:length(mY)])
		magsSPT[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
		magsSPT[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l in 1:length(mZ)])
	end

	ind = numpy.where(lam .== round(3*pi/4; digits=15))[1] .+ 1

	p1, = ax[1].plot(theta, numpy.abs(mags[ind, :, 1])[1, :], markeredgewidth=1, linewidth=1, marker="x", markersize=4, color=COLORS[4], linestyle="-", zorder=105, label=latexstring(L"$|m^{x}_{\mathrm{uni}}|$"))
	p1.set_marker(matplotlib.markers.MarkerStyle(p1.get_marker(), capstyle="round"))

	p2, = ax[1].plot(theta, numpy.abs(mags[ind, :, 6])[1, :], markeredgewidth=1, linewidth=1, marker=".", markersize=6, color=COLORS[1], linestyle="-", zorder=105, label=latexstring(L"$|m^{z}_{\mathrm{sta}}|$"))
	p2.set_marker(matplotlib.markers.MarkerStyle(p2.get_marker(), capstyle="round"))

	p3, = ax[1].plot(theta, numpy.abs(mags[ind, :, 5])[1, :], markeredgewidth=1, linewidth=1, marker=".", markersize=6, color=COLORS[3], linestyle="--", zorder=105, label=latexstring(L"$|m^{z}_{\mathrm{uni}}|$"))
	p3.set_marker(matplotlib.markers.MarkerStyle(p3.get_marker(), capstyle="round"))

	ind = numpy.where(theta .== 0)[1] .+ 1

	l1, = ax[2].plot(lam, numpy.abs(mags[:, ind, 1])[:, 1], markeredgewidth=1, linewidth=1, marker="x", markersize=4, color=COLORS[4], linestyle="-", zorder=105, label=latexstring(L"$|m^{x}_{\mathrm{uni}}|$"))
	l1.set_marker(matplotlib.markers.MarkerStyle(l1.get_marker(), capstyle="round"))

	l2, = ax[2].plot(lam, numpy.abs(magsSPT[:, ind, 2])[:, 1], markeredgewidth=1, linewidth=1, marker="x", markersize=4, color=COLORS[6], linestyle="--", zorder=105, label=latexstring(L"$|m^{x}_{\mathrm{sta,SPT}}|$"))
	l2.set_marker(matplotlib.markers.MarkerStyle(l2.get_marker(), capstyle="round"))

	push!(handles, p1, l2, p2, p3)
end

ax[1].minorticks_on()
ax[2].minorticks_on()

ax[1].legend(handles=handles, loc="lower center", ncol=4, labelspacing=0.4, columnspacing=2, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1.1, 1.02)).set_zorder(110)

ax[1].set_xlabel(latexstring(L"$\theta$"))
ax[2].set_xlabel(latexstring(L"$\lambda$"))

ax[2].axvline(3*pi/4, color="k", linestyle="--", linewidth=0.8)

ax[1].set_title(latexstring(L"(a) $\lambda=3\pi/4$"), loc="left", pad=10)
ax[2].set_title(latexstring(L"(b) $\theta=0$"), loc="right", pad=10)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "op.pdf"))
end

plt.close()