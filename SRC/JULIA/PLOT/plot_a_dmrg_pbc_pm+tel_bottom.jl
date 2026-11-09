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
simul_name = "PM+TEL/BOTTOM"

PBCs = true

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

Ns = []
chis = []
folders = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	push!(Ns, N)
	push!(chis, chi)
	push!(folders, folder)
end

chis_Ns = numpy.asarray([chis, Ns])
sort = numpy.lexsort(chis_Ns, axis=0) .+ 1

Ns = Ns[sort]
chis = chis[sort]
folders = folders[sort]

Ns_unique = numpy.unique(Ns)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[3],COLORS[4]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=[(0.5, (0, 2)), (2, (2, 2)), "-"], markersize=[10, 8, 6], color=cmap(numpy.linspace(0, 1, 3))))

fig, ax = plt.subplots(figsize=(2, 2.5))

for n in 1:length(Ns_unique)
	
	ind = numpy.where(Ns .== Ns_unique[n])[1] .+ 1

	chis_N = chis[ind]
	folders_N = folders[ind]

	magsX = numpy.zeros((length(chis_N), length(lam), length(theta), 6))
	magsXSPT = numpy.zeros((length(chis_N), length(lam), length(theta), 6))

	for x in 1:length(chis_N)
		
		folder = folders_N[x]

		data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = py"load_file"(data_folder, expression)

		for i in 1:length(data_filenames)

			data = 0
			data = load(joinpath(data_folder, data_filenames[i]))
			
			println("··· ", data_filenames[i])

			mX = data["mags"][1, 1, :]
			mY = data["mags"][1, 2, :]
			mZ = data["mags"][1, 3, :]

			magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
			magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l in 1:length(mX)])
			magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
			magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l in 1:length(mY)])
			magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
			magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l in 1:length(mZ)])

			mX = data["opSPT"][1, 1, :]
			mY = data["opSPT"][1, 2, :]
			mZ = data["opSPT"][1, 3, :]

			magsXSPT[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
			magsXSPT[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l in 1:length(mX)])
			magsXSPT[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
			magsXSPT[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l in 1:length(mY)])
			magsXSPT[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
			magsXSPT[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l in 1:length(mZ)])
		end
	end

	for l in 1:length(lam)
		for t in 1:length(theta)

			p, = ax.plot(chis_N, numpy.abs(magsX[:, l, t, 2])+numpy.abs(magsXSPT[:, l, t, 4]), marker="o", linewidth=1, zorder=105+n, label=latexstring(L"$2N=%$(Ns_unique[n])$"))
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		end
	end
end

ax.set_ylim([-0.1, 1.1])

ax.set_xticks([numpy.min(chis_Ns), 128, 256, numpy.max(chis_Ns)])
ax.set_yticks([0, 1])

ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(numpy.arange(32, 512, 32)))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(numpy.linspace(0, 1, 11)))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.text(
	0.03, 0.97,
	"(b)",
	transform=ax.transAxes,
	ha="left",
	va="top",
	fontsize=12)

ax.legend(
	loc="lower right",
	ncol=1,
	handlelength=1.5,
	bbox_to_anchor=(1.05, 0.57)).set_zorder(1001)

ax.set_xlabel(latexstring(L"$\chi$"))
ax.set_ylabel(latexstring(L"$|m^{x}_{\mathrm{sta}}|+|m^{\mathrm{SPT},y}_{\mathrm{sta}}|$"), labelpad=-5)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "op.pdf"))
	fig.clf()
end

plt.close()


